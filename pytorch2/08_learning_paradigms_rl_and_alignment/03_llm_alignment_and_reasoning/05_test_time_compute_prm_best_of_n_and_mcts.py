"""Test Time Compute PRM Best of N and Mcts — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts
Golden weights: weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def prm_product_aggregation(step_probs: torch.Tensor, step_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Process Reward Model product aggregation over valid reasoning steps.
    
    R_prod(i) = exp( sum_{s=1}^S m_{i,s} * ln(p_{i,s} + eps) )
    """
    log_probs = torch.log(step_probs.clamp(min=eps)) * step_mask
    sum_log_probs = log_probs.sum(dim=-1)
    return torch.exp(sum_log_probs)


def prm_min_aggregation(step_probs: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
    """Process Reward Model min aggregation (weakest link principle).
    
    R_min(i) = min_{s: m_{i,s} = 1} p_{i,s}
    Invalid steps are replaced with 1.0 so they do not constrain the minimum.
    """
    masked_probs = torch.where(step_mask > 0.5, step_probs, torch.ones_like(step_probs))
    return masked_probs.min(dim=-1).values


def best_of_n_rerank(
    orm_scores: torch.Tensor,
    prm_scores: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combines outcome reward model (ORM) and process reward model (PRM) scores
    and identifies the top-ranked candidate.
    
    S_i = alpha * ORM_i + (1 - alpha) * PRM_i
    Returns (combined_scores, best_idx_scalar).
    """
    combined = alpha * orm_scores + (1.0 - alpha) * prm_scores
    best_idx = torch.argmax(combined).to(torch.float32).unsqueeze(0)
    return combined, best_idx


def puct_node_selection(
    q_values: torch.Tensor,
    priors: torch.Tensor,
    visit_counts: torch.Tensor,
    c_puct: float = 1.414,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AlphaZero-style Predictor Upper Confidence bounds for Trees (PUCT).
    
    U(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt( sum_b N(s, b) ) / (1.0 + N(s, a))
    Returns (puct_scores, selected_action_scalar).
    """
    total_visits = visit_counts.sum()
    exploration = c_puct * priors * torch.sqrt(total_visits) / (1.0 + visit_counts)
    puct_scores = q_values + exploration
    selected_action = torch.argmax(puct_scores).to(torch.float32).unsqueeze(0)
    return puct_scores, selected_action


def mcts_backpropagation(
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    action: int,
    leaf_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Incremental running-average backpropagation update along search trajectory.
    
    N'(s, a) = N(s, a) + 1
    Q'(s, a) = Q(s, a) + (V - Q(s, a)) / N'(s, a)
    """
    new_visits = visit_counts.clone()
    new_q = q_values.clone()
    new_visits[action] += 1.0
    new_q[action] += (leaf_value - new_q[action]) / new_visits[action]
    return new_q, new_visits


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # PRM inputs: batch B=4 rollouts, max S=5 steps
    step_probs = torch.tensor([
        [0.95, 0.90, 0.85, 0.92, 0.00],
        [0.80, 0.85, 0.40, 0.90, 0.95],
        [0.98, 0.96, 0.95, 0.97, 0.94],
        [0.50, 0.70, 0.00, 0.00, 0.00],
    ], dtype=torch.float32)
    step_mask = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    prm_prod_ref = prm_product_aggregation(step_probs, step_mask)
    prm_min_ref = prm_min_aggregation(step_probs, step_mask)

    # Best-of-N reranking
    orm_scores = torch.tensor([0.75, 0.45, 0.92, 0.30], dtype=torch.float32)
    combined_scores_ref, best_idx_ref = best_of_n_rerank(orm_scores, prm_min_ref, alpha=0.5)

    # PUCT MCTS selection (A = 5 candidate reasoning action branches)
    q_values = torch.tensor([0.65, 0.72, 0.40, 0.85, 0.55], dtype=torch.float32)
    priors = torch.tensor([0.10, 0.35, 0.05, 0.40, 0.10], dtype=torch.float32)
    visit_counts = torch.tensor([4.0, 12.0, 2.0, 18.0, 3.0], dtype=torch.float32)

    puct_scores_ref, selected_action_ref = puct_node_selection(q_values, priors, visit_counts, c_puct=1.414)

    # MCTS Backpropagation on selected action
    action_to_update = int(selected_action_ref.item())
    leaf_value = 0.90
    q_updated_ref, visits_updated_ref = mcts_backpropagation(q_values, visit_counts, action_to_update, leaf_value)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "step_probs": step_probs,
        "step_mask": step_mask,
        "prm_prod_ref": prm_prod_ref,
        "prm_min_ref": prm_min_ref,
        "orm_scores": orm_scores,
        "combined_scores_ref": combined_scores_ref,
        "best_idx_ref": best_idx_ref,
        "q_values": q_values,
        "priors": priors,
        "visit_counts": visit_counts,
        "puct_scores_ref": puct_scores_ref,
        "selected_action_ref": selected_action_ref,
        "q_updated_ref": q_updated_ref,
        "visits_updated_ref": visits_updated_ref,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
