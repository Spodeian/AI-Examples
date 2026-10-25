"""Population Based Training PBT and DARTS Neural Architecture Search — PyTorch 2.x reference implementation.

Topic: 09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search
Golden weights: weights/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.safetensors

``python pytorch2/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.py`` verifies against the golden checkpoint;
``python pytorch2/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def pbt_exploit_explore_step(
    scores: torch.Tensor,
    weights_pop: torch.Tensor,
    hyperparams_pop: torch.Tensor,
    perturb_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Population-Based Training exploit-and-explore step.
    
    Identifies top and bottom performers:
    Bottom performer copies weights and hyperparameters from top performer,
    then mutates hyperparameters by perturbation factors.
    """
    top_idx = torch.argmax(scores).item()
    bot_idx = torch.argmin(scores).item()

    w_new = weights_pop.clone()
    h_new = hyperparams_pop.clone()

    w_new[bot_idx] = weights_pop[top_idx]
    h_new[bot_idx] = hyperparams_pop[top_idx] * perturb_factors

    return w_new, h_new


def darts_mixed_op_forward(
    alpha: torch.Tensor,
    candidate_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DARTS continuous relaxation over candidate operations for each edge.
    
    alpha: [E, M]
    candidate_outputs: [E, M, B, C]
    mixed_outputs: [E, B, C]
    softmax_weights: [E, M]
    """
    weights = torch.softmax(alpha, dim=-1)  # [E, M]
    w_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # [E, M, 1, 1]
    mixed_outputs = torch.sum(w_expanded * candidate_outputs, dim=1)  # [E, B, C]
    return mixed_outputs, weights


def darts_discrete_decode(
    alpha: torch.Tensor,
    candidate_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretely decode DARTS architecture by selecting argmax operation per edge."""
    selected_ops = torch.argmax(alpha, dim=-1)  # [E]
    e_indices = torch.arange(alpha.shape[0])
    discrete_outputs = candidate_outputs[e_indices, selected_ops]  # [E, B, C]
    return selected_ops.float(), discrete_outputs


def darts_architecture_entropy(softmax_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute entropy of architecture parameter distribution H(w) = -sum w log(w)."""
    return -torch.sum(softmax_weights * torch.log(softmax_weights + eps))


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. PBT setup (P=4 workers, D=4 weight dims, M=2 hyperparameters [lr, wd])
    pbt_scores = torch.tensor([12.5, 45.0, 8.2, 38.0], dtype=torch.float32)
    pbt_weights = torch.tensor([
        [0.1, -0.2, 0.3, 0.4],
        [1.0, 1.2, -0.8, 0.5],
        [-0.5, 0.2, 0.1, -0.3],
        [0.8, 0.9, -0.4, 0.2],
    ], dtype=torch.float32)
    pbt_hyperparams = torch.tensor([
        [0.01, 1e-4],
        [0.05, 5e-4],
        [0.001, 1e-3],
        [0.02, 2e-4],
    ], dtype=torch.float32)
    pbt_perturb = torch.tensor([1.2, 0.8], dtype=torch.float32)

    pbt_w_up_ref, pbt_h_up_ref = pbt_exploit_explore_step(
        pbt_scores, pbt_weights, pbt_hyperparams, pbt_perturb
    )

    # 2. DARTS setup (E=2 edges, M=4 operations, B=4 batch, C=4 channels)
    darts_alpha = torch.tensor([
        [0.5, 2.1, -0.3, 1.4],
        [1.8, 0.2, -1.0, 0.9],
    ], dtype=torch.float32)

    darts_candidate_ops = torch.randn(2, 4, 4, 4)

    darts_mixed_out_ref, darts_softmax_weights_ref = darts_mixed_op_forward(
        darts_alpha, darts_candidate_ops
    )
    darts_selected_ops_ref, darts_discrete_out_ref = darts_discrete_decode(
        darts_alpha, darts_candidate_ops
    )
    darts_entropy_ref = darts_architecture_entropy(darts_softmax_weights_ref)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "pbt_scores": pbt_scores,
        "pbt_weights": pbt_weights,
        "pbt_hyperparams": pbt_hyperparams,
        "pbt_perturb": pbt_perturb,
        "pbt_weights_updated_ref": pbt_w_up_ref,
        "pbt_hyperparams_updated_ref": pbt_h_up_ref,
        "darts_alpha": darts_alpha,
        "darts_candidate_ops": darts_candidate_ops,
        "darts_mixed_out_ref": darts_mixed_out_ref,
        "darts_softmax_weights_ref": darts_softmax_weights_ref,
        "darts_selected_ops_ref": darts_selected_ops_ref,
        "darts_discrete_out_ref": darts_discrete_out_ref,
        "darts_entropy_ref": darts_entropy_ref,
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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
