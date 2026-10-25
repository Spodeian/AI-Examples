"""Offline RL Imitation Cql IQL Gail and Decision Transformer — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def cql_penalty(q_values: torch.Tensor, dataset_actions: torch.Tensor) -> torch.Tensor:
    """Conservative Q-Learning (CQL) conservative gap penalty:
    L_cql = mean( logsumexp(Q(s, a)) - Q(s, a_data) )
    q_values: [B, A]
    dataset_actions: [B] integer action indices
    Returns: scalar penalty.
    """
    logsumexp_q = torch.logsumexp(q_values, dim=-1)  # [B]
    data_q = torch.gather(q_values, dim=-1, index=dataset_actions.unsqueeze(-1)).squeeze(-1)  # [B]
    return torch.mean(logsumexp_q - data_q)


def iql_expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """Implicit Q-Learning (IQL) asymmetric expectile regression loss:
    L_2^tau(u) = |tau - I(u < 0)| * u^2
    diff: [B] (Q(s, a) - V(s))
    tau: expectile parameter in (0.5, 1.0)
    Returns: scalar loss.
    """
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return torch.mean(weight * (diff ** 2))


def iql_actor_loss(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    adv: torch.Tensor,
    beta: float = 3.0,
    max_weight: float = 100.0,
) -> torch.Tensor:
    """Advantage-Weighted Regression (AWR) actor extraction in IQL:
    w = clamp(exp(beta * adv), max=max_weight)
    L_actor = mean( w * ||pred_action - target_action||^2 )
    pred_action: [B, D]
    target_action: [B, D]
    adv: [B] (Q(s, a) - V(s))
    Returns: scalar loss.
    """
    weights = torch.clamp(torch.exp(beta * adv), max=max_weight)  # [B]
    sq_err = torch.sum((pred_action - target_action) ** 2, dim=-1)  # [B]
    return torch.mean(weights * sq_err)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. CQL Penalty
    q_values = torch.randn(8, 5) * 2.0
    dataset_actions = torch.randint(0, 5, (8,))
    cql_penalty_ref = cql_penalty(q_values, dataset_actions)

    # 2. IQL Expectile Loss
    diff = torch.randn(8) * 1.5
    iql_expectile_ref = iql_expectile_loss(diff, tau=0.7)

    # 3. IQL Actor Loss
    pred_action = torch.randn(8, 3)
    target_action = torch.randn(8, 3)
    adv = torch.randn(8) * 0.8
    iql_actor_ref = iql_actor_loss(pred_action, target_action, adv, beta=3.0, max_weight=100.0)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "q_values": q_values,
        "dataset_actions": dataset_actions,
        "cql_penalty_ref": cql_penalty_ref,
        "diff": diff,
        "iql_expectile_ref": iql_expectile_ref,
        "pred_action": pred_action,
        "target_action": target_action,
        "adv": adv,
        "iql_actor_ref": iql_actor_ref,
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
