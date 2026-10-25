"""Continual Learning EWC Synaptic Intelligence Lwf and Replay — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay
Golden weights: weights/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ewc_penalty(
    theta: torch.Tensor,
    theta_star: torch.Tensor,
    fisher_diag: torch.Tensor,
    lambda_ewc: float = 10.0,
) -> torch.Tensor:
    """Elastic Weight Consolidation quadratic penalty: 0.5 * lambda * sum(F_i * (theta_i - theta_star_i)^2)."""
    penalty = 0.5 * lambda_ewc * torch.sum(fisher_diag * (theta - theta_star) ** 2)
    return penalty.unsqueeze(0)


def synaptic_intelligence_omega(
    delta_theta: torch.Tensor,
    path_integral: torch.Tensor,
    xi: float = 0.1,
) -> torch.Tensor:
    """Synaptic Intelligence importance parameter: max(0, omega_k) / (Delta theta_k^2 + xi)."""
    return torch.clamp(path_integral, min=0.0) / (delta_theta ** 2 + xi)


def lwf_distillation_loss(
    new_logits: torch.Tensor,
    old_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Learning without Forgetting (LwF) knowledge distillation loss under temperature scaling."""
    p_old = F.softmax(old_logits / temperature, dim=-1)
    log_p_new = F.log_softmax(new_logits / temperature, dim=-1)
    log_p_old = F.log_softmax(old_logits / temperature, dim=-1)
    kl = (p_old * (log_p_old - log_p_new)).sum(dim=-1).mean()
    return (temperature * temperature * kl).unsqueeze(0)


def der_dark_replay_loss(
    current_logits: torch.Tensor,
    past_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    """Dark Experience Replay (DER++) loss: alpha * MSE(current, past) + beta * CE(current, labels)."""
    mse_term = F.mse_loss(current_logits, past_logits)
    ce_term = F.cross_entropy(current_logits, labels)
    return (alpha * mse_term + beta * ce_term).unsqueeze(0)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Linear anchor
    test_input = torch.randn(4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    test_output = linear_anchor(test_input, weight, bias)

    # 2. EWC
    theta = torch.randn(8)
    theta_star = torch.randn(8)
    fisher_diag = torch.abs(torch.randn(8)) + 0.1
    ewc_penalty_ref = ewc_penalty(theta, theta_star, fisher_diag, lambda_ewc=10.0)

    # 3. SI
    delta_theta = torch.randn(8) * 0.2
    path_integral = torch.randn(8)
    si_omega_ref = synaptic_intelligence_omega(delta_theta, path_integral, xi=0.1)

    # 4. LwF
    new_logits = torch.randn(4, 4)
    old_logits = torch.randn(4, 4)
    lwf_loss_ref = lwf_distillation_loss(new_logits, old_logits, temperature=2.0)

    # 5. DER++
    current_logits = torch.randn(4, 4)
    past_logits = torch.randn(4, 4)
    labels = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
    der_loss_ref = der_dark_replay_loss(current_logits, past_logits, labels, alpha=0.5, beta=0.5)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "theta": theta,
        "theta_star": theta_star,
        "fisher_diag": fisher_diag,
        "ewc_penalty_ref": ewc_penalty_ref,
        "delta_theta": delta_theta,
        "path_integral": path_integral,
        "si_omega_ref": si_omega_ref,
        "new_logits": new_logits,
        "old_logits": old_logits,
        "lwf_loss_ref": lwf_loss_ref,
        "current_logits": current_logits,
        "past_logits": past_logits,
        "labels": labels,
        "der_loss_ref": der_loss_ref,
    }


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
