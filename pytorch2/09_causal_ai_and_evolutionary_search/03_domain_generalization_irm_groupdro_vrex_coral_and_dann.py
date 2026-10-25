"""Domain Generalization IRM GroupDRO VREx CORAL and DANN — PyTorch 2.x reference implementation.

Topic: 09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann
Golden weights: weights/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.safetensors

``python pytorch2/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.py`` verifies against the golden checkpoint;
``python pytorch2/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def irm_v1_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Invariant Risk Minimization (IRMv1 - Arjovsky et al. 2019) gradient penalty.
    
    grad = 1/N sum_i [ logits_i * (sigmoid(logits_i) - targets_i) ]
    penalty = grad^2
    """
    probs = torch.sigmoid(logits)
    error = probs - targets
    grad = torch.mean(logits * error)
    return (grad * grad).unsqueeze(0)


def vrex_variance_penalty(env_losses: torch.Tensor) -> torch.Tensor:
    """Variance Risk Extrapolation (V-REx - Krueger et al. 2021) penalty across environments.
    
    penalty = 1/E sum_e (L_e - L_bar)^2
    """
    mean_loss = torch.mean(env_losses)
    diff = env_losses - mean_loss
    return torch.mean(diff * diff).unsqueeze(0)


def group_dro_update(
    env_losses: torch.Tensor,
    q_weights: torch.Tensor,
    eta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group Distributionally Robust Optimization (GroupDRO - Sagawa et al. 2020).
    
    loss = sum_e q_e * L_e
    q'_e = q_e * exp(eta * L_e) / sum_k [ q_k * exp(eta * L_k) ]
    """
    dro_loss = torch.sum(q_weights * env_losses).unsqueeze(0)
    unnormalized_q = q_weights * torch.exp(eta * env_losses)
    q_updated = unnormalized_q / torch.sum(unnormalized_q)
    return dro_loss, q_updated


def deep_coral_loss(phi_s: torch.Tensor, phi_t: torch.Tensor) -> torch.Tensor:
    """Deep CORAL (Sun & Saenko 2016) second-order covariance alignment loss.
    
    C = 1/(N - 1) * (Phi - mean(Phi))^T (Phi - mean(Phi))
    loss = 1 / (4 * d^2) * || C_s - C_t ||_F^2
    """
    ns, d = phi_s.shape
    nt, _ = phi_t.shape

    mean_s = torch.mean(phi_s, dim=0, keepdim=True)
    mean_t = torch.mean(phi_t, dim=0, keepdim=True)

    cent_s = phi_s - mean_s
    cent_t = phi_t - mean_t

    cov_s = (cent_s.t() @ cent_s) / float(ns - 1)
    cov_t = (cent_t.t() @ cent_t) / float(nt - 1)

    diff = cov_s - cov_t
    coral_loss = torch.sum(diff * diff) / (4.0 * float(d * d))
    return coral_loss.unsqueeze(0)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. IRMv1 inputs
    irm_logits = torch.tensor([[1.2], [-0.5], [0.8], [-1.4], [2.1], [-0.3], [0.4], [-1.1]], dtype=torch.float32)
    irm_targets = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)
    irm_penalty_ref = irm_v1_penalty(irm_logits, irm_targets)

    # 2. V-REx inputs across E=3 environments
    env_losses = torch.tensor([0.45, 0.82, 0.31], dtype=torch.float32)
    vrex_penalty_ref = vrex_variance_penalty(env_losses)

    # 3. GroupDRO inputs
    q_weights = torch.tensor([0.3333, 0.3333, 0.3334], dtype=torch.float32)
    dro_loss_ref, q_updated_ref = group_dro_update(env_losses, q_weights, eta=0.1)

    # 4. Deep CORAL inputs
    phi_source = torch.tensor([
        [1.1, -0.2, 0.5, 0.8],
        [0.4, 0.9, -0.3, 1.2],
        [-0.5, 0.1, 0.8, -0.4],
        [1.3, -0.8, 0.2, 0.6],
        [-0.2, 0.4, 1.1, -0.1],
        [0.8, 0.5, -0.6, 0.9],
        [-1.0, -0.3, 0.4, -0.5],
        [0.6, 0.7, 0.1, 1.0],
    ], dtype=torch.float32)

    phi_target = torch.tensor([
        [0.8, 0.1, 0.7, 0.5],
        [-0.1, 1.2, -0.1, 0.9],
        [-0.3, 0.4, 0.5, -0.2],
        [0.9, -0.5, 0.4, 0.3],
        [0.1, 0.6, 0.9, 0.2],
        [0.5, 0.8, -0.3, 0.7],
        [-0.7, 0.0, 0.2, -0.3],
        [0.4, 0.9, 0.3, 0.8],
    ], dtype=torch.float32)

    coral_loss_ref = deep_coral_loss(phi_source, phi_target)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "irm_logits": irm_logits,
        "irm_targets": irm_targets,
        "irm_penalty_ref": irm_penalty_ref,
        "env_losses": env_losses,
        "vrex_penalty_ref": vrex_penalty_ref,
        "q_weights": q_weights,
        "dro_loss_ref": dro_loss_ref,
        "q_updated_ref": q_updated_ref,
        "phi_source": phi_source,
        "phi_target": phi_target,
        "coral_loss_ref": coral_loss_ref,
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
