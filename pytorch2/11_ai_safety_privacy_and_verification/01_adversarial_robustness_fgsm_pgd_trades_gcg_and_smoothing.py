"""Adversarial Robustness FGSM PGD TRADES GCG and Smoothing — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing
Golden weights: weights/11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def fgsm_perturbation(
    x: torch.Tensor,
    grad: torch.Tensor,
    eps: float = 0.1,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2014).
    
    x_adv = clamp(x + eps * sign(grad), x_min, x_max)
    """
    x_adv = x + float(eps) * torch.sign(grad)
    return torch.clamp(x_adv, min=float(x_min), max=float(x_max))


def pgd_step(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    grad: torch.Tensor,
    alpha: float = 0.04,
    eps: float = 0.1,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> torch.Tensor:
    """Projected Gradient Descent single step in L_infinity ball (Madry et al., 2018).
    
    x' = clamp(x_adv + alpha * sign(grad), x_orig - eps, x_orig + eps)
    """
    step = x_adv + float(alpha) * torch.sign(grad)
    step_proj = torch.clamp(step, min=x_orig - float(eps), max=x_orig + float(eps))
    return torch.clamp(step_proj, min=float(x_min), max=float(x_max))


def trades_loss(
    clean_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """TRADES adversarial training regularized loss (Zhang et al., 2019).
    
    L = CE(clean, y) + beta * KL(P(clean) || P(adv))
    """
    ce_loss = torch.nn.functional.cross_entropy(clean_logits, labels.long())
    p_clean = torch.softmax(clean_logits, dim=-1)
    log_p_clean = torch.log_softmax(clean_logits, dim=-1)
    log_p_adv = torch.log_softmax(adv_logits, dim=-1)
    kl = torch.sum(p_clean * (log_p_clean - log_p_adv), dim=-1).mean()
    return ce_loss + float(beta) * kl


def randomized_smoothing_radius(
    p_top: torch.Tensor,
    p_runnerup: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Certified L2 robustness radius via Randomized Smoothing (Cohen et al., 2019).
    
    R = (sigma / 2) * (Phi^{-1}(p_top) - Phi^{-1}(p_runnerup))
    """
    sqrt2 = 1.4142135623730951
    probit_top = sqrt2 * torch.erfinv(2.0 * p_top - 1.0)
    probit_run = sqrt2 * torch.erfinv(2.0 * p_runnerup - 1.0)
    return (float(sigma) / 2.0) * (probit_top - probit_run)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. FGSM & PGD inputs: [4 samples, 3 input features]
    adv_x = torch.tensor([
        [0.5, 0.2, 0.8],
        [0.1, 0.9, 0.4],
        [0.3, 0.7, 0.2],
        [0.6, 0.5, 0.5],
    ], dtype=torch.float32)
    adv_grad = torch.tensor([
        [0.8, -0.4, 0.1],
        [-0.5, 0.2, -0.9],
        [0.3, -0.6, 0.4],
        [-0.2, 0.1, -0.3],
    ], dtype=torch.float32)

    fgsm_adv_ref = fgsm_perturbation(adv_x, adv_grad, eps=0.1, x_min=0.0, x_max=1.0)
    pgd_step_ref = pgd_step(fgsm_adv_ref, adv_x, adv_grad, alpha=0.04, eps=0.1, x_min=0.0, x_max=1.0)

    # 2. TRADES inputs: [4 samples, 3 classes]
    clean_logits = torch.tensor([
        [2.1, 0.5, -1.2],
        [-0.8, 1.9, 0.2],
        [0.3, -0.5, 2.4],
        [1.5, 1.2, -0.8],
    ], dtype=torch.float32)
    adv_logits = torch.tensor([
        [1.6, 0.9, -0.8],
        [-0.3, 1.4, 0.6],
        [0.1, -0.2, 2.0],
        [1.1, 1.4, -0.5],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.float32)
    trades_loss_ref = trades_loss(clean_logits, adv_logits, labels, beta=1.0)

    # 3. Randomized Smoothing inputs: [4 samples]
    p_top = torch.tensor([0.85, 0.92, 0.78, 0.95], dtype=torch.float32)
    p_runnerup = torch.tensor([0.10, 0.05, 0.15, 0.03], dtype=torch.float32)
    smoothing_radius_ref = randomized_smoothing_radius(p_top, p_runnerup, sigma=0.25)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "adv_x": adv_x,
        "adv_grad": adv_grad,
        "fgsm_adv_ref": fgsm_adv_ref,
        "pgd_step_ref": pgd_step_ref,
        "clean_logits": clean_logits,
        "adv_logits": adv_logits,
        "labels": labels,
        "trades_loss_ref": trades_loss_ref,
        "p_top": p_top,
        "p_runnerup": p_runnerup,
        "smoothing_radius_ref": smoothing_radius_ref,
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
