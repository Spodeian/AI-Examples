"""OOD Detection Abstention Conformal and Algorithmic Fairness — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness
Golden weights: weights/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def energy_ood_score(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Helmholtz free energy OOD score (Liu et al. 2020): E(x) = -T * logsumexp(logits / T)."""
    return -float(temperature) * torch.logsumexp(logits / float(temperature), dim=-1)


def split_conformal_quantile(cal_scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Split Conformal Prediction (Vovk et al. 2005; Angelopoulos & Bates 2021) 1 - alpha quantile."""
    sorted_scores, _ = torch.sort(cal_scores)
    n = sorted_scores.shape[0]
    idx = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - float(alpha))) - 1))
    return sorted_scores[idx]


def selective_net_loss(
    ce_losses: torch.Tensor,
    selection_probs: torch.Tensor,
    target_coverage: float,
    penalty_weight: float,
) -> torch.Tensor:
    """SelectiveNet risk with coverage penalty constraint (Geifman & El-Yaniv 2019)."""
    phi_hat = torch.mean(selection_probs)
    empirical_risk = torch.sum(ce_losses * selection_probs) / (torch.sum(selection_probs) + 1e-8)
    coverage_deficit = torch.clamp(float(target_coverage) - phi_hat, min=0.0)
    penalty = float(penalty_weight) * (coverage_deficit ** 2)
    return empirical_risk + penalty


def equalized_odds_discrepancy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    sensitive_attr: torch.Tensor,
) -> torch.Tensor:
    """Equalized Odds discrepancy (Hardt et al. 2016): |TPR_0 - TPR_1| + |FPR_0 - FPR_1|."""
    mask_a0 = sensitive_attr == 0.0
    mask_a1 = sensitive_attr == 1.0
    mask_y1 = y_true == 1.0
    mask_y0 = y_true == 0.0

    tpr_0 = torch.sum(y_pred * mask_a0 * mask_y1) / (torch.sum(mask_a0 * mask_y1) + 1e-8)
    tpr_1 = torch.sum(y_pred * mask_a1 * mask_y1) / (torch.sum(mask_a1 * mask_y1) + 1e-8)
    fpr_0 = torch.sum(y_pred * mask_a0 * mask_y0) / (torch.sum(mask_a0 * mask_y0) + 1e-8)
    fpr_1 = torch.sum(y_pred * mask_a1 * mask_y0) / (torch.sum(mask_a1 * mask_y0) + 1e-8)

    return torch.abs(tpr_0 - tpr_1) + torch.abs(fpr_0 - fpr_1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Energy OOD: B=4, C=6
    logits = torch.tensor([
        [4.2, 1.1, 0.3, -1.0, 0.5, 0.2],
        [0.2, 0.1, -0.1, 0.0, 0.3, -0.2],
        [8.5, 2.1, 1.0, -0.5, 0.0, 1.2],
        [-1.0, -2.0, -0.5, -1.2, -0.8, -1.5],
    ])
    energy_ref = energy_ood_score(logits, temperature=1.0)

    # 2. Conformal quantile: n=10, alpha=0.1
    cal_scores = torch.tensor([0.12, 0.25, 0.31, 0.44, 0.52, 0.59, 0.68, 0.73, 0.82, 0.91])
    q_hat_ref = split_conformal_quantile(cal_scores, alpha=0.1)

    # 3. SelectiveNet: B=6
    ce_losses = torch.tensor([0.2, 1.5, 0.1, 2.8, 0.4, 0.05])
    selection_probs = torch.tensor([0.95, 0.10, 0.99, 0.05, 0.90, 0.98])
    selective_loss_ref = selective_net_loss(ce_losses, selection_probs, target_coverage=0.7, penalty_weight=5.0)

    # 4. Equalized Odds: B=8
    y_pred = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    y_true = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    sensitive_attr = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    eo_discrepancy_ref = equalized_odds_discrepancy(y_pred, y_true, sensitive_attr)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "logits": logits,
        "energy_ref": energy_ref,
        "cal_scores": cal_scores,
        "q_hat_ref": q_hat_ref,
        "ce_losses": ce_losses,
        "selection_probs": selection_probs,
        "selective_loss_ref": selective_loss_ref,
        "y_pred": y_pred,
        "y_true": y_true,
        "sensitive_attr": sensitive_attr,
        "eo_discrepancy_ref": eo_discrepancy_ref,
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
