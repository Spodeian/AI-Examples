"""Losses: Regression, Classification, and Barron Adaptive Robust (C03 Topic 07).

Implements fundamental and modern loss formulations:
1. Barron's General Adaptive Robust Loss: Continuous parameterized interpolation between L2, Charbonnier, Cauchy, Geman-McClure, and Welsch (Barron 2019).
2. Huber Loss: Smooth L1 piecewise regression robust against outlier gradients (Huber 1964).
3. Focal Loss: Dynamically scaled cross-entropy down-weighting well-classified examples (Lin et al. 2017).
4. PolyLoss: Taylor expansion framework fine-tuning polynomial coefficients of cross-entropy (Leng et al. 2022).
5. Asymmetric Loss (ASL): Decoupled positive/negative focusing with probability margin shift for extreme multi-label imbalance (Ridnik et al. 2021).
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file


def compute_barron_robust_loss(
    r: Float[torch.Tensor, "..."],
    alpha: float = 1.0,
    c: float = 1.5,
) -> torch.Tensor:
    """Barron's General Adaptive Robust Loss (Barron 2019).

    rho(r, alpha, c) = (|alpha - 2| / alpha) * ( (( (r/c)^2 / |alpha - 2| + 1 )^(alpha/2)) - 1 )
    """
    d_alpha = abs(alpha - 2.0)
    scaled_sq = (r / c).pow(2)
    elem_loss = (d_alpha / alpha) * (torch.pow(scaled_sq / d_alpha + 1.0, alpha / 2.0) - 1.0)
    return elem_loss.mean()


def compute_huber_loss(
    r: Float[torch.Tensor, "..."],
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber Loss (Smooth L1 piecewise regression)."""
    abs_r = torch.abs(r)
    elem_loss = torch.where(abs_r <= delta, 0.5 * r.pow(2), delta * (abs_r - 0.5 * delta))
    return elem_loss.mean()


def compute_focal_loss(
    logits: Float[torch.Tensor, "batch num_classes"],
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Multi-Class Focal Loss (Lin et al. 2017): -(1 - p_t)^gamma * log(p_t)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    batch_size = logits.shape[0]
    target_log_probs = log_probs[torch.arange(batch_size), targets]
    target_probs = probs[torch.arange(batch_size), targets]
    elem_loss = -(1.0 - target_probs).pow(gamma) * target_log_probs
    return elem_loss.mean()


def compute_polyloss(
    logits: Float[torch.Tensor, "batch num_classes"],
    targets: torch.Tensor,
    epsilon_1: float = 1.0,
) -> torch.Tensor:
    """Poly-1 Loss (Leng et al. 2022): -log(p_t) + epsilon_1 * (1 - p_t)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    batch_size = logits.shape[0]
    target_log_probs = log_probs[torch.arange(batch_size), targets]
    target_probs = probs[torch.arange(batch_size), targets]
    elem_loss = -target_log_probs + epsilon_1 * (1.0 - target_probs)
    return elem_loss.mean()


def compute_asymmetric_loss(
    logits: Float[torch.Tensor, "batch num_labels"],
    targets: Float[torch.Tensor, "batch num_labels"],
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip_margin: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Asymmetric Loss (ASL) for multi-label classification (Ridnik et al. 2021)."""
    probs = torch.sigmoid(logits)
    probs_m = torch.clamp(probs - clip_margin, min=0.0)

    pos_loss = -targets * torch.pow(1.0 - probs, gamma_pos) * torch.log(probs.clamp(min=eps))
    neg_loss = -(1.0 - targets) * torch.pow(probs_m, gamma_neg) * torch.log((1.0 - probs_m).clamp(min=eps))
    return (pos_loss + neg_loss).mean()


class LossesUnifiedModule(nn.Module):
    """Unified module combining loss operators with a linear anchor."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[LossesUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = LossesUnifiedModule(in_dim=8, out_dim=4)

    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])

    return module, weights


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/07_losses_regression_classification_and_barron_robust.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    module, weights = load_checkpoint(str(weights_path))
    module.eval()

    test_in = weights["test_input"]
    r_barron = weights["r_barron"]
    alpha_b = float(weights["alpha_barron"][0])
    c_b = float(weights["c_barron"][0])

    r_huber = weights["r_huber"]
    delta_h = float(weights["delta_huber"][0])

    logits_clf = weights["logits_clf"]
    targets_clf = weights["targets_clf"].to(torch.int64)
    gamma_f = float(weights["gamma_focal"][0])
    eps_p = float(weights["eps_poly"][0])

    logits_asl = weights["logits_asl"]
    targets_asl = weights["targets_asl"]
    gamma_p = float(weights["gamma_pos"][0])
    gamma_n = float(weights["gamma_neg"][0])
    margin_asl = float(weights["margin_asl"][0])

    with torch.no_grad():
        # 1. Linear Anchor Parity
        y_anchor = module(test_in)
        diff_anchor = (y_anchor - weights["test_output"]).abs().max().item()
        assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

        # 2. Barron Robust Loss Parity
        loss_b = compute_barron_robust_loss(r_barron, alpha_b, c_b)
        diff_b = (loss_b - weights["loss_barron"][0]).abs().item()
        assert diff_b < 1e-5, f"Barron loss mismatch: {diff_b:.2e}"

        # 3. Huber Loss Parity
        loss_h = compute_huber_loss(r_huber, delta_h)
        diff_h = (loss_h - weights["loss_huber"][0]).abs().item()
        assert diff_h < 1e-5, f"Huber loss mismatch: {diff_h:.2e}"

        # 4. Focal Loss Parity
        loss_f = compute_focal_loss(logits_clf, targets_clf, gamma_f)
        diff_f = (loss_f - weights["loss_focal"][0]).abs().item()
        assert diff_f < 1e-5, f"Focal loss mismatch: {diff_f:.2e}"

        # 5. PolyLoss Parity
        loss_poly = compute_polyloss(logits_clf, targets_clf, eps_p)
        diff_poly = (loss_poly - weights["loss_poly"][0]).abs().item()
        assert diff_poly < 1e-5, f"PolyLoss mismatch: {diff_poly:.2e}"

        # 6. Asymmetric Loss Parity
        loss_asl = compute_asymmetric_loss(logits_asl, targets_asl, gamma_p, gamma_n, margin_asl)
        diff_asl = (loss_asl - weights["loss_asl"][0]).abs().item()
        assert diff_asl < 1e-5, f"ASL mismatch: {diff_asl:.2e}"

    print(
        f"All PyTorch assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"Barron: {diff_b:.2e}, Huber: {diff_h:.2e}, Focal: {diff_f:.2e}, "
        f"Poly: {diff_poly:.2e}, ASL: {diff_asl:.2e})"
    )


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/03_building_blocks_and_losses/07_losses_regression_classification_and_barron_robust.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "03_building_blocks_and_losses/07_losses_regression_classification_and_barron_robust"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
