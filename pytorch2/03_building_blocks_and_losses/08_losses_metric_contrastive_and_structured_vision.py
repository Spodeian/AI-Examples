"""Losses: Metric, Contrastive, and Structured Vision (C03 Topic 08).

Implements fundamental metric, contrastive, and vision objective formulations:
1. InfoNCE Loss: Temperature-scaled normalized cross-entropy for contrastive representation learning (Oord et al. 2018).
2. SigLIP Loss: Pairwise sigmoid BCE loss eliminating global all-gather softmax normalization (Zhai et al. 2023).
3. ArcFace Loss: Additive Angular Margin loss on hypersphere cos(theta + m) (Deng et al. 2019).
4. Soft-Dice Loss: Continuous relaxation of the Dice/F1 coefficient for semantic segmentation (Milletari et al. 2016).
5. Generalized IoU (GIoU) Loss: Scale-invariant bounding box regression addressing empty intersection (Rezatofighi et al. 2019).
6. Linear Anchor: Standard linear mapping for cross-track structural baseline.
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file


def compute_infonce_loss(
    q: Float[torch.Tensor, "batch dim"],
    k: Float[torch.Tensor, "batch dim"],
    tau: float = 0.07,
) -> torch.Tensor:
    """InfoNCE Loss with L2 normalization."""
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    logits = torch.matmul(q_norm, k_norm.t()) / tau
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)


def compute_siglip_loss(
    q: Float[torch.Tensor, "batch dim"],
    k: Float[torch.Tensor, "batch dim"],
    scale: float = 10.0,
    bias: float = -10.0,
) -> torch.Tensor:
    """SigLIP Loss: Pairwise sigmoid BCE over text-image similarity matrix."""
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    logits = torch.matmul(q_norm, k_norm.t()) * scale + bias
    b = q.shape[0]
    labels = 2.0 * torch.eye(b, device=q.device) - 1.0
    loss = F.softplus(-labels * logits)
    return loss.mean()


def compute_arcface_loss(
    embeddings: Float[torch.Tensor, "batch dim"],
    weights: Float[torch.Tensor, "num_classes dim"],
    targets: torch.Tensor,
    s: float = 30.0,
    m: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """ArcFace Loss: Additive Angular Margin in cos-space."""
    emb_norm = F.normalize(embeddings, p=2, dim=-1)
    w_norm = F.normalize(weights, p=2, dim=-1)
    cos_theta = torch.matmul(emb_norm, w_norm.t()).clamp(-1.0 + eps, 1.0 - eps)

    sin_theta = torch.sqrt((1.0 - cos_theta.pow(2)).clamp(min=eps))
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

    one_hot = F.one_hot(targets, num_classes=weights.shape[0]).float()
    logits = s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
    return F.cross_entropy(logits, targets)


def compute_soft_dice_loss(
    probs: Float[torch.Tensor, "..."],
    targets: Float[torch.Tensor, "..."],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice Loss: 1 - (2 * sum(p * y) + eps) / (sum(p^2) + sum(y^2) + eps)."""
    p_flat = probs.reshape(-1)
    t_flat = targets.reshape(-1)
    intersection = (p_flat * t_flat).sum()
    denominator = p_flat.pow(2).sum() + t_flat.pow(2).sum()
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def compute_giou_loss(
    boxes1: Float[torch.Tensor, "batch 4"],
    boxes2: Float[torch.Tensor, "batch 4"],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Generalized IoU (GIoU) Loss."""
    x1_i = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1_i = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2_i = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2_i = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter_area = (x2_i - x1_i).clamp(min=0.0) * (y2_i - y1_i).clamp(min=0.0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area

    x1_c = torch.min(boxes1[:, 0], boxes2[:, 0])
    y1_c = torch.min(boxes1[:, 1], boxes2[:, 1])
    x2_c = torch.max(boxes1[:, 2], boxes2[:, 2])
    y2_c = torch.max(boxes1[:, 3], boxes2[:, 3])
    c_area = (x2_c - x1_c).clamp(min=0.0) * (y2_c - y1_c).clamp(min=0.0) + eps

    giou = iou - (c_area - union_area) / c_area
    return (1.0 - giou).mean()


class MetricVisionUnifiedModule(nn.Module):
    """Unified module combining metric, contrastive, and vision losses with a linear anchor."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[MetricVisionUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = MetricVisionUnifiedModule(in_dim=8, out_dim=4)

    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])

    return module, weights


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/08_losses_metric_contrastive_and_structured_vision.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    module, weights = load_checkpoint(str(weights_path))
    module.eval()

    test_in = weights["test_input"]
    q_repr = weights["q_repr"]
    k_repr = weights["k_repr"]
    tau = float(weights["tau_val"][0])

    siglip_scale = float(weights["siglip_scale"][0])
    siglip_bias = float(weights["siglip_bias"][0])

    arc_emb = weights["arc_emb"]
    arc_w = weights["arc_weights"]
    arc_targets = weights["arc_targets"].to(torch.int64)
    arc_s = float(weights["arc_s"][0])
    arc_m = float(weights["arc_m"][0])

    dice_probs = weights["dice_probs"]
    dice_targets = weights["dice_targets"]

    boxes1 = weights["boxes1"]
    boxes2 = weights["boxes2"]

    with torch.no_grad():
        # 1. Linear Anchor Parity
        y_anchor = module(test_in)
        diff_anchor = (y_anchor - weights["test_output"]).abs().max().item()
        assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

        # 2. InfoNCE Loss Parity
        loss_infonce = compute_infonce_loss(q_repr, k_repr, tau)
        diff_infonce = (loss_infonce - weights["loss_infonce"][0]).abs().item()
        assert diff_infonce < 1e-5, f"InfoNCE mismatch: {diff_infonce:.2e}"

        # 3. SigLIP Loss Parity
        loss_siglip = compute_siglip_loss(q_repr, k_repr, siglip_scale, siglip_bias)
        diff_siglip = (loss_siglip - weights["loss_siglip"][0]).abs().item()
        assert diff_siglip < 1e-5, f"SigLIP mismatch: {diff_siglip:.2e}"

        # 4. ArcFace Loss Parity
        loss_arcface = compute_arcface_loss(arc_emb, arc_w, arc_targets, arc_s, arc_m)
        diff_arcface = (loss_arcface - weights["loss_arcface"][0]).abs().item()
        assert diff_arcface < 1e-5, f"ArcFace mismatch: {diff_arcface:.2e}"

        # 5. Soft-Dice Loss Parity
        loss_dice = compute_soft_dice_loss(dice_probs, dice_targets)
        diff_dice = (loss_dice - weights["loss_dice"][0]).abs().item()
        assert diff_dice < 1e-5, f"Soft-Dice mismatch: {diff_dice:.2e}"

        # 6. GIoU Loss Parity
        loss_giou = compute_giou_loss(boxes1, boxes2)
        diff_giou = (loss_giou - weights["loss_giou"][0]).abs().item()
        assert diff_giou < 1e-5, f"GIoU mismatch: {diff_giou:.2e}"

    print(
        f"All PyTorch assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"InfoNCE: {diff_infonce:.2e}, SigLIP: {diff_siglip:.2e}, ArcFace: {diff_arcface:.2e}, "
        f"Dice: {diff_dice:.2e}, GIoU: {diff_giou:.2e})"
    )


if __name__ == "__main__":
    main()
