"""Losses: Metric, Contrastive, and Structured Vision in JAX / Functional NumPy (C03 Topic 08).

Pure functional implementations with transitive parity:
1. InfoNCE Loss: Normalized temperature-scaled cross-entropy.
2. SigLIP Loss: Pairwise sigmoid binary cross-entropy.
3. ArcFace Loss: Additive Angular Margin in cos-space.
4. Soft-Dice Loss: Continuous relaxation for segmentation masks.
5. Generalized IoU (GIoU) Loss: Bounding box regression with convex hull penalty.
6. Linear Anchor: W @ x + b.
"""

from pathlib import Path
from typing import Dict, Tuple
import math
import numpy as np
from safetensors.numpy import load_file


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.sqrt(np.maximum(np.sum(x ** 2, axis=axis, keepdims=True), eps))
    return x / norm


def log_sum_exp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    max_val = np.max(x, axis=axis, keepdims=True)
    lse = max_val + np.log(np.sum(np.exp(x - max_val), axis=axis, keepdims=True))
    if not keepdims:
        lse = np.squeeze(lse, axis=axis)
    return lse


def linear_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.dot(x, weight) + bias


def infonce_loss_forward(
    q: np.ndarray,
    k: np.ndarray,
    tau: float = 0.07,
) -> np.ndarray:
    """InfoNCE Loss forward in pure functional NumPy."""
    q_norm = l2_normalize(q, axis=-1)
    k_norm = l2_normalize(k, axis=-1)
    logits = np.dot(q_norm, k_norm.T) / tau
    b = q.shape[0]

    lse = log_sum_exp(logits, axis=-1)
    diag_logits = np.diag(logits)
    losses = -(diag_logits - lse)
    return np.mean(losses)


def siglip_loss_forward(
    q: np.ndarray,
    k: np.ndarray,
    scale: float = 10.0,
    bias: float = -10.0,
) -> np.ndarray:
    """SigLIP Loss forward."""
    q_norm = l2_normalize(q, axis=-1)
    k_norm = l2_normalize(k, axis=-1)
    logits = np.dot(q_norm, k_norm.T) * scale + bias
    b = q.shape[0]
    labels = 2.0 * np.eye(b) - 1.0

    # softplus(-y * z) = log(1 + exp(-y * z))
    z = -labels * logits
    # Numerically stable softplus: max(0, z) + log1p(exp(-abs(z)))
    losses = np.maximum(0.0, z) + np.log1p(np.exp(-np.abs(z)))
    return np.mean(losses)


def arcface_loss_forward(
    embeddings: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    s: float = 30.0,
    m: float = 0.5,
    eps: float = 1e-7,
) -> np.ndarray:
    """ArcFace Loss forward."""
    emb_norm = l2_normalize(embeddings, axis=-1)
    w_norm = l2_normalize(weights, axis=-1)
    cos_theta = np.clip(np.dot(emb_norm, w_norm.T), -1.0 + eps, 1.0 - eps)

    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, eps, None))
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

    num_classes = weights.shape[0]
    batch_size = embeddings.shape[0]
    one_hot = np.eye(num_classes)[targets]
    logits = s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)

    lse = log_sum_exp(logits, axis=-1)
    target_logits = logits[np.arange(batch_size), targets]
    losses = -(target_logits - lse)
    return np.mean(losses)


def soft_dice_loss_forward(
    probs: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Soft Dice Loss forward."""
    p_flat = probs.reshape(-1)
    t_flat = targets.reshape(-1)
    intersection = np.sum(p_flat * t_flat)
    denominator = np.sum(p_flat ** 2) + np.sum(t_flat ** 2)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def giou_loss_forward(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Generalized IoU (GIoU) Loss forward."""
    x1_i = np.maximum(boxes1[:, 0], boxes2[:, 0])
    y1_i = np.maximum(boxes1[:, 1], boxes2[:, 1])
    x2_i = np.minimum(boxes1[:, 2], boxes2[:, 2])
    y2_i = np.minimum(boxes1[:, 3], boxes2[:, 3])
    inter_area = np.maximum(0.0, x2_i - x1_i) * np.maximum(0.0, y2_i - y1_i)

    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area

    x1_c = np.minimum(boxes1[:, 0], boxes2[:, 0])
    y1_c = np.minimum(boxes1[:, 1], boxes2[:, 1])
    x2_c = np.maximum(boxes1[:, 2], boxes2[:, 2])
    y2_c = np.maximum(boxes1[:, 3], boxes2[:, 3])
    c_area = np.maximum(0.0, x2_c - x1_c) * np.maximum(0.0, y2_c - y1_c) + eps

    giou = iou - (c_area - union_area) / c_area
    return np.mean(1.0 - giou)


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/08_losses_metric_contrastive_and_structured_vision.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    weights = load_file(str(weights_path))

    test_in = weights["test_input"]
    weight = weights["weight"]
    bias = weights["bias"]
    test_out_gold = weights["test_output"]

    q_repr = weights["q_repr"]
    k_repr = weights["k_repr"]
    tau = float(weights["tau_val"][0])

    siglip_scale = float(weights["siglip_scale"][0])
    siglip_bias = float(weights["siglip_bias"][0])

    arc_emb = weights["arc_emb"]
    arc_w = weights["arc_weights"]
    arc_targets = weights["arc_targets"].astype(np.int64)
    arc_s = float(weights["arc_s"][0])
    arc_m = float(weights["arc_m"][0])

    dice_probs = weights["dice_probs"]
    dice_targets = weights["dice_targets"]

    boxes1 = weights["boxes1"]
    boxes2 = weights["boxes2"]

    # 1. Linear Anchor Parity
    y_anchor = linear_forward(test_in, weight, bias)
    diff_anchor = float(np.max(np.abs(y_anchor - test_out_gold)))
    assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

    # 2. InfoNCE Loss Parity
    loss_infonce = infonce_loss_forward(q_repr, k_repr, tau)
    diff_infonce = float(np.abs(loss_infonce - weights["loss_infonce"][0]))
    assert diff_infonce < 1e-5, f"InfoNCE mismatch: {diff_infonce:.2e}"

    # 3. SigLIP Loss Parity
    loss_siglip = siglip_loss_forward(q_repr, k_repr, siglip_scale, siglip_bias)
    diff_siglip = float(np.abs(loss_siglip - weights["loss_siglip"][0]))
    assert diff_siglip < 1e-5, f"SigLIP mismatch: {diff_siglip:.2e}"

    # 4. ArcFace Loss Parity
    loss_arcface = arcface_loss_forward(arc_emb, arc_w, arc_targets, arc_s, arc_m)
    diff_arcface = float(np.abs(loss_arcface - weights["loss_arcface"][0]))
    assert diff_arcface < 1e-5, f"ArcFace mismatch: {diff_arcface:.2e}"

    # 5. Soft-Dice Loss Parity
    loss_dice = soft_dice_loss_forward(dice_probs, dice_targets)
    diff_dice = float(np.abs(loss_dice - weights["loss_dice"][0]))
    assert diff_dice < 1e-5, f"Soft-Dice mismatch: {diff_dice:.2e}"

    # 6. GIoU Loss Parity
    loss_giou = giou_loss_forward(boxes1, boxes2)
    diff_giou = float(np.abs(loss_giou - weights["loss_giou"][0]))
    assert diff_giou < 1e-5, f"GIoU mismatch: {diff_giou:.2e}"

    print(
        f"All JAX / NumPy assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"InfoNCE: {diff_infonce:.2e}, SigLIP: {diff_siglip:.2e}, ArcFace: {diff_arcface:.2e}, "
        f"Dice: {diff_dice:.2e}, GIoU: {diff_giou:.2e})"
    )


if __name__ == "__main__":
    main()
