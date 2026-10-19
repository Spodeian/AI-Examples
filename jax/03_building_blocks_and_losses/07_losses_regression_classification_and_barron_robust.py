"""Functional JAX/NumPy implementation of Losses: Regression, Classification, and Barron Robust (C03 Topic 07).

Provides functional reference implementations for:
1. Barron's General Adaptive Robust Loss
2. Huber Loss
3. Focal Loss
4. PolyLoss
5. Asymmetric Loss (ASL)
6. Linear Anchor
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file


def barron_robust_loss_forward(
    r: np.ndarray,
    alpha: float = 1.0,
    c: float = 1.5,
) -> float:
    """Computes Barron's General Adaptive Robust Loss."""
    d_alpha = abs(alpha - 2.0)
    scaled_sq = (r / c) ** 2
    elem_loss = (d_alpha / alpha) * (np.power(scaled_sq / d_alpha + 1.0, alpha / 2.0) - 1.0)
    return float(np.mean(elem_loss))


def huber_loss_forward(
    r: np.ndarray,
    delta: float = 1.0,
) -> float:
    """Computes Huber Loss (Smooth L1 piecewise regression)."""
    abs_r = np.abs(r)
    elem_loss = np.where(abs_r <= delta, 0.5 * r ** 2, delta * (abs_r - 0.5 * delta))
    return float(np.mean(elem_loss))


def focal_loss_forward(
    logits: np.ndarray,
    targets: np.ndarray,
    gamma: float = 2.0,
) -> float:
    """Computes Multi-Class Focal Loss: -(1 - p_t)^gamma * log(p_t)."""
    targets = targets.astype(int)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True)) + max_logits
    log_probs = logits - lse
    probs = np.exp(log_probs)

    batch_idx = np.arange(logits.shape[0])
    target_log_probs = log_probs[batch_idx, targets]
    target_probs = probs[batch_idx, targets]
    elem_loss = -(1.0 - target_probs) ** gamma * target_log_probs
    return float(np.mean(elem_loss))


def polyloss_forward(
    logits: np.ndarray,
    targets: np.ndarray,
    epsilon_1: float = 1.0,
) -> float:
    """Computes Poly-1 Loss: -log(p_t) + epsilon_1 * (1 - p_t)."""
    targets = targets.astype(int)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True)) + max_logits
    log_probs = logits - lse
    probs = np.exp(log_probs)

    batch_idx = np.arange(logits.shape[0])
    target_log_probs = log_probs[batch_idx, targets]
    target_probs = probs[batch_idx, targets]
    elem_loss = -target_log_probs + epsilon_1 * (1.0 - target_probs)
    return float(np.mean(elem_loss))


def asymmetric_loss_forward(
    logits: np.ndarray,
    targets: np.ndarray,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip_margin: float = 0.05,
    eps: float = 1e-8,
) -> float:
    """Computes Asymmetric Loss (ASL) for multi-label classification."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs_m = np.maximum(probs - clip_margin, 0.0)

    pos_loss = -targets * ((1.0 - probs) ** gamma_pos) * np.log(np.maximum(probs, eps))
    neg_loss = -(1.0 - targets) * (probs_m ** gamma_neg) * np.log(np.maximum(1.0 - probs_m, eps))
    return float(np.mean(pos_loss + neg_loss))


def linear_forward(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Computes linear transformation: Y = X @ W + b."""
    return np.dot(x, w) + b


def load_checkpoint(checkpoint_path: str) -> Dict[str, np.ndarray]:
    return load_file(checkpoint_path)


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/07_losses_regression_classification_and_barron_robust.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    weights = load_checkpoint(str(weights_path))

    test_in = weights["test_input"]
    w = weights["weight"]
    b = weights["bias"]
    test_out = weights["test_output"]

    r_barron = weights["r_barron"]
    alpha_b = float(weights["alpha_barron"][0])
    c_b = float(weights["c_barron"][0])
    loss_b_gold = float(weights["loss_barron"][0])

    r_huber = weights["r_huber"]
    delta_h = float(weights["delta_huber"][0])
    loss_h_gold = float(weights["loss_huber"][0])

    logits_clf = weights["logits_clf"]
    targets_clf = weights["targets_clf"]
    gamma_f = float(weights["gamma_focal"][0])
    loss_f_gold = float(weights["loss_focal"][0])

    eps_p = float(weights["eps_poly"][0])
    loss_poly_gold = float(weights["loss_poly"][0])

    logits_asl = weights["logits_asl"]
    targets_asl = weights["targets_asl"]
    gamma_p = float(weights["gamma_pos"][0])
    gamma_n = float(weights["gamma_neg"][0])
    margin_asl = float(weights["margin_asl"][0])
    loss_asl_gold = float(weights["loss_asl"][0])

    # 1. Linear Anchor Parity
    y_anchor = linear_forward(test_in, w, b)
    diff_anchor = float(np.max(np.abs(y_anchor - test_out)))
    assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

    # 2. Barron Robust Loss Parity
    loss_b = barron_robust_loss_forward(r_barron, alpha_b, c_b)
    diff_b = abs(loss_b - loss_b_gold)
    assert diff_b < 1e-5, f"Barron mismatch: {diff_b:.2e}"

    # 3. Huber Loss Parity
    loss_h = huber_loss_forward(r_huber, delta_h)
    diff_h = abs(loss_h - loss_h_gold)
    assert diff_h < 1e-5, f"Huber mismatch: {diff_h:.2e}"

    # 4. Focal Loss Parity
    loss_f = focal_loss_forward(logits_clf, targets_clf, gamma_f)
    diff_f = abs(loss_f - loss_f_gold)
    assert diff_f < 1e-5, f"Focal mismatch: {diff_f:.2e}"

    # 5. PolyLoss Parity
    loss_poly = polyloss_forward(logits_clf, targets_clf, eps_p)
    diff_poly = abs(loss_poly - loss_poly_gold)
    assert diff_poly < 1e-5, f"PolyLoss mismatch: {diff_poly:.2e}"

    # 6. Asymmetric Loss Parity
    loss_asl = asymmetric_loss_forward(logits_asl, targets_asl, gamma_p, gamma_n, margin_asl)
    diff_asl = abs(loss_asl - loss_asl_gold)
    assert diff_asl < 1e-5, f"ASL mismatch: {diff_asl:.2e}"

    print(
        f"All JAX/NumPy assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"Barron: {diff_b:.2e}, Huber: {diff_h:.2e}, Focal: {diff_f:.2e}, "
        f"Poly: {diff_poly:.2e}, ASL: {diff_asl:.2e})"
    )


if __name__ == "__main__":
    main()
