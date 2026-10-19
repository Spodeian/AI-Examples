"""Functional JAX/NumPy implementation of Regularization and Training Stabilizers (C03 Topic 06).

Provides functional reference implementations for:
1. Inverted Dropout with deterministic mask
2. DropPath (Stochastic Depth)
3. Logit Soft-Capping
4. Label Smoothing Cross-Entropy
5. Router Z-Loss
6. Linear Anchor
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file


def dropout_forward(
    x: np.ndarray,
    mask: np.ndarray,
    p: float = 0.5,
) -> np.ndarray:
    """Computes Inverted Dropout: (x * mask) / (1 - p)."""
    return (x * mask) / (1.0 - p)


def droppath_forward(
    x_identity: np.ndarray,
    x_branch: np.ndarray,
    mask: np.ndarray,
    drop_prob: float = 0.25,
) -> np.ndarray:
    """Computes DropPath / Stochastic Depth: identity + (branch * mask) / (1 - p)."""
    scale = 1.0 / (1.0 - drop_prob)
    return x_identity + (x_branch * mask) * scale


def softcapping_forward(
    x: np.ndarray,
    cap: float = 30.0,
) -> np.ndarray:
    """Computes Logit Soft-Capping: cap * tanh(x / cap)."""
    return cap * np.tanh(x / cap)


def label_smoothing_loss_forward(
    logits: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Computes Label Smoothing Cross-Entropy Loss."""
    num_classes = logits.shape[-1]
    targets = targets.astype(int)

    # Numerically stable log-softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True)) + max_logits
    log_probs = logits - lse

    # Cross entropy with target class
    batch_idx = np.arange(logits.shape[0])
    target_log_probs = log_probs[batch_idx, targets]
    nll = -target_log_probs

    # Uniform component
    uniform_loss = -np.sum(log_probs, axis=-1) / num_classes

    # Smoothed loss
    loss = (1.0 - alpha) * nll + alpha * uniform_loss
    return float(np.mean(loss))


def router_z_loss_forward(
    router_logits: np.ndarray,
    z_coef: float = 1e-4,
) -> float:
    """Computes Router Z-Loss: (z_coef / B) * sum( (log sum exp(z))^2 )."""
    max_logits = np.max(router_logits, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(router_logits - max_logits), axis=-1)) + max_logits.squeeze(-1)
    return float(z_coef * np.mean(lse ** 2))


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
    weights_path = Path("weights/03_building_blocks_and_losses/06_regularization_and_training_stabilizers.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    weights = load_checkpoint(str(weights_path))

    test_in = weights["test_input"]
    w = weights["weight"]
    b = weights["bias"]
    test_out = weights["test_output"]

    x_drop = weights["x_drop"]
    drop_mask = weights["drop_mask"]
    drop_p = float(weights["drop_rate"][0])
    y_drop_gold = weights["y_drop"]

    x_id = weights["x_identity"]
    x_br = weights["x_branch"]
    dp_mask = weights["droppath_mask"]
    dp_rate = float(weights["droppath_rate"][0])
    y_droppath_gold = weights["y_droppath"]

    x_log = weights["x_logits"]
    softcap = float(weights["softcap_val"][0])
    y_softcap_gold = weights["y_softcap"]

    logits_ls = weights["logits_ls"]
    targets_ls = weights["targets_ls"]
    alpha_ls = float(weights["smooth_alpha"][0])
    loss_ls_gold = float(weights["loss_ls"][0])

    router_logits = weights["router_logits"]
    z_coef = float(weights["z_coef"][0])
    loss_z_gold = float(weights["loss_z"][0])

    # 1. Linear Anchor Parity
    y_anchor = linear_forward(test_in, w, b)
    diff_anchor = float(np.max(np.abs(y_anchor - test_out)))
    assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

    # 2. Inverted Dropout Parity
    y_dp = dropout_forward(x_drop, drop_mask, drop_p)
    diff_dp = float(np.max(np.abs(y_dp - y_drop_gold)))
    assert diff_dp < 1e-5, f"Dropout mismatch: {diff_dp:.2e}"

    # 3. DropPath Parity
    y_dpath = droppath_forward(x_id, x_br, dp_mask, dp_rate)
    diff_dpath = float(np.max(np.abs(y_dpath - y_droppath_gold)))
    assert diff_dpath < 1e-5, f"DropPath mismatch: {diff_dpath:.2e}"

    # 4. Logit Soft-Capping Parity
    y_sc = softcapping_forward(x_log, softcap)
    diff_sc = float(np.max(np.abs(y_sc - y_softcap_gold)))
    assert diff_sc < 1e-5, f"Soft-Capping mismatch: {diff_sc:.2e}"

    # 5. Label Smoothing Loss Parity
    loss_ls = label_smoothing_loss_forward(logits_ls, targets_ls, alpha_ls)
    diff_ls = abs(loss_ls - loss_ls_gold)
    assert diff_ls < 1e-5, f"Label Smoothing mismatch: {diff_ls:.2e}"

    # 6. Router Z-Loss Parity
    loss_z = router_z_loss_forward(router_logits, z_coef)
    diff_z = abs(loss_z - loss_z_gold)
    assert diff_z < 1e-5, f"Router Z-Loss mismatch: {diff_z:.2e}"

    print(
        f"All JAX/NumPy assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"Dropout: {diff_dp:.2e}, DropPath: {diff_dpath:.2e}, SoftCap: {diff_sc:.2e}, "
        f"LabelSmooth: {diff_ls:.2e}, RouterZ: {diff_z:.2e})"
    )


if __name__ == "__main__":
    main()
