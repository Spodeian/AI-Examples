"""JAX / NumPy implementation for Category 05 Topic 06:
Lossless Deliberation: Reversible RevNet, Hyper-Connections, and Recurrent Looped Depth
in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def gelu_np(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation function using math.erf."""
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def layer_norm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization across the trailing dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def eval_mlp_np(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Evaluates 2-layer MLP: GELU(x @ w1 + b1) @ w2 + b2."""
    h = gelu_np(np.matmul(x, w1) + b1)
    return np.matmul(h, w2) + b2


def revnet_forward_np(
    x: np.ndarray,
    f_w1: np.ndarray,
    f_b1: np.ndarray,
    f_w2: np.ndarray,
    f_b2: np.ndarray,
    g_w1: np.ndarray,
    g_b1: np.ndarray,
    g_w2: np.ndarray,
    g_b2: np.ndarray,
) -> np.ndarray:
    """Reversible RevNet forward pass."""
    d_sub = x.shape[-1] // 2
    x1 = x[..., :d_sub]
    x2 = x[..., d_sub:]

    f_x2 = eval_mlp_np(x2, f_w1, f_b1, f_w2, f_b2)
    y1 = x1 + f_x2

    g_y1 = eval_mlp_np(y1, g_w1, g_b1, g_w2, g_b2)
    y2 = x2 + g_y1

    return np.concatenate([y1, y2], axis=-1)


def revnet_inverse_np(
    y: np.ndarray,
    f_w1: np.ndarray,
    f_b1: np.ndarray,
    f_w2: np.ndarray,
    f_b2: np.ndarray,
    g_w1: np.ndarray,
    g_b1: np.ndarray,
    g_w2: np.ndarray,
    g_b2: np.ndarray,
) -> np.ndarray:
    """Reversible RevNet exact algebraic inverse reconstruction."""
    d_sub = y.shape[-1] // 2
    y1 = y[..., :d_sub]
    y2 = y[..., d_sub:]

    g_y1 = eval_mlp_np(y1, g_w1, g_b1, g_w2, g_b2)
    x2 = y2 - g_y1

    f_x2 = eval_mlp_np(x2, f_w1, f_b1, f_w2, f_b2)
    x1 = y1 - f_x2

    return np.concatenate([x1, x2], axis=-1)


def hyper_connections_forward_np(
    h1: np.ndarray,
    h2: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    hc_w1: np.ndarray,
    hc_b1: np.ndarray,
    hc_w2: np.ndarray,
    hc_b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-lane Hyper-Connections forward pass."""
    u = alpha[0] * h1 + alpha[1] * h2
    v = eval_mlp_np(u, hc_w1, hc_b1, hc_w2, hc_b2)

    h1_next = h1 + beta[0] * v
    h2_next = h2 + beta[1] * v
    combined = 0.5 * (h1_next + h2_next)

    return h1_next, h2_next, combined


def looped_depth_forward_np(
    x: np.ndarray,
    loop_w1: np.ndarray,
    loop_b1: np.ndarray,
    loop_w2: np.ndarray,
    loop_b2: np.ndarray,
    steps: int = 4,
    gamma: float = 0.3,
) -> np.ndarray:
    """Looped Recurrent Depth forward pass with persistent input anchoring."""
    z = np.copy(x)
    for _ in range(steps):
        z_in = z + gamma * x
        delta = eval_mlp_np(z_in, loop_w1, loop_b1, loop_w2, loop_b2)
        z = layer_norm_np(z + delta)
    return z


def linear_anchor(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Standard linear anchor for test_cross_track_parity.py."""
    return np.matmul(x, weight) + bias


def verify_against_golden() -> None:
    """Verifies JAX / NumPy implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = float(np.max(np.abs(test_out - weights["test_output"])))
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]

    # 2. RevNet forward and inverse verification
    out_rev = revnet_forward_np(
        x,
        weights["f_w1"], weights["f_b1"], weights["f_w2"], weights["f_b2"],
        weights["g_w1"], weights["g_b1"], weights["g_w2"], weights["g_b2"],
    )
    err_rev = float(np.max(np.abs(out_rev - weights["revnet_output"])))
    assert err_rev < 1e-6, f"RevNet parity error {err_rev:.2e} exceeds 1e-6"

    x_rec = revnet_inverse_np(
        out_rev,
        weights["f_w1"], weights["f_b1"], weights["f_w2"], weights["f_b2"],
        weights["g_w1"], weights["g_b1"], weights["g_w2"], weights["g_b2"],
    )
    err_rec = float(np.max(np.abs(x_rec - x)))
    assert err_rec < 1e-6, f"RevNet reconstruction error {err_rec:.2e} exceeds 1e-6"

    # 3. Hyper-Connections verification
    h1 = np.copy(x)
    h2 = x * 0.5
    _, _, out_hc = hyper_connections_forward_np(
        h1, h2,
        weights["alpha"], weights["beta"],
        weights["hc_w1"], weights["hc_b1"], weights["hc_w2"], weights["hc_b2"],
    )
    err_hc = float(np.max(np.abs(out_hc - weights["hc_output"])))
    assert err_hc < 1e-6, f"Hyper-Connections parity error {err_hc:.2e} exceeds 1e-6"

    # 4. Looped Depth verification
    out_loop = looped_depth_forward_np(
        x,
        weights["loop_w1"], weights["loop_b1"], weights["loop_w2"], weights["loop_b2"],
        steps=4, gamma=float(weights["gamma"]),
    )
    err_loop = float(np.max(np.abs(out_loop - weights["looped_output"])))
    assert err_loop < 1e-6, f"Looped Depth parity error {err_loop:.2e} exceeds 1e-6"

    print("JAX / NumPy Lossless Deliberation parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
