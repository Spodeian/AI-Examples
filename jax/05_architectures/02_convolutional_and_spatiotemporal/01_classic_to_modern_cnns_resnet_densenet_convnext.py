"""JAX / NumPy implementation for Category 05 Subfolder 02 Topic 01:
Classic to Modern CNNs: ResNet v1/v2, DenseNet, MobileNetV2, and ConvNeXt-V2 (GRN)
in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def gelu_np(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation function using math.erf."""
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def relu_np(x: np.ndarray) -> np.ndarray:
    """Standard ReLU activation function."""
    return np.maximum(x, 0.0)


def conv2d_np(x: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int = 0, groups: int = 1) -> np.ndarray:
    """Exact 2D Convolution in NumPy with grouped and padded support."""
    B, C_in, H, W = x.shape
    C_out, in_g, K_h, K_w = w.shape
    out_h = H + 2 * padding - K_h + 1
    out_w = W + 2 * padding - K_w + 1

    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    else:
        x_padded = x

    c_in_per_g = C_in // groups
    c_out_per_g = C_out // groups
    out = np.zeros((B, C_out, out_h, out_w), dtype=np.float32)

    for g in range(groups):
        x_g = x_padded[:, g * c_in_per_g:(g + 1) * c_in_per_g, :, :]
        for o in range(c_out_per_g):
            c_out_idx = g * c_out_per_g + o
            w_kernel = w[c_out_idx]  # (c_in_per_g, K_h, K_w)
            for i in range(out_h):
                for j in range(out_w):
                    patch = x_g[:, :, i:i + K_h, j:j + K_w]  # (B, c_in_per_g, K_h, K_w)
                    out[:, c_out_idx, i, j] = np.sum(patch * w_kernel, axis=(1, 2, 3)) + b[c_out_idx]
    return out


def global_response_norm_np(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Global Response Normalization (GRN) for ConvNeXt-V2."""
    gx = np.sqrt(np.sum(np.square(x), axis=(2, 3), keepdims=True) + eps)
    nx = gx / (np.mean(gx, axis=1, keepdims=True) + eps)
    return gamma[None, :, None, None] * (x * nx) + beta[None, :, None, None] + x


def convnext_v2_forward_np(
    x: np.ndarray,
    dw_w: np.ndarray,
    dw_b: np.ndarray,
    pw1_w: np.ndarray,
    pw1_b: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    pw2_w: np.ndarray,
    pw2_b: np.ndarray,
) -> np.ndarray:
    """ConvNeXt-V2 Block forward pass."""
    in_channels = x.shape[1]
    h_dw = conv2d_np(x, dw_w, dw_b, padding=3, groups=in_channels)
    h_pw1 = conv2d_np(h_dw, pw1_w, pw1_b, padding=0, groups=1)
    h_act = gelu_np(h_pw1)
    h_grn = global_response_norm_np(h_act, gamma, beta)
    h_pw2 = conv2d_np(h_grn, pw2_w, pw2_b, padding=0, groups=1)
    return x + h_pw2


def preact_resnet_forward_np(
    x: np.ndarray,
    res_w1: np.ndarray,
    res_b1: np.ndarray,
    res_w2: np.ndarray,
    res_b2: np.ndarray,
) -> np.ndarray:
    """Pre-Activation ResNet Block forward pass."""
    r1 = relu_np(x)
    r2 = conv2d_np(r1, res_w1, res_b1, padding=1)
    r3 = relu_np(r2)
    r4 = conv2d_np(r3, res_w2, res_b2, padding=1)
    return x + r4


def linear_anchor(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Standard linear anchor for test_cross_track_parity.py."""
    return np.matmul(x, weight) + bias


def verify_against_golden() -> None:
    """Verifies JAX / NumPy implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = float(np.max(np.abs(test_out - weights["test_output"])))
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]

    # 2. ConvNeXt-V2 Block verification
    out_convnext = convnext_v2_forward_np(
        x,
        weights["dw_w"], weights["dw_b"],
        weights["pw1_w"], weights["pw1_b"],
        weights["grn_gamma"], weights["grn_beta"],
        weights["pw2_w"], weights["pw2_b"],
    )
    err_convnext = float(np.max(np.abs(out_convnext - weights["convnext_output"])))
    assert err_convnext < 1e-5, f"ConvNeXt-V2 parity error {err_convnext:.2e} exceeds 1e-5"

    # 3. Pre-Activation ResNet verification
    out_resnet = preact_resnet_forward_np(
        x,
        weights["res_w1"], weights["res_b1"],
        weights["res_w2"], weights["res_b2"],
    )
    err_resnet = float(np.max(np.abs(out_resnet - weights["resnet_output"])))
    assert err_resnet < 1e-5, f"ResNet parity error {err_resnet:.2e} exceeds 1e-5"

    print("JAX / NumPy ConvNeXt & ResNet parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
