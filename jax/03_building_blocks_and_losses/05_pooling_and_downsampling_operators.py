"""Functional JAX/NumPy implementation of Pooling and Downsampling Operators (C03 Topic 05).

Provides functional reference implementations for:
1. Generalized Mean Pooling (GeM)
2. Anti-Aliased Downsampling (BlurPool)
3. PixelUnshuffle
4. Pooling by Multihead Attention (PMA)
5. Linear Anchor
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file


def gem_pool_forward(
    x: np.ndarray,
    p: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Computes Generalized Mean Pooling over spatial dimensions (-2, -1)."""
    x_clamped = np.clip(x, a_min=eps, a_max=None)
    x_pow = np.power(x_clamped, p)
    x_mean = np.mean(x_pow, axis=(-2, -1))
    return np.power(x_mean, 1.0 / p)


def blur_pool_forward(
    x: np.ndarray,
    kernel: np.ndarray,
    stride: int = 2,
) -> np.ndarray:
    """Computes 2D Anti-Aliased Downsampling (BlurPool) with reflect padding."""
    # Reflect pad spatial dimensions
    x_padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
    batch_size, channels, height, width = x_padded.shape
    kernel_h, kernel_w = kernel.shape[2], kernel.shape[3]
    out_h = (height - kernel_h) // stride + 1
    out_w = (width - kernel_w) // stride + 1

    out = np.zeros((batch_size, channels, out_h, out_w), dtype=np.float32)
    for b in range(batch_size):
        for c in range(channels):
            k = kernel[c, 0]
            for oh in range(out_h):
                h_start = oh * stride
                for ow in range(out_w):
                    w_start = ow * stride
                    window = x_padded[b, c, h_start : h_start + kernel_h, w_start : w_start + kernel_w]
                    out[b, c, oh, ow] = np.sum(window * k)
    return out


def pixel_unshuffle_forward(
    x: np.ndarray,
    downscale_factor: int = 2,
) -> np.ndarray:
    """Computes Lossless Spatial-to-Channel Downsampling (PixelUnshuffle)."""
    batch_size, channels, height, width = x.shape
    r = downscale_factor
    out_h = height // r
    out_w = width // r
    # Reshape: [B, C, H//r, r, W//r, r] -> Permute: [B, C, r, r, H//r, W//r] -> Reshape: [B, C*r*r, H//r, W//r]
    x_reshaped = x.reshape(batch_size, channels, out_h, r, out_w, r)
    x_permuted = x_reshaped.transpose(0, 1, 3, 5, 2, 4)
    return x_permuted.reshape(batch_size, channels * r * r, out_h, out_w)


def pma_forward(
    x: np.ndarray,
    s_seed: np.ndarray,
) -> np.ndarray:
    """Computes Pooling by Multihead Attention (PMA)."""
    batch_size, seq_len, dim = x.shape
    q = np.broadcast_to(s_seed, (batch_size, s_seed.shape[1], dim)) # [B, 1, D]
    # Dot-product attention scores
    scores = np.matmul(q, x.transpose(0, 2, 1)) / math.sqrt(float(dim)) # [B, 1, L]
    # Numerically stable softmax along last axis
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    out = np.matmul(weights, x) # [B, 1, D]
    return np.squeeze(out, axis=1) # [B, D]


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
    weights_path = Path("weights/03_building_blocks_and_losses/05_pooling_and_downsampling_operators.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    weights = load_checkpoint(str(weights_path))

    test_in = weights["test_input"]
    w = weights["weight"]
    b = weights["bias"]
    test_out = weights["test_output"]

    x_gem = weights["x_gem"]
    p_gem = float(weights["p_gem"][0])
    y_gem_gold = weights["y_gem"]

    x_blur = weights["x_blur"]
    blur_kernel = weights["blur_kernel"]
    y_blur_gold = weights["y_blur"]

    x_pixel = weights["x_pixel"]
    y_pixel_gold = weights["y_pixel"]

    x_pma = weights["x_pma"]
    s_seed = weights["s_seed"]
    y_pma_gold = weights["y_pma"]

    # 1. Linear Anchor Parity
    y_anchor = linear_forward(test_in, w, b)
    diff_anchor = float(np.max(np.abs(y_anchor - test_out)))
    assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

    # 2. GeM Pooling Parity
    y_gem = gem_pool_forward(x_gem, p=p_gem)
    diff_gem = float(np.max(np.abs(y_gem - y_gem_gold)))
    assert diff_gem < 1e-5, f"GeM mismatch: {diff_gem:.2e}"

    # 3. BlurPool Parity
    y_blur = blur_pool_forward(x_blur, blur_kernel, stride=2)
    diff_blur = float(np.max(np.abs(y_blur - y_blur_gold)))
    assert diff_blur < 1e-5, f"BlurPool mismatch: {diff_blur:.2e}"

    # 4. PixelUnshuffle Parity
    y_pixel = pixel_unshuffle_forward(x_pixel, downscale_factor=2)
    diff_pixel = float(np.max(np.abs(y_pixel - y_pixel_gold)))
    assert diff_pixel < 1e-5, f"PixelUnshuffle mismatch: {diff_pixel:.2e}"

    # 5. PMA Attention Pooling Parity
    y_pma = pma_forward(x_pma, s_seed)
    diff_pma = float(np.max(np.abs(y_pma - y_pma_gold)))
    assert diff_pma < 1e-5, f"PMA mismatch: {diff_pma:.2e}"

    print(
        f"All JAX/NumPy assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"GeM: {diff_gem:.2e}, Blur: {diff_blur:.2e}, Pixel: {diff_pixel:.2e}, PMA: {diff_pma:.2e})"
    )


if __name__ == "__main__":
    main()
