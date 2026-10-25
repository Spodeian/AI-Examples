"""Smoothquant and Outlier Migration W8a8 — functional JAX implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8. Run ``python jax/14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def compute_smoothquant_scale(
    x_calib: jnp.ndarray,
    weight: jnp.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Computes migration scale s_j = (max |X_j|)^alpha / (max |W_j|)^(1 - alpha)."""
    act_max = jnp.max(jnp.abs(x_calib), axis=0)
    weight_max = jnp.max(jnp.abs(weight), axis=1)
    act_max = jnp.clip(act_max, a_min=eps)
    weight_max = jnp.clip(weight_max, a_min=eps)

    s = (act_max ** alpha) / (weight_max ** (1.0 - alpha))
    s = jnp.clip(s, a_min=eps)
    return s


def quantize_per_token_int8(
    x: jnp.ndarray,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-token symmetric INT8 activation quantization."""
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / qmax
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    q = jnp.clip(jnp.round(x / scale), qmin, qmax)
    return q, scale


def quantize_per_channel_int8(
    w: jnp.ndarray,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-column symmetric INT8 weight quantization."""
    scale = jnp.max(jnp.abs(w), axis=0, keepdims=True) / qmax
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    q = jnp.clip(jnp.round(w / scale), qmin, qmax)
    return q, scale


def w8a8_gemm_unfused(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Standard W8A8 GEMM without smoothing."""
    q_x, s_x = quantize_per_token_int8(x)
    q_w, s_w = quantize_per_channel_int8(weight)
    deq_x = q_x * s_x
    deq_w = q_w * s_w
    out = deq_x @ deq_w
    if bias is not None:
        out = out + bias
    return out


def smoothquant_w8a8_gemm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """SmoothQuant W8A8 GEMM: scales activations and weights, then performs INT8 GEMM."""
    x_smooth = x / jnp.expand_dims(scale, 0)
    w_smooth = weight * jnp.expand_dims(scale, 1)

    q_x, s_x = quantize_per_token_int8(x_smooth)
    q_w, s_w = quantize_per_channel_int8(w_smooth)

    deq_x = q_x * s_x
    deq_w = q_w * s_w
    out = deq_x @ deq_w
    if bias is not None:
        out = out + bias
    return out, x_smooth, w_smooth


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Smooth scale
    scale = compute_smoothquant_scale(g["x_calib"], g["w_orig"], alpha=0.5)
    assert float(jnp.max(jnp.abs(scale - g["scale"]))) < 1e-5

    # 3. Unsmoothed W8A8 output
    out_unsmoothed = w8a8_gemm_unfused(g["x_calib"][:8], g["w_orig"], g["b_orig"])
    assert float(jnp.max(jnp.abs(out_unsmoothed - g["out_unsmoothed"]))) < 1e-5

    # 4. Smoothed activations and weights
    out_smooth, x_smooth, w_smooth = smoothquant_w8a8_gemm(
        g["x_calib"][:8], g["w_orig"], g["scale"], g["b_orig"]
    )
    assert float(jnp.max(jnp.abs(x_smooth - g["x_smooth"]))) < 1e-5
    assert float(jnp.max(jnp.abs(w_smooth - g["w_smooth"]))) < 1e-5
    assert float(jnp.max(jnp.abs(out_smooth - g["out_smooth"]))) < 1e-5

    print("14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
