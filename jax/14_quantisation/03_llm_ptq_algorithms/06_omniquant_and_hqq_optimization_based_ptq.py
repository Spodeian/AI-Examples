"""Omniquant and HQQ Optimization Based PTQ — functional JAX implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq. Run ``python jax/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def lwc_quantize(
    w: jnp.ndarray,
    h_l: jnp.ndarray,
    h_u: jnp.ndarray,
    qmin: float = 0.0,
    qmax: float = 15.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Learnable Weight Clipping (LWC) asymmetric quantization."""
    w_min = jnp.min(w, axis=0, keepdims=True)
    w_max = jnp.max(w, axis=0, keepdims=True)

    lower = w_min * jax.nn.sigmoid(h_l)
    upper = w_max * jax.nn.sigmoid(h_u)

    scale = (upper - lower) / qmax
    scale = jnp.clip(scale, a_min=1e-5)

    q = jnp.clip(jnp.round((w - lower) / scale), qmin, qmax)
    w_deq = q * scale + lower
    return w_deq, scale, lower


def omniquant_forward(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    gamma: jnp.ndarray,
    h_l: jnp.ndarray,
    h_u: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """OmniQuant forward pass combining Learnable Equivalent Transformation (LET) & LWC."""
    s = jnp.exp(gamma)
    x_scaled = x / jnp.expand_dims(s, 0)
    w_scaled = weight * jnp.expand_dims(s, 1)

    w_deq, _, _ = lwc_quantize(w_scaled, h_l, h_u)
    out = x_scaled @ w_deq
    if bias is not None:
        out = out + bias
    return out, w_deq


def hqq_proximal_quantize(
    w: jnp.ndarray,
    nbits: int = 4,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Half-Quadratic Quantization (HQQ) closed-form optimal scale and zero-point derivation."""
    qmax = float(2 ** nbits - 1)
    w_min = jnp.min(w, axis=0, keepdims=True)
    w_max = jnp.max(w, axis=0, keepdims=True)

    scale = (w_max - w_min) / qmax
    scale = jnp.clip(scale, a_min=1e-5)
    zero = -w_min / scale

    q = jnp.clip(jnp.round(w / scale + zero), 0.0, qmax)
    w_deq = (q - zero) * scale
    return w_deq, scale, zero


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. HQQ quantization parity
    w_hqq, hqq_scale, hqq_zero = hqq_proximal_quantize(g["w_orig"], nbits=4)
    assert float(jnp.max(jnp.abs(w_hqq - g["w_hqq"]))) < 1e-5
    assert float(jnp.max(jnp.abs(hqq_scale - g["hqq_scale"]))) < 1e-5
    assert float(jnp.max(jnp.abs(hqq_zero - g["hqq_zero"]))) < 1e-5

    out_hqq = g["x_calib"] @ w_hqq + g["b_orig"]
    assert float(jnp.max(jnp.abs(out_hqq - g["out_hqq"]))) < 1e-5

    # 3. OmniQuant forward with optimized parameters
    out_omni, _ = omniquant_forward(
        g["x_calib"], g["w_orig"], g["gamma_opt"], g["h_l_opt"], g["h_u_opt"], g["b_orig"]
    )
    assert float(jnp.max(jnp.abs(out_omni - g["out_omniquant"]))) < 1e-5

    print("14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
