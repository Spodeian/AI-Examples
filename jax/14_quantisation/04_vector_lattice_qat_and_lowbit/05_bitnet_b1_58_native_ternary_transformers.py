"""BitNet b1.58 Native Ternary Transformers — functional JAX implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers.
Run ``python jax/14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers.py``
to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def weight_quant_bitnet(w: jnp.ndarray, eps: float = 1e-5) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """BitNet b1.58 weight quantization: scales by mean(|W|) and clips to {-1, 0, +1}."""
    beta = jnp.mean(jnp.abs(w), keepdims=True)
    scale = jnp.clip(beta, a_min=eps)
    w_ternary = jnp.clip(jnp.round(w / scale), -1.0, 1.0)
    w_deq = w_ternary * scale
    return w_deq, w_ternary, beta


def activation_quant_bitnet(
    x: jnp.ndarray,
    q_b: float = 127.0,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """BitNet b1.58 per-token activation quantization to 8-bit integers."""
    gamma = jnp.max(jnp.abs(x), axis=1, keepdims=True)
    scale = jnp.clip(gamma / q_b, a_min=eps)
    x_int8 = jnp.clip(jnp.round(x / scale), -q_b, q_b)
    x_deq = x_int8 * scale
    return x_deq, x_int8, gamma


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Root Mean Square Layer Normalization (SubLN / RMSNorm)."""
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / jnp.sqrt(variance + eps)) * weight


def bitlinear_forward(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    rms_weight: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """BitLinear layer forward pass: SubLN -> INT8 act quant -> Ternary weight quant -> MatMul."""
    x_norm = rms_norm(x, rms_weight)
    x_q, _, _ = activation_quant_bitnet(x_norm, q_b=127.0)
    w_q, _, _ = weight_quant_bitnet(weight)
    y = x_q @ w_q
    if bias is not None:
        y = y + bias
    return y, x_q, w_q


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. RMSNorm
    x_norm = rms_norm(g["x_raw"], g["rms_weight"])
    err = float(jnp.max(jnp.abs(x_norm - g["x_norm"])))
    assert err < 1e-5, f"rms_norm error {err:.2e}"

    # 3. Activation quantization
    x_q, x_int8, gamma = activation_quant_bitnet(g["x_norm"], q_b=127.0)
    err = float(jnp.max(jnp.abs(x_q - g["x_q"])))
    assert err < 1e-5, f"x_q error {err:.2e}"
    err = float(jnp.max(jnp.abs(x_int8 - g["x_int8"])))
    assert err < 1e-5, f"x_int8 error {err:.2e}"
    err = float(jnp.max(jnp.abs(gamma - g["gamma"])))
    assert err < 1e-5, f"gamma error {err:.2e}"

    # 4. Weight quantization
    w_q, w_ternary, beta = weight_quant_bitnet(g["w_orig"])
    err = float(jnp.max(jnp.abs(w_q - g["w_q"])))
    assert err < 1e-5, f"w_q error {err:.2e}"
    err = float(jnp.max(jnp.abs(w_ternary - g["w_ternary"])))
    assert err < 1e-5, f"w_ternary error {err:.2e}"
    err = float(jnp.max(jnp.abs(beta - g["beta"])))
    assert err < 1e-5, f"beta error {err:.2e}"

    # 5. Pure integer accumulator
    int_acc = g["x_int8"] @ g["w_ternary"]
    err = float(jnp.max(jnp.abs(int_acc - g["integer_acc"])))
    assert err < 1e-5, f"integer_acc error {err:.2e}"

    # 6. BitLinear forward pass
    out_bitlinear, _, _ = bitlinear_forward(g["x_raw"], g["w_orig"], g["rms_weight"], g["b_orig"])
    err = float(jnp.max(jnp.abs(out_bitlinear - g["out_bitlinear"])))
    assert err < 1e-5, f"out_bitlinear error {err:.2e}"

    # 7. Scaled integer forward
    out_int_scaled = int_acc * (g["gamma"] * g["beta"] / 127.0) + g["b_orig"]
    err = float(jnp.max(jnp.abs(out_int_scaled - g["out_integer_scaled"])))
    assert err < 1e-5, f"out_integer_scaled error {err:.2e}"

    # 8. FP32 baseline
    out_fp32 = g["x_norm"] @ g["w_orig"] + g["b_orig"]
    err = float(jnp.max(jnp.abs(out_fp32 - g["out_fp32"])))
    assert err < 1e-5, f"out_fp32 error {err:.2e}"

    print("14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
