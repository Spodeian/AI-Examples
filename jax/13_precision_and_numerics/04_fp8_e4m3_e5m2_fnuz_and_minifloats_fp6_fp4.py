"""FP8 E4M3 E5M2 FNUZ and Minifloats FP6 FP4 — functional JAX implementation.

Topic: 13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4. Run ``python jax/13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def quantize_minifloat(
    x: jnp.ndarray,
    e_bits: int,
    m_bits: int,
    bias: int,
    max_val: float,
) -> jnp.ndarray:
    """Pure JAX minifloat quantizer with subnormal support."""
    x_abs = jnp.abs(x)
    is_zero = (x_abs == 0.0)
    x_safe = jnp.where(is_zero, jnp.ones_like(x_abs), x_abs)
    
    e = jnp.floor(jnp.log2(x_safe))
    e_min_normal = 1.0 - float(bias)
    
    step = jnp.where(
        e < e_min_normal,
        2.0 ** (e_min_normal - float(m_bits)),
        2.0 ** (e - float(m_bits)),
    )
    
    q = jnp.round(x_safe / step) * step
    q = jnp.clip(q, 0.0, float(max_val))
    return jnp.where(is_zero, jnp.zeros_like(x), jnp.sign(x) * q)


def fp8_e4m3fn_quantize(x: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
    scaled = x * (1.0 / float(scale))
    q = quantize_minifloat(scaled, e_bits=4, m_bits=3, bias=7, max_val=448.0)
    return q * float(scale)


def fp8_e5m2_quantize(x: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
    scaled = x * (1.0 / float(scale))
    q = quantize_minifloat(scaled, e_bits=5, m_bits=2, bias=15, max_val=57344.0)
    return q * float(scale)


def fp4_e2m1_quantize(x: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
    scaled = x * (1.0 / float(scale))
    q = quantize_minifloat(scaled, e_bits=2, m_bits=1, bias=1, max_val=6.0)
    return q * float(scale)


def delayed_scaled_fp8_gemm(
    a: jnp.ndarray,
    b: jnp.ndarray,
    a_scale: float,
    b_scale: float,
) -> tuple[jnp.ndarray, float, float]:
    a_q = fp8_e4m3fn_quantize(a, scale=a_scale)
    b_q = fp8_e4m3fn_quantize(b, scale=b_scale)
    out = a_q @ b_q
    
    a_max = float(jnp.max(jnp.abs(a)))
    b_max = float(jnp.max(jnp.abs(b)))
    next_a_scale = max(1e-6, a_max / 448.0)
    next_b_scale = max(1e-6, b_max / 448.0)
    return out, next_a_scale, next_b_scale


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. FP8 E4M3FN
    e4m3 = fp8_e4m3fn_quantize(g["fp8_input"], scale=0.5)
    err_e4m3 = float(jnp.max(jnp.abs(e4m3 - g["e4m3_ref"])))
    assert err_e4m3 < 1e-5, f"e4m3 error {err_e4m3:.2e}"
    
    # 3. FP8 E5M2
    e5m2 = fp8_e5m2_quantize(g["fp8_input"], scale=0.5)
    err_e5m2 = float(jnp.max(jnp.abs(e5m2 - g["e5m2_ref"])))
    assert err_e5m2 < 1e-5, f"e5m2 error {err_e5m2:.2e}"
    
    # 4. FP4 E2M1
    fp4 = fp4_e2m1_quantize(g["fp4_input"], scale=0.25)
    err_fp4 = float(jnp.max(jnp.abs(fp4 - g["fp4_ref"])))
    assert err_fp4 < 1e-5, f"fp4 error {err_fp4:.2e}"
    
    # 5. Delayed Scaled FP8 GEMM
    out, next_a, next_b = delayed_scaled_fp8_gemm(g["gemm_a"], g["gemm_b"], a_scale=0.01, b_scale=0.01)
    err_out = float(jnp.max(jnp.abs(out - g["gemm_out_ref"])))
    assert err_out < 1e-5, f"gemm out error {err_out:.2e}"
    assert abs(next_a - float(g["next_scale_a"])) < 1e-5, f"next_a error {abs(next_a - float(g['next_scale_a']))}"
    assert abs(next_b - float(g["next_scale_b"])) < 1e-5, f"next_b error {abs(next_b - float(g['next_scale_b']))}"
    
    print("13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
