"""INT UINT Fixed Point Q Format Dyadic and RNS Arithmetic — functional JAX implementation.

Topic: 14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic. Run ``python jax/14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def affine_quantize_dequantize(
    x: jnp.ndarray,
    num_bits: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray, float, float]:
    qmax = float((1 << num_bits) - 1)
    xmin = float(jnp.min(x))
    xmax = float(jnp.max(x))
    
    scale = max(1e-8, (xmax - xmin) / qmax)
    zero_point = float(jnp.clip(jnp.round(-xmin / scale), 0.0, qmax))
    
    q = jnp.clip(jnp.round(x * (1.0 / scale)) + zero_point, 0.0, qmax)
    x_hat = (q - zero_point) * scale
    return x_hat, q, scale, zero_point


def symmetric_quantize_dequantize(
    x: jnp.ndarray,
    num_bits: int = 8,
    full_range: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    qmax = float(1 << (num_bits - 1)) - 1.0
    qmin = -float(1 << (num_bits - 1)) if full_range else -qmax
    
    amax = float(jnp.max(jnp.abs(x)))
    scale = max(1e-8, amax / qmax)
    
    q = jnp.clip(jnp.round(x * (1.0 / scale)), qmin, qmax)
    x_hat = q * scale
    return x_hat, q, scale


def fixed_point_qmn(
    x: jnp.ndarray,
    m: int = 4,
    n: int = 4,
) -> jnp.ndarray:
    scale = float(1 << n)
    qmax = float(1 << (m + n - 1)) - 1.0
    qmin = -float(1 << (m + n - 1))
    
    q = jnp.clip(jnp.round(x * scale), qmin, qmax)
    return q * (1.0 / scale)


def dyadic_rescale_tensor(
    acc_int32: jnp.ndarray,
    m0: int,
    shift: int,
) -> jnp.ndarray:
    acc_64 = acc_int32.astype(jnp.int64)
    scaled = (acc_64 * int(m0) + (1 << 30)) >> 31
    final_shift = shift - 31
    if final_shift > 0:
        res = (scaled + (1 << (final_shift - 1))) >> final_shift
    else:
        res = scaled << (-final_shift)
    return res.astype(jnp.float32)


def rns_multiply_and_crt(
    a: jnp.ndarray,
    b: jnp.ndarray,
    moduli: tuple[int, ...] = (7, 11, 13),
) -> jnp.ndarray:
    m_total = 1
    for m in moduli:
        m_total *= m
        
    a_int = a.astype(jnp.int64) % m_total
    b_int = b.astype(jnp.int64) % m_total
    
    m_primes = [m_total // m for m in moduli]
    inverses = [pow(m_primes[i], -1, moduli[i]) for i in range(len(moduli))]
    
    c_components = []
    for m in moduli:
        c_i = ((a_int % m) * (b_int % m)) % m
        c_components.append(c_i)
        
    reconstructed = jnp.zeros_like(a_int)
    for i in range(len(moduli)):
        weight = m_primes[i] * inverses[i]
        reconstructed = reconstructed + c_components[i] * weight
        
    return (reconstructed % m_total).astype(jnp.float32)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. Affine INT8
    aff_hat, aff_q, aff_s, aff_z = affine_quantize_dequantize(g["raw_relu"], num_bits=8)
    assert float(jnp.max(jnp.abs(aff_hat - g["affine_hat"]))) < 1e-5
    assert float(jnp.max(jnp.abs(aff_q - g["affine_q"]))) < 1e-5
    assert abs(aff_s - float(g["affine_s"])) < 1e-5
    assert abs(aff_z - float(g["affine_z"])) < 1e-5
    
    # 3. Symmetric INT8
    sym_restr_hat, sym_restr_q, sym_restr_s = symmetric_quantize_dequantize(g["raw_sym"], num_bits=8, full_range=False)
    assert float(jnp.max(jnp.abs(sym_restr_hat - g["sym_hat_restr"]))) < 1e-5
    assert float(jnp.max(jnp.abs(sym_restr_q - g["sym_q_restr"]))) < 1e-5
    assert abs(sym_restr_s - float(g["sym_s_restr"])) < 1e-5
    
    sym_full_hat, sym_full_q, sym_full_s = symmetric_quantize_dequantize(g["raw_sym"], num_bits=8, full_range=True)
    assert float(jnp.max(jnp.abs(sym_full_hat - g["sym_hat_full"]))) < 1e-5
    assert float(jnp.max(jnp.abs(sym_full_q - g["sym_q_full"]))) < 1e-5
    assert abs(sym_full_s - float(g["sym_s_full"])) < 1e-5
    
    # 4. Fixed-Point Q4.4
    qmn = fixed_point_qmn(g["raw_fp"], m=4, n=4)
    assert float(jnp.max(jnp.abs(qmn - g["qmn_hat"]))) < 1e-5
    
    # 5. Dyadic Rescaling
    dyadic_res = dyadic_rescale_tensor(g["acc_int"].astype(jnp.int32), int(g["m0"]), int(g["shift"]))
    assert float(jnp.max(jnp.abs(dyadic_res - g["dyadic_out"]))) < 1e-5
    
    # 6. RNS Multiplication
    rns_res = rns_multiply_and_crt(g["rns_a"], g["rns_b"], moduli=(7, 11, 13))
    assert float(jnp.max(jnp.abs(rns_res - g["rns_out"]))) < 1e-5
    
    print("14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
