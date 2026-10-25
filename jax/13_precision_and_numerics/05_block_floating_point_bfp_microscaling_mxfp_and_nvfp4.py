"""Block Floating Point BFP Microscaling MXFP and NVFP4 — functional JAX implementation.

Topic: 13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4. Run ``python jax/13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def quantize_minifloat_core(
    x: jnp.ndarray,
    e_bits: int,
    m_bits: int,
    bias: int,
    max_val: float,
) -> jnp.ndarray:
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


def block_floating_point_quantize(
    x: jnp.ndarray,
    block_size: int = 16,
    mantissa_bits: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    
    amax = jnp.max(jnp.abs(x_reshaped), axis=-1, keepdims=True)
    is_zero = (amax == 0.0)
    amax_safe = jnp.where(is_zero, jnp.ones_like(amax), amax)
    
    shared_exp = jnp.floor(jnp.log2(amax_safe))
    shared_scale = 2.0 ** shared_exp
    
    norm_x = x_reshaped / shared_scale
    max_int = float((1 << (mantissa_bits - 1)) - 1)
    q_mantissa = jnp.clip(jnp.round(norm_x * max_int), -max_int, max_int)
    
    dequant = (q_mantissa / max_int) * shared_scale
    dequant = jnp.where(is_zero, jnp.zeros_like(dequant), dequant)
    
    return dequant.reshape(orig_shape), shared_scale.reshape(-1)


def ocp_mxfp4_quantize(
    x: jnp.ndarray,
    block_size: int = 32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    
    amax = jnp.max(jnp.abs(x_reshaped), axis=-1, keepdims=True)
    is_zero = (amax == 0.0)
    amax_safe = jnp.where(is_zero, jnp.ones_like(amax), amax)
    
    exp_scale = jnp.ceil(jnp.log2(amax_safe / 6.0))
    scale_block = 2.0 ** exp_scale
    scale_block = jnp.where(is_zero, jnp.ones_like(scale_block), scale_block)
    
    elem_scaled = x_reshaped / scale_block
    elem_q = quantize_minifloat_core(elem_scaled, e_bits=2, m_bits=1, bias=1, max_val=6.0)
    
    dequant = elem_q * scale_block
    dequant = jnp.where(is_zero, jnp.zeros_like(dequant), dequant)
    
    return dequant.reshape(orig_shape), scale_block.reshape(-1)


def nvfp4_two_level_quantize(
    x: jnp.ndarray,
    block_size: int = 16,
) -> tuple[jnp.ndarray, float, jnp.ndarray]:
    global_max = float(jnp.max(jnp.abs(x)))
    global_scale = max(1e-8, global_max / 6.0)
    
    x_glob = x / global_scale
    dequant_local, local_scales = ocp_mxfp4_quantize(x_glob, block_size=block_size)
    dequant = dequant_local * global_scale
    
    return dequant, global_scale, local_scales


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. Block Floating Point
    bfp_dequant, bfp_scales = block_floating_point_quantize(g["bfp_input"], block_size=16, mantissa_bits=8)
    err_bfp = float(jnp.max(jnp.abs(bfp_dequant - g["bfp_dequant_ref"])))
    assert err_bfp < 1e-5, f"bfp dequant error {err_bfp:.2e}"
    assert float(jnp.max(jnp.abs(bfp_scales - g["bfp_scales_ref"]))) < 1e-5
    
    # 3. OCP MXFP4
    mx_dequant, mx_scales = ocp_mxfp4_quantize(g["mx_input"], block_size=32)
    err_mx = float(jnp.max(jnp.abs(mx_dequant - g["mx_dequant_ref"])))
    assert err_mx < 1e-5, f"mx dequant error {err_mx:.2e}"
    assert float(jnp.max(jnp.abs(mx_scales - g["mx_scales_ref"]))) < 1e-5
    
    # 4. NVFP4 Two-Level
    nv_dequant, nv_glob, nv_loc = nvfp4_two_level_quantize(g["nv_input"], block_size=16)
    err_nv = float(jnp.max(jnp.abs(nv_dequant - g["nv_dequant_ref"])))
    assert err_nv < 1e-5, f"nv dequant error {err_nv:.2e}"
    assert abs(nv_glob - float(g["nv_global_scale"])) < 1e-5
    assert float(jnp.max(jnp.abs(nv_loc - g["nv_local_scales_ref"]))) < 1e-5
    
    print("13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
