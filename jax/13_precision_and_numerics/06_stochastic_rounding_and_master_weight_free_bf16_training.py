"""Stochastic Rounding and Master Weight Free BF16 Training — functional JAX implementation.

Topic: 13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training. Run ``python jax/13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def stochastic_rounding_bf16(x: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
    x_abs = jnp.abs(x)
    is_zero = (x_abs == 0.0)
    x_safe = jnp.where(is_zero, jnp.ones_like(x_abs), x_abs)
    
    e = jnp.floor(jnp.log2(x_safe))
    ulp = 2.0 ** (e - 7.0)
    
    step_ratio = x_safe / ulp
    floor_ratio = jnp.floor(step_ratio)
    frac = step_ratio - floor_ratio
    
    round_up = (noise < frac)
    q_mag = jnp.where(round_up, (floor_ratio + 1.0) * ulp, floor_ratio * ulp)
    return jnp.where(is_zero, jnp.zeros_like(x), jnp.sign(x) * q_mag)


def deterministic_rtne_bf16(x: jnp.ndarray) -> jnp.ndarray:
    x_abs = jnp.abs(x)
    is_zero = (x_abs == 0.0)
    x_safe = jnp.where(is_zero, jnp.ones_like(x_abs), x_abs)
    
    e = jnp.floor(jnp.log2(x_safe))
    ulp = 2.0 ** (e - 7.0)
    
    step_ratio = x_safe / ulp
    q_mag = jnp.round(step_ratio) * ulp
    return jnp.where(is_zero, jnp.zeros_like(x), jnp.sign(x) * q_mag)


def master_weight_free_sgd_step(
    w_bf16: jnp.ndarray,
    grad: jnp.ndarray,
    lr: float,
    noise: jnp.ndarray,
    use_stochastic_rounding: bool = True,
) -> jnp.ndarray:
    target_w = w_bf16 - float(lr) * grad
    if use_stochastic_rounding:
        return stochastic_rounding_bf16(target_w, noise)
    return deterministic_rtne_bf16(target_w)


def kahan_accumulate_step(
    accum: jnp.ndarray,
    val: jnp.ndarray,
    comp: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    y = val - comp
    t = accum + y
    comp_new = (t - accum) - y
    accum_new = t
    return accum_new, comp_new


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. Stochastic Rounding BF16
    sr_out = stochastic_rounding_bf16(g["x_float"], g["noise"])
    err_sr = float(jnp.max(jnp.abs(sr_out - g["sr_out_ref"])))
    assert err_sr < 1e-5, f"sr_out error {err_sr:.2e}"
    
    rtne_out = deterministic_rtne_bf16(g["x_float"])
    err_rtne = float(jnp.max(jnp.abs(rtne_out - g["rtne_out_ref"])))
    assert err_rtne < 1e-5, f"rtne_out error {err_rtne:.2e}"
    
    # 3. Master-weight-free SGD
    sgd_sr = master_weight_free_sgd_step(g["w_init"], g["grad_tiny"], lr=0.01, noise=g["noise"], use_stochastic_rounding=True)
    err_sgd_sr = float(jnp.max(jnp.abs(sgd_sr - g["sgd_sr_ref"])))
    assert err_sgd_sr < 1e-5, f"sgd_sr error {err_sgd_sr:.2e}"
    
    sgd_rtne = master_weight_free_sgd_step(g["w_init"], g["grad_tiny"], lr=0.01, noise=g["noise"], use_stochastic_rounding=False)
    err_sgd_rtne = float(jnp.max(jnp.abs(sgd_rtne - g["sgd_rtne_ref"])))
    assert err_sgd_rtne < 1e-5, f"sgd_rtne error {err_sgd_rtne:.2e}"
    
    # 4. Kahan accumulation
    k_accum, k_comp = kahan_accumulate_step(g["k_accum"], g["k_val"], g["k_comp"])
    err_k_accum = float(jnp.max(jnp.abs(k_accum - g["k_accum_ref"])))
    err_k_comp = float(jnp.max(jnp.abs(k_comp - g["k_comp_ref"])))
    assert err_k_accum < 1e-5, f"k_accum error {err_k_accum:.2e}"
    assert err_k_comp < 1e-5, f"k_comp error {err_k_comp:.2e}"
    
    print("13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
