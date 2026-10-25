"""Logarithmic PoT APoT μ-Law and Posit Quantization — functional JAX implementation.

Topic: 14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization. Run ``python jax/14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def pot_quantize(x: jnp.ndarray, num_bits: int = 4) -> jnp.ndarray:
    x_abs = jnp.abs(x)
    max_p = float((1 << (num_bits - 1)) - 2)
    threshold = 2.0 ** (-max_p - 1.0)
    
    is_zero = (x_abs < threshold)
    x_safe = jnp.where(is_zero, jnp.ones_like(x_abs), x_abs)
    
    p = jnp.clip(jnp.round(-jnp.log2(x_safe)), 0.0, max_p)
    q = jnp.sign(x) * (2.0 ** (-p))
    return jnp.where(is_zero, jnp.zeros_like(x), q)


def apot_quantize(x: jnp.ndarray) -> jnp.ndarray:
    levels = jnp.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0])
    x_abs = jnp.clip(jnp.abs(x), 0.0, 1.0)
    
    diff = jnp.abs(jnp.expand_dims(x_abs, -1) - levels)
    idx = jnp.argmin(diff, axis=-1)
    q_mag = levels[idx]
    return jnp.sign(x) * q_mag


def mu_law_compand_expand(
    x: jnp.ndarray,
    mu: float = 255.0,
    num_bits: int = 8,
) -> jnp.ndarray:
    x_clamped = jnp.clip(x, -1.0, 1.0)
    ln_1_mu = math.log(1.0 + float(mu))
    
    y = jnp.sign(x_clamped) * jnp.log(1.0 + float(mu) * jnp.abs(x_clamped)) / ln_1_mu
    
    qmax = float((1 << num_bits) - 1)
    y_norm = (y + 1.0) * 0.5
    y_q = jnp.round(y_norm * qmax) / qmax
    y_quant = y_q * 2.0 - 1.0
    
    x_hat = jnp.sign(y_quant) * ((1.0 + float(mu)) ** jnp.abs(y_quant) - 1.0) / float(mu)
    return x_hat


def base_sqrt2_log_quantize(x: jnp.ndarray) -> jnp.ndarray:
    x_abs = jnp.abs(x)
    threshold = 2.0 ** (-7.5)
    is_zero = (x_abs < threshold)
    x_safe = jnp.where(is_zero, jnp.ones_like(x_abs), x_abs)
    
    k = jnp.clip(jnp.round(2.0 * jnp.log2(x_safe)), -14.0, 0.0)
    q = jnp.sign(x) * (2.0 ** (k * 0.5))
    return jnp.where(is_zero, jnp.zeros_like(x), q)


def posit4_quantize(x: jnp.ndarray) -> jnp.ndarray:
    levels = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0])
    x_abs = jnp.clip(jnp.abs(x), 0.0, 4.0)
    
    diff = jnp.abs(jnp.expand_dims(x_abs, -1) - levels)
    idx = jnp.argmin(diff, axis=-1)
    q_mag = levels[idx]
    return jnp.sign(x) * q_mag


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. PoT Quantization
    pot_out = pot_quantize(g["raw_pot"], num_bits=4)
    assert float(jnp.max(jnp.abs(pot_out - g["pot_ref"]))) < 1e-5
    
    # 3. APoT Quantization
    apot_out = apot_quantize(g["raw_apot"])
    assert float(jnp.max(jnp.abs(apot_out - g["apot_ref"]))) < 1e-5
    
    # 4. Mu-Law Companding
    mulaw_out = mu_law_compand_expand(g["raw_audio"], mu=255.0, num_bits=8)
    assert float(jnp.max(jnp.abs(mulaw_out - g["mulaw_ref"]))) < 1e-5
    
    # 5. Base-sqrt2 Log Quantization
    sqrt2_out = base_sqrt2_log_quantize(g["raw_sqrt2"])
    assert float(jnp.max(jnp.abs(sqrt2_out - g["sqrt2_ref"]))) < 1e-5
    
    # 6. Posit4 Quantization
    posit4_out = posit4_quantize(g["raw_posit"])
    assert float(jnp.max(jnp.abs(posit4_out - g["posit4_ref"]))) < 1e-5
    
    print("14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
