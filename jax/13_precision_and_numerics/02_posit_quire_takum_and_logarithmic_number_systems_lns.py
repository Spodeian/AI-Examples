"""Posit Quire Takum and Logarithmic Number Systems LNS — functional JAX implementation.

Topic: 13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns. Run ``python jax/13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def posit_tapered_quantize(x: jnp.ndarray, n_bits: int = 8, es: int = 1) -> jnp.ndarray:
    is_zero = (x == 0.0)
    x_safe = jnp.where(is_zero, 1.0, jnp.abs(x))
    log2_x = jnp.floor(jnp.log2(x_safe))
    useed = 2.0 ** float(es)
    k = jnp.floor(log2_x / useed)
    regime_bits = jnp.where(k >= 0.0, k + 2.0, -k + 1.0)
    frac_bits = jnp.clip(float(n_bits) - 1.0 - regime_bits - float(es), a_min=0.0, a_max=float(n_bits - 2))
    step = 2.0 ** (log2_x - frac_bits)
    q = jnp.round(x_safe / step) * step
    out = jnp.sign(x) * q
    return jnp.where(is_zero, 0.0, out)


def lns_multiplication(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    sign_out = jnp.sign(a) * jnp.sign(b)
    log_a = jnp.log2(jnp.abs(a) + float(eps))
    log_b = jnp.log2(jnp.abs(b) + float(eps))
    log_out = log_a + log_b
    return sign_out * (2.0 ** log_out)


def lns_gaussian_addition(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    log_a = jnp.log2(jnp.clip(a, a_min=float(eps)))
    log_b = jnp.log2(jnp.clip(b, a_min=float(eps)))
    diff = jnp.abs(log_a - log_b)
    correction = jnp.log2(1.0 + (2.0 ** (-diff)))
    log_out = jnp.maximum(log_a, log_b) + correction
    return 2.0 ** log_out


def quire_dot_product_sim(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(a.astype(jnp.float64) * b.astype(jnp.float64), axis=-1).astype(jnp.float32)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Posit Quantization
    posit = posit_tapered_quantize(g["posit_in"], n_bits=8, es=1)
    assert float(jnp.max(jnp.abs(posit - g["posit_ref"]))) < 1e-5

    # 2. LNS Multiplication
    lns_mul = lns_multiplication(g["lns_a"], g["lns_b"])
    assert float(jnp.max(jnp.abs(lns_mul - g["lns_mul_ref"]))) < 1e-5

    # 3. LNS Addition
    lns_add = lns_gaussian_addition(g["add_a"], g["add_b"])
    assert float(jnp.max(jnp.abs(lns_add - g["lns_add_ref"]))) < 1e-5

    # 4. Quire Dot Product
    quire = quire_dot_product_sim(g["quire_mat"], g["quire_vec"])
    assert float(jnp.max(jnp.abs(quire - g["quire_ref"]))) < 1e-5

    print("13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
