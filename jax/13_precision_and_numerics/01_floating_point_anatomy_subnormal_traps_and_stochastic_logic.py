"""Floating Point Anatomy Subnormal Traps and Stochastic Logic — functional JAX implementation.

Topic: 13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic. Run ``python jax/13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ftz_daz_simulate(x: jnp.ndarray, threshold: float = 1e-4) -> jnp.ndarray:
    return jnp.where(jnp.abs(x) < float(threshold), 0.0, x)


def fp32_to_bf16_rtne(x: jnp.ndarray) -> jnp.ndarray:
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def stochastic_unipolar_mul(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    u1: jnp.ndarray,
    u2: jnp.ndarray,
) -> jnp.ndarray:
    b1 = (u1 < jnp.expand_dims(p1, 1)).astype(jnp.float32)
    b2 = (u2 < jnp.expand_dims(p2, 1)).astype(jnp.float32)
    b_mul = b1 * b2
    return jnp.mean(b_mul, axis=-1)


def stochastic_bipolar_mul(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    u1: jnp.ndarray,
    u2: jnp.ndarray,
) -> jnp.ndarray:
    p1 = (x1 + 1.0) * 0.5
    p2 = (x2 + 1.0) * 0.5
    b1 = (u1 < jnp.expand_dims(p1, 1)).astype(jnp.float32)
    b2 = (u2 < jnp.expand_dims(p2, 1)).astype(jnp.float32)
    b_xnor = b1 * b2 + (1.0 - b1) * (1.0 - b2)
    p_out = jnp.mean(b_xnor, axis=-1)
    return 2.0 * p_out - 1.0


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. FTZ
    ftz = ftz_daz_simulate(g["raw_vals"], threshold=1e-4)
    assert float(jnp.max(jnp.abs(ftz - g["ftz_ref"]))) < 1e-5

    # 2. BF16
    bf16 = fp32_to_bf16_rtne(g["bf16_in"])
    assert float(jnp.max(jnp.abs(bf16 - g["bf16_ref"]))) < 1e-5

    # 3. Unipolar Stochastic
    unipolar = stochastic_unipolar_mul(g["p1"], g["p2"], g["u1"], g["u2"])
    assert float(jnp.max(jnp.abs(unipolar - g["unipolar_ref"]))) < 1e-5

    # 4. Bipolar Stochastic
    bipolar = stochastic_bipolar_mul(g["x1"], g["x2"], g["u1"], g["u2"])
    assert float(jnp.max(jnp.abs(bipolar - g["bipolar_ref"]))) < 1e-5

    print("13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
