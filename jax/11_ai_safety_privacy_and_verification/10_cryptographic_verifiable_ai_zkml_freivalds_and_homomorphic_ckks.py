"""Cryptographic Verifiable AI ZKML Freivalds and Homomorphic CKKS — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks. Run ``python jax/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def freivalds_verify_residual(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    r: jnp.ndarray,
) -> jnp.ndarray:
    """Freivalds' randomized matrix multiplication check: v = A(Br) - Cr in O(n^2)."""
    return a @ (b @ r) - c @ r


def ckks_degree2_activation(
    x: jnp.ndarray,
    a: float,
    b: float,
    c: float,
) -> jnp.ndarray:
    """CKKS / CryptoNets degree-2 polynomial activation."""
    return float(a) * (x ** 2) + float(b) * x + float(c)


def ckks_degree4_activation(
    x: jnp.ndarray,
    c4: float,
    c2: float,
    c0: float,
) -> jnp.ndarray:
    """CKKS degree-4 polynomial activation."""
    x2 = x ** 2
    return float(c4) * (x2 ** 2) + float(c2) * x2 + float(c0)


def zkml_fixed_point_linear(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    """ZKML fixed-point quantized linear layer."""
    y = x @ weight + bias
    return jnp.round(y * float(scale)) / float(scale)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Freivalds valid
    v_val = freivalds_verify_residual(g["mat_a"], g["mat_b"], g["mat_c"], g["rand_r"])
    err = float(jnp.max(jnp.abs(v_val - g["freivalds_valid_ref"])))
    assert err < 1e-5, f"freivalds_valid error {err:.2e}"

    # 3. Freivalds tampered
    v_tamp = freivalds_verify_residual(g["mat_a"], g["mat_b"], g["mat_c_tampered"], g["rand_r"])
    err = float(jnp.max(jnp.abs(v_tamp - g["freivalds_tampered_ref"])))
    assert err < 1e-5, f"freivalds_tampered error {err:.2e}"

    # 4. CKKS poly2
    p2 = ckks_degree2_activation(g["x_poly"], a=0.125, b=0.5, c=0.25)
    err = float(jnp.max(jnp.abs(p2 - g["poly2_out_ref"])))
    assert err < 1e-5, f"poly2 error {err:.2e}"

    # 5. CKKS poly4
    p4 = ckks_degree4_activation(g["x_poly"], c4=-0.005, c2=0.2, c0=0.15)
    err = float(jnp.max(jnp.abs(p4 - g["poly4_out_ref"])))
    assert err < 1e-5, f"poly4 error {err:.2e}"

    # 6. ZKML fixed-point
    zk_out = zkml_fixed_point_linear(g["zk_x"], g["zk_w"], g["zk_b"], scale=16.0)
    err = float(jnp.max(jnp.abs(zk_out - g["zk_out_ref"])))
    assert err < 1e-5, f"zk_out error {err:.2e}"

    print("11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
