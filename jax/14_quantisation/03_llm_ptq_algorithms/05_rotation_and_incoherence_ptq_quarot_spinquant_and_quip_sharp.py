"""Rotation and Incoherence PTQ QuaRot Spinquant and Quip Sharp — functional JAX implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp. Run ``python jax/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def construct_hadamard_matrix(dim: int) -> jnp.ndarray:
    """Recursively constructs a normalized Walsh-Hadamard matrix of size dim = 2^k."""
    if dim == 1:
        return jnp.array([[1.0]], dtype=jnp.float32)
    h_sub = construct_hadamard_matrix(dim // 2)
    factor = 1.0 / (2.0 ** 0.5)
    top = jnp.concatenate([h_sub, h_sub], axis=1)
    bottom = jnp.concatenate([h_sub, -h_sub], axis=1)
    return factor * jnp.concatenate([top, bottom], axis=0)


def construct_randomized_hadamard(dim: int, signs: jnp.ndarray) -> jnp.ndarray:
    h = construct_hadamard_matrix(dim)
    return h * jnp.expand_dims(signs, 0)


def cayley_orthogonal_rotation(skew_matrix: jnp.ndarray) -> jnp.ndarray:
    dim = skew_matrix.shape[0]
    eye = jnp.eye(dim, dtype=skew_matrix.dtype)
    a = 0.5 * (skew_matrix - skew_matrix.T)
    r = jnp.linalg.inv(eye + a) @ (eye - a)
    return r


def quantize_per_token_int4(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / 7.0
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    q = jnp.clip(jnp.round(x / scale), -8.0, 7.0)
    return q, scale


def quantize_per_column_int4(w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = jnp.max(jnp.abs(w), axis=0, keepdims=True) / 7.0
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    q = jnp.clip(jnp.round(w / scale), -8.0, 7.0)
    return q, scale


def w4a4_gemm(x: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    q_x, s_x = quantize_per_token_int4(x)
    q_w, s_w = quantize_per_column_int4(weight)
    return (q_x * s_x) @ (q_w * s_w)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Hadamard matrix & Randomized Hadamard rotation
    h16 = construct_hadamard_matrix(16)
    assert float(jnp.max(jnp.abs(h16 - g["hadamard_16"]))) < 1e-5
    q_had = construct_randomized_hadamard(16, g["signs"])
    assert float(jnp.max(jnp.abs(q_had - g["q_hadamard"]))) < 1e-5

    # 3. Rotated activations and weights
    x_rot = g["x_calib"] @ g["q_hadamard"]
    w_rot = g["q_hadamard"].T @ g["w_orig"]
    assert float(jnp.max(jnp.abs(x_rot - g["x_rot"]))) < 1e-5
    assert float(jnp.max(jnp.abs(w_rot - g["w_rot"]))) < 1e-5

    # 4. W4A4 outputs
    out_unrot = w4a4_gemm(g["x_calib"], g["w_orig"]) + g["b_orig"]
    out_quarot = w4a4_gemm(x_rot, w_rot) + g["b_orig"]
    assert float(jnp.max(jnp.abs(out_unrot - g["out_unrotated_w4a4"]))) < 1e-5
    assert float(jnp.max(jnp.abs(out_quarot - g["out_quarot_w4a4"]))) < 1e-5

    print("14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
