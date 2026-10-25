"""Neural Audio Codecs Encodec RVQ and Quantizer Dropout — functional JAX implementation.

Topic: 05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout. Run ``python jax/05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def snake_activation(x: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Snake periodic activation function: Snake(x) = x + (1 / alpha) * sin^2(alpha * x)."""
    return x + (1.0 / alpha) * jnp.square(jnp.sin(alpha * x))


def residual_vector_quantization(
    z: jnp.ndarray,
    codebooks: jnp.ndarray,
    n_active: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Residual Vector Quantization in pure JAX."""
    N_q, K, D = codebooks.shape
    active = N_q if n_active is None else n_active

    residual = z
    z_q = jnp.zeros_like(z)
    all_indices = []

    for k in range(N_q):
        cb = codebooks[k]  # [K, D]
        dists = (
            jnp.sum(jnp.square(residual), axis=-1, keepdims=True)
            - 2.0 * (residual @ cb.T)
            + jnp.expand_dims(jnp.sum(jnp.square(cb), axis=-1), 0)
        )
        idx = jnp.argmin(dists, axis=-1)
        all_indices.append(idx)
        quantized_k = cb[idx]

        if k < active:
            z_q = z_q + quantized_k
            residual = residual - quantized_k
        else:
            residual = residual - quantized_k

    indices_tensor = jnp.stack(all_indices, axis=0)
    return z_q, indices_tensor, residual


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Snake
    snake_out = snake_activation(g["x_audio"], g["alpha"])
    err_snake = float(jnp.max(jnp.abs(snake_out - g["snake_out"])))
    assert err_snake < 1e-5, f"snake error {err_snake:.2e}"

    # 3. RVQ
    n_active = int(g["n_active"][0])
    z_q, indices, residual = residual_vector_quantization(g["z_in"], g["codebooks"], n_active=n_active)
    err_zq = float(jnp.max(jnp.abs(z_q - g["z_q"])))
    assert err_zq < 1e-5, f"z_q error {err_zq:.2e}"
    err_idx = float(jnp.max(jnp.abs(indices - g["indices"])))
    assert err_idx < 1e-5, f"indices error {err_idx:.2e}"
    err_res = float(jnp.max(jnp.abs(residual - g["residual"])))
    assert err_res < 1e-5, f"residual error {err_res:.2e}"

    print("05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
