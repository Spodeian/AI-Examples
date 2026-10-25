"""Multimodal Fusion FiLM, Perceiver Resampler, and VLM Connectors — functional JAX implementation.

Topic: 05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors. Run ``python jax/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def film_modulation(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    return gamma * x + beta


def perceiver_resampler(
    latent_queries: jnp.ndarray,
    visual_tokens: jnp.ndarray,
    w_q: jnp.ndarray,
    w_k: jnp.ndarray,
    w_v: jnp.ndarray,
) -> jnp.ndarray:
    d = latent_queries.shape[-1]
    q = latent_queries @ w_q
    k = visual_tokens @ w_k
    v = visual_tokens @ w_v

    scores = (q @ jnp.swapaxes(k, -2, -1)) / math.sqrt(d)
    attn = jax.nn.softmax(scores, axis=-1)
    return attn @ v


def vlm_spatial_pool_connector(
    patches: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
) -> jnp.ndarray:
    b, num_patches, d = patches.shape
    unshuffled = jnp.reshape(patches, (b, 1, num_patches * d))
    h = jax.nn.gelu(unshuffled @ w1 + b1, approximate=False)
    return h @ w2 + b2


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    h_film = film_modulation(g["x_feat"], g["gamma"], g["beta"])
    assert float(jnp.max(jnp.abs(h_film - g["h_film"]))) < 1e-5, "h_film parity mismatch"

    h_resampled = perceiver_resampler(g["latent_queries"], g["x_feat"], g["w_q"], g["w_k"], g["w_v"])
    assert float(jnp.max(jnp.abs(h_resampled - g["h_resampled"]))) < 1e-5, "h_resampled parity mismatch"

    h_vlm = vlm_spatial_pool_connector(g["patches"], g["w1"], g["b1"], g["w2"], g["b2"])
    assert float(jnp.max(jnp.abs(h_vlm - g["h_vlm"]))) < 1e-5, "h_vlm parity mismatch"

    print("05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
