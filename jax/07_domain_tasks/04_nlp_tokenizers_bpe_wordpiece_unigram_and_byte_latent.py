"""Nlp Tokenizers Bpe Wordpiece Unigram and Byte Latent — functional JAX implementation.

Topic: 07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent. Run ``python jax/07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def wordpiece_pair_score(
    bigram_counts: jnp.ndarray,
    unigram_counts: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    outer = jnp.expand_dims(unigram_counts, 1) * jnp.expand_dims(unigram_counts, 0)
    return bigram_counts / (outer + eps)


def byte_patch_pooling(
    byte_embeddings: jnp.ndarray,
    patch_size: int = 4,
) -> jnp.ndarray:
    b, t, d = byte_embeddings.shape
    num_patches = t // patch_size
    reshaped = byte_embeddings.reshape((b, num_patches, patch_size, d))
    return jnp.mean(reshaped, axis=2)


def entropy_boundary_detector(
    probs: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    log_probs = jnp.log2(probs + eps)
    return -jnp.sum(probs * log_probs, axis=-1)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. WordPiece
    wp_calc = wordpiece_pair_score(g["bigram_counts"], g["unigram_counts"])
    err_wp = float(jnp.max(jnp.abs(wp_calc - g["wp_scores"])))
    assert err_wp < 1e-5, f"wordpiece score error {err_wp:.2e}"

    # 2. Byte Patch Pooling
    patch_calc = byte_patch_pooling(g["byte_embeds"], patch_size=4)
    err_patch = float(jnp.max(jnp.abs(patch_calc - g["patch_embeds"])))
    assert err_patch < 1e-5, f"patch pooling error {err_patch:.2e}"

    # 3. Dynamic Boundary Entropy
    ent_calc = entropy_boundary_detector(g["probs"])
    err_ent = float(jnp.max(jnp.abs(ent_calc - g["entropy_ref"])))
    assert err_ent < 1e-5, f"boundary entropy error {err_ent:.2e}"

    print("07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
