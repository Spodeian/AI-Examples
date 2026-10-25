"""Autoencoders: VAE, Beta-VAE, VQ, FSQ, and LFQ — JAX reference implementation.

Topic: 05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq
Golden weights: weights/05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def vae_reparameterize_and_kl(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    eps: jnp.ndarray,
    beta: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reparameterization trick and Beta-VAE analytical KL divergence in JAX."""
    std = jnp.exp(0.5 * logvar)
    z = mu + std * eps
    kl_per_sample = -0.5 * beta * jnp.sum(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)
    return z, kl_per_sample


def finite_scalar_quantization(
    z: jnp.ndarray,
    half_width: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Finite Scalar Quantization (FSQ) in JAX."""
    z_clamped = jnp.clip(z, -half_width, half_width)
    z_round = jnp.round(z_clamped)
    z_q = z + jax.lax.stop_gradient(z_round - z)
    indices = (z_round + half_width).astype(jnp.int32)
    return z_q, indices


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. VAE
    beta = float(golden["beta"][0])
    z, kl = vae_reparameterize_and_kl(golden["mu"], golden["logvar"], golden["eps"], beta)
    err_z = float(jnp.max(jnp.abs(z - golden["z"])))
    err_kl = float(jnp.max(jnp.abs(kl - golden["kl_loss"])))
    assert err_z < 1e-5, f"z max abs error {err_z:.2e}"
    assert err_kl < 1e-5, f"kl max abs error {err_kl:.2e}"

    # 3. FSQ
    half_width = float(golden["half_width"][0])
    z_q, idx = finite_scalar_quantization(golden["z_continuous"], half_width)
    err_zq = float(jnp.max(jnp.abs(z_q - golden["z_q"])))
    err_idx = float(jnp.max(jnp.abs(idx.astype(jnp.float32) - golden["fsq_indices"])))
    assert err_zq < 1e-5, f"z_q max abs error {err_zq:.2e}"
    assert err_idx < 1e-5, f"idx max abs error {err_idx:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
