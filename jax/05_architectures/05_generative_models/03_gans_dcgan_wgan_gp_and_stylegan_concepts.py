"""GANs Dcgan Wgan GP and Stylegan Concepts — functional JAX implementation.

Topic: 05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts. Run ``python jax/05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def adain(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Adaptive Instance Normalization in pure JAX."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / jnp.sqrt(var + eps)
    return norm * scale + bias


def stylegan2_weight_demodulation(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    styles: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """StyleGAN2 Weight Demodulation in pure JAX."""
    # weights: [C_out, C_in], styles: [B, C_in]
    w_mod = jnp.expand_dims(weights, 0) * jnp.expand_dims(styles, 1)  # [B, C_out, C_in]
    demod_factors = 1.0 / jnp.sqrt(jnp.sum(jnp.square(w_mod), axis=-1, keepdims=True) + eps)
    w_demod = w_mod * demod_factors
    y = jnp.squeeze(w_demod @ jnp.expand_dims(x, -1), -1)
    return y


def wgan_critic_loss(real_scores: jnp.ndarray, fake_scores: jnp.ndarray) -> jnp.ndarray:
    """WGAN critic loss in pure JAX."""
    return jnp.mean(fake_scores) - jnp.mean(real_scores)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. AdaIN
    adain_out = adain(g["x_adain"], g["scale"], g["bias_adain"])
    err_adain = float(jnp.max(jnp.abs(adain_out - g["adain_out"])))
    assert err_adain < 1e-5, f"adain error {err_adain:.2e}"

    # 3. Demod
    demod_out = stylegan2_weight_demodulation(g["x_feat"], g["w_conv"], g["s_style"])
    err_demod = float(jnp.max(jnp.abs(demod_out - g["demod_out"])))
    assert err_demod < 1e-5, f"demod error {err_demod:.2e}"

    # 4. WGAN loss
    wgan_loss = wgan_critic_loss(g["real_scores"], g["fake_scores"])
    err_wgan = float(jnp.max(jnp.abs(wgan_loss - g["wgan_loss"][0])))
    assert err_wgan < 1e-5, f"wgan loss error {err_wgan:.2e}"

    print("05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
