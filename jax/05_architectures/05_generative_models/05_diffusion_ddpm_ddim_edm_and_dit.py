"""Diffusion DDPM Ddim Edm and DiT — functional JAX implementation.

Topic: 05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit. Run ``python jax/05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ddpm_forward_diffuse(x_0: jnp.ndarray, noise: jnp.ndarray, alpha_bar: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(alpha_bar) * x_0 + jnp.sqrt(1.0 - alpha_bar) * noise


def ddim_step(
    x_t: jnp.ndarray,
    eps_pred: jnp.ndarray,
    alpha_bar_t: jnp.ndarray,
    alpha_bar_prev: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    sqrt_ab_t = jnp.sqrt(alpha_bar_t)
    sqrt_one_minus_ab_t = jnp.sqrt(1.0 - alpha_bar_t)
    x_0_pred = (x_t - sqrt_one_minus_ab_t * eps_pred) / sqrt_ab_t

    sqrt_ab_prev = jnp.sqrt(alpha_bar_prev)
    sqrt_one_minus_ab_prev = jnp.sqrt(1.0 - alpha_bar_prev)
    x_prev = sqrt_ab_prev * x_0_pred + sqrt_one_minus_ab_prev * eps_pred
    return x_0_pred, x_prev


def classifier_free_guidance(eps_uncond: jnp.ndarray, eps_cond: jnp.ndarray, cfg_scale: float) -> jnp.ndarray:
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


def adaln_zero_modulation(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    alpha: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / jnp.sqrt(var + eps)
    h = norm * (1.0 + gamma) + beta
    return alpha * h


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. DDPM
    x_t = ddpm_forward_diffuse(g["x_0"], g["noise"], g["alpha_bar_t"])
    err_xt = float(jnp.max(jnp.abs(x_t - g["x_t"])))
    assert err_xt < 1e-5, f"x_t error {err_xt:.2e}"

    # 3. DDIM
    x_0_pred, x_prev = ddim_step(g["x_t"], g["eps_pred"], g["alpha_bar_t"], g["alpha_bar_prev"])
    err_x0 = float(jnp.max(jnp.abs(x_0_pred - g["x_0_pred"])))
    err_xp = float(jnp.max(jnp.abs(x_prev - g["x_prev"])))
    assert err_x0 < 1e-5, f"x_0_pred error {err_x0:.2e}"
    assert err_xp < 1e-5, f"x_prev error {err_xp:.2e}"

    # 4. CFG
    cfg_scale = float(g["cfg_scale"][0])
    cfg_out = classifier_free_guidance(g["eps_uncond"], g["eps_cond"], cfg_scale)
    err_cfg = float(jnp.max(jnp.abs(cfg_out - g["cfg_out"])))
    assert err_cfg < 1e-5, f"cfg_out error {err_cfg:.2e}"

    # 5. DiT adaLN-Zero
    adaln_out = adaln_zero_modulation(g["x_dit"], g["gamma"], g["beta_dit"], g["alpha_dit"])
    err_adaln = float(jnp.max(jnp.abs(adaln_out - g["adaln_out"])))
    assert err_adaln < 1e-5, f"adaln_out error {err_adaln:.2e}"

    print("05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
