"""Visual Autoregressive MaskGIT, VAR, and MAR — functional JAX implementation.

Topic: 05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar. Run ``python jax/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def maskgit_cosine_schedule(r: jnp.ndarray) -> jnp.ndarray:
    return jnp.cos(0.5 * math.pi * r)


def maskgit_apply_mask(x: jnp.ndarray, mask: jnp.ndarray, mask_token: jnp.ndarray) -> jnp.ndarray:
    return (1.0 - mask) * x + mask * mask_token


def var_coarse_to_fine_upsample(
    coarse_map: jnp.ndarray,
    repeats: int,
    w_scale: jnp.ndarray,
    b_scale: jnp.ndarray,
) -> jnp.ndarray:
    upsampled = jnp.repeat(coarse_map, repeats, axis=1)
    residual = jax.nn.gelu(upsampled @ w_scale + b_scale, approximate=False)
    return upsampled + residual


def mar_masked_diffusion_loss(
    v_pred: jnp.ndarray,
    v_target: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    sq_err = (v_pred - v_target) ** 2
    masked_sq_err = sq_err * mask
    loss = jnp.sum(masked_sq_err) / (jnp.sum(mask) * v_pred.shape[-1] + 1e-6)
    return loss.reshape((1,))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    gamma = maskgit_cosine_schedule(g["r"])
    assert float(jnp.max(jnp.abs(gamma - g["gamma"]))) < 1e-5, "gamma parity mismatch"

    masked_tokens = maskgit_apply_mask(g["x_tokens"], g["mask"], g["mask_token"])
    assert float(jnp.max(jnp.abs(masked_tokens - g["masked_tokens"]))) < 1e-5, "masked_tokens parity mismatch"

    fine_map = var_coarse_to_fine_upsample(g["coarse_map"], repeats=4, w_scale=g["w_scale"], b_scale=g["b_scale"])
    assert float(jnp.max(jnp.abs(fine_map - g["fine_map"]))) < 1e-5, "fine_map parity mismatch"

    mar_loss = mar_masked_diffusion_loss(g["v_pred"], g["v_target"], g["mask"])
    assert float(jnp.max(jnp.abs(mar_loss - g["mar_loss"]))) < 1e-5, "mar_loss parity mismatch"

    print("05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
