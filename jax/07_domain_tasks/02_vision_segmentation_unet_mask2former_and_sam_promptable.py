"""Vision Segmentation Unet Mask2former and SAM Promptable — functional JAX implementation.

Topic: 07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable. Run ``python jax/07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def dice_loss(pred_mask: jnp.ndarray, target_mask: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    intersection = jnp.sum(pred_mask * target_mask, axis=(-2, -1))
    cardinality = jnp.sum(pred_mask**2, axis=(-2, -1)) + jnp.sum(target_mask**2, axis=(-2, -1))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice


def mask2former_masked_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_mask: jnp.ndarray,
) -> jnp.ndarray:
    d = q.shape[-1]
    scores = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d)
    masked_scores = scores + attn_mask
    # numerically stable softmax over last dim
    probs = jax.nn.softmax(masked_scores, axis=-1)
    return probs @ v


def sam_fourier_prompt_embed(
    points: jnp.ndarray,
    gaussian_matrix: jnp.ndarray,
) -> jnp.ndarray:
    proj = 2.0 * jnp.pi * (points @ gaussian_matrix.T)
    return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)


def verify_against_golden() -> None:
    import jax
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Soft Dice
    dice_out = dice_loss(g["pred_mask"], g["target_mask"])
    err_dice = float(jnp.max(jnp.abs(dice_out - g["dice_loss_ref"])))
    assert err_dice < 1e-5, f"dice error {err_dice:.2e}"

    # 2. Mask2Former
    m2f_calc = mask2former_masked_attention(g["q"], g["k"], g["v"], g["attn_mask"])
    err_m2f = float(jnp.max(jnp.abs(m2f_calc - g["m2f_out"])))
    assert err_m2f < 1e-5, f"mask2former error {err_m2f:.2e}"

    # 3. SAM Fourier
    sam_calc = sam_fourier_prompt_embed(g["points"], g["gaussian_matrix"])
    err_sam = float(jnp.max(jnp.abs(sam_calc - g["sam_prompt_embed"])))
    assert err_sam < 1e-5, f"sam prompt error {err_sam:.2e}"

    print("07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
