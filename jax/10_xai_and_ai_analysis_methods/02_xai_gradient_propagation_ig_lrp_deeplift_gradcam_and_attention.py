"""XAI Gradient Propagation IG LRP DeepLIFT Grad-CAM and Attention — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention. Run ``python jax/10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def integrated_gradients(
    x: jnp.ndarray,
    x_baseline: jnp.ndarray,
    interpolated_grads: jnp.ndarray,
) -> jnp.ndarray:
    diff = x - x_baseline
    avg_grad = jnp.mean(interpolated_grads, axis=0)
    return diff * avg_grad


def lrp_epsilon_rule(
    activations: jnp.ndarray,
    weights: jnp.ndarray,
    relevance_next: jnp.ndarray,
    epsilon: float = 1e-4,
) -> jnp.ndarray:
    z_ij = jnp.expand_dims(activations, -1) * weights
    z_j = jnp.sum(z_ij, axis=0)
    sign_z_j = jnp.where(z_j >= 0.0, 1.0, -1.0)
    z_j_stab = z_j + epsilon * sign_z_j
    s = relevance_next / z_j_stab
    return jnp.squeeze(z_ij @ jnp.expand_dims(s, -1), -1)


def grad_cam_2d(
    activations: jnp.ndarray,
    grads: jnp.ndarray,
) -> jnp.ndarray:
    alpha = jnp.mean(grads, axis=(-2, -1))
    cam = jnp.sum(jnp.reshape(alpha, (-1, 1, 1)) * activations, axis=0)
    return jnp.maximum(cam, 0.0)


def cosine_similarity(
    s_orig: jnp.ndarray,
    s_rand: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    dot = jnp.sum(s_orig * s_rand)
    norm_orig = jnp.sqrt(jnp.sum(s_orig * s_orig))
    norm_rand = jnp.sqrt(jnp.sum(s_rand * s_rand))
    return dot / (norm_orig * norm_rand + eps)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Integrated Gradients
    ig = integrated_gradients(g["ig_x"], g["ig_baseline"], g["ig_grads"])
    assert float(jnp.max(jnp.abs(ig - g["ig_ref"]))) < 1e-5, "IG mismatch"

    # LRP
    lrp = lrp_epsilon_rule(g["lrp_acts"], g["lrp_weights"], g["lrp_rel_next"], epsilon=1e-4)
    assert float(jnp.max(jnp.abs(lrp - g["lrp_ref"]))) < 1e-5, "LRP mismatch"

    # Grad-CAM
    cam = grad_cam_2d(g["cam_acts"], g["cam_grads"])
    assert float(jnp.max(jnp.abs(cam - g["cam_ref"]))) < 1e-5, "Grad-CAM mismatch"

    # Adebayo
    sim = cosine_similarity(g["adebayo_orig"], g["adebayo_rand"])
    assert float(jnp.max(jnp.abs(sim - g["adebayo_sim_ref"]))) < 1e-5, "Adebayo similarity mismatch"

    print("10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
