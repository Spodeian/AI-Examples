"""Robotics Embodied Ai Diffusion Policy and Act — JAX reference implementation.

Topic: 07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act
Golden weights: weights/07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def temporal_ensemble_actions(action_preds: jnp.ndarray, decay: float = 0.1) -> jnp.ndarray:
    """Action Chunking with Ensembling (ACT) temporal weighting in JAX."""
    k = action_preds.shape[1]
    steps = jnp.arange(k, dtype=action_preds.dtype)
    weights = jnp.exp(-decay * steps)
    weights = weights / jnp.sum(weights)
    return jnp.sum(action_preds * weights.reshape(1, -1, 1), axis=1)


def diffusion_policy_ddpm_step(
    action: jnp.ndarray,
    noise_pred: jnp.ndarray,
    noise_z: jnp.ndarray,
    alpha_t: float,
    alpha_bar_t: float,
    sigma_t: float,
) -> jnp.ndarray:
    """Single reverse DDPM denoising step for continuous action diffusion in JAX."""
    coeff = (1.0 - alpha_t) / (jnp.sqrt(1.0 - alpha_bar_t))
    mu = (1.0 / jnp.sqrt(alpha_t)) * (action - coeff * noise_pred)
    return mu + sigma_t * noise_z


def act_cvae_kl_loss(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """ACT CVAE Gaussian latent regularization loss in JAX."""
    kl = -0.5 * jnp.sum(1.0 + logvar - mu ** 2 - jnp.exp(logvar), axis=-1)
    return jnp.mean(kl)


def robot_action_safety_clamp(
    actions: jnp.ndarray,
    a_min: jnp.ndarray,
    a_max: jnp.ndarray,
) -> jnp.ndarray:
    """Clamps continuous actions within physical safety envelope in JAX."""
    return jnp.clip(actions, a_min.reshape(1, 1, -1), a_max.reshape(1, 1, -1))


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    raw = load_file(str(path))
    tensors = {k: jnp.array(v) for k, v in raw.items()}

    # Anchor check
    out = linear_anchor(tensors["test_input"], tensors["weight"], tensors["bias"])
    err = float(jnp.max(jnp.abs(out - tensors["test_output"])))
    assert err < 1e-5, f"test_output parity error: {err}"

    # 1. Temporal Ensemble Actions
    ens = temporal_ensemble_actions(tensors["action_preds"], decay=0.1)
    err_ens = float(jnp.max(jnp.abs(ens - tensors["ensembled_ref"])))
    assert err_ens < 1e-5, f"ensemble actions parity error: {err_ens}"

    # 2. Diffusion Policy DDPM Step
    denoised = diffusion_policy_ddpm_step(
        tensors["action_t"],
        tensors["noise_pred"],
        tensors["noise_z"],
        alpha_t=0.99,
        alpha_bar_t=0.85,
        sigma_t=0.05,
    )
    err_denoised = float(jnp.max(jnp.abs(denoised - tensors["denoised_ref"])))
    assert err_denoised < 1e-5, f"denoised actions parity error: {err_denoised}"

    # 3. ACT CVAE KL Loss
    kl = act_cvae_kl_loss(tensors["cvae_mu"], tensors["cvae_logvar"])
    err_kl = float(jnp.abs(kl - tensors["kl_loss_ref"]))
    assert err_kl < 1e-5, f"kl loss parity error: {err_kl}"

    # 4. Robot Action Safety Clamp
    clamped = robot_action_safety_clamp(tensors["raw_actions"], tensors["a_min"], tensors["a_max"])
    err_clamp = float(jnp.max(jnp.abs(clamped - tensors["clamped_ref"])))
    assert err_clamp < 1e-5, f"clamped actions parity error: {err_clamp}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
