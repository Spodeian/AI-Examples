"""Continuous Control Ddpg Td3 and SAC — JAX reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.safetensors

``python jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.dot(x, weight) + bias


def td3_target_smoothing(
    target_actions: jax.Array,
    noise: jax.Array,
    noise_clip: float = 0.5,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> jax.Array:
    """TD3 Target Action Smoothing."""
    clipped_noise = jnp.clip(noise, -noise_clip, noise_clip)
    return jnp.clip(target_actions + clipped_noise, action_min, action_max)


def td3_clipped_critic_target(
    rewards: jax.Array,
    dones: jax.Array,
    q1_target: jax.Array,
    q2_target: jax.Array,
    gamma: float = 0.99,
) -> jax.Array:
    """TD3 Clipped Double-Q Bellman Target."""
    min_q = jnp.minimum(q1_target, q2_target)
    return rewards + gamma * (1.0 - dones) * min_q


def sac_squashed_gaussian_log_prob(
    u: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """SAC Squashed Gaussian policy action and log-probability."""
    std = jnp.exp(log_std)
    var = std ** 2
    log_scale = 2.0 * log_std
    gaussian_log_p = -0.5 * jnp.sum(
        ((u - mean) ** 2) / var + log_scale + math.log(2.0 * math.pi),
        axis=-1,
    )

    actions = jnp.tanh(u)
    jacobian = jnp.sum(jnp.log(jnp.clip(1.0 - actions ** 2, a_min=eps)), axis=-1)
    log_probs = gaussian_log_p - jacobian
    return actions, log_probs


def sac_temperature_loss(
    log_alpha: jax.Array,
    log_probs: jax.Array,
    target_entropy: float,
) -> jax.Array:
    """SAC Temperature Auto-Tuning Loss."""
    alpha = jnp.exp(log_alpha)
    return -jnp.mean(alpha * (log_probs + target_entropy))


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.asarray(v) for k, v in raw.items()}

    # 1. Linear anchor
    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear anchor max abs error {err:.2e}"

    # 2. TD3 Target Smoothing
    smoothed = td3_target_smoothing(golden["target_actions"], golden["noise"], noise_clip=0.5, action_min=-1.0, action_max=1.0)
    err = float(jnp.max(jnp.abs(smoothed - golden["smoothed_actions_ref"])))
    assert err < 1e-5, f"td3_target_smoothing max abs error {err:.2e}"

    # 3. TD3 Clipped Critic Target
    targets = td3_clipped_critic_target(golden["rewards"], golden["dones"], golden["q1_target"], golden["q2_target"], gamma=0.99)
    err = float(jnp.max(jnp.abs(targets - golden["td3_targets_ref"])))
    assert err < 1e-5, f"td3_clipped_critic_target max abs error {err:.2e}"

    # 4. SAC Squashed Gaussian Policy
    actions, log_probs = sac_squashed_gaussian_log_prob(golden["u"], golden["mean"], golden["log_std"])
    err_act = float(jnp.max(jnp.abs(actions - golden["sac_actions_ref"])))
    assert err_act < 1e-5, f"sac_actions max abs error {err_act:.2e}"
    err_lp = float(jnp.max(jnp.abs(log_probs - golden["sac_log_probs_ref"])))
    assert err_lp < 1e-5, f"sac_log_probs max abs error {err_lp:.2e}"

    # 5. SAC Alpha Temperature Loss
    alpha_loss = sac_temperature_loss(golden["log_alpha"], golden["sac_log_probs_ref"], target_entropy=-3.0)
    err_alpha = float(jnp.max(jnp.abs(alpha_loss - golden["alpha_loss_ref"])))
    assert err_alpha < 1e-5, f"alpha_loss max abs error {err_alpha:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
