"""Latent World Models Rssm Dreamer Tdmpc and Rnd Curiosity — JAX reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.safetensors

``python jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.dot(x, weight) + bias


def gaussian_kl_divergence(
    mu_q: jax.Array,
    log_std_q: jax.Array,
    mu_p: jax.Array,
    log_std_p: jax.Array,
) -> jax.Array:
    """Analytical KL divergence between diagonal Gaussians."""
    var_q = jnp.exp(2.0 * log_std_q)
    var_p = jnp.exp(2.0 * log_std_p)
    diff = mu_q - mu_p
    kl_element = (var_q + diff ** 2) / var_p - 1.0 + 2.0 * (log_std_p - log_std_q)
    return 0.5 * jnp.sum(kl_element, axis=-1)


def rssm_kl_balancing(
    mu_q: jax.Array,
    log_std_q: jax.Array,
    mu_p: jax.Array,
    log_std_p: jax.Array,
    beta_prior: float = 0.8,
    beta_post: float = 0.2,
) -> jax.Array:
    """Dreamer / RSSM KL balancing objective."""
    kl_prior = gaussian_kl_divergence(jax.lax.stop_gradient(mu_q), jax.lax.stop_gradient(log_std_q), mu_p, log_std_p)
    kl_post = gaussian_kl_divergence(mu_q, log_std_q, jax.lax.stop_gradient(mu_p), jax.lax.stop_gradient(log_std_p))
    return beta_prior * jnp.mean(kl_prior) + beta_post * jnp.mean(kl_post)


def rnd_intrinsic_reward(
    target_features: jax.Array,
    predictor_features: jax.Array,
) -> jax.Array:
    """Random Network Distillation (RND) intrinsic curiosity reward."""
    return jnp.mean((predictor_features - target_features) ** 2, axis=-1)


def tdmpc_latent_rollout_return(
    latent_rewards: jax.Array,
    terminal_q: jax.Array,
    gamma: float = 0.99,
) -> jax.Array:
    """TD-MPC2 latent imagination rollout return."""
    horizon = latent_rewards.shape[1]
    discounts = gamma ** jnp.arange(horizon, dtype=jnp.float32)
    discounted_rewards = jnp.sum(latent_rewards * jnp.expand_dims(discounts, axis=0), axis=-1)
    terminal_discount = gamma ** horizon
    return discounted_rewards + terminal_discount * terminal_q


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.asarray(v) for k, v in raw.items()}

    # 1. Linear anchor
    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear anchor max abs error {err:.2e}"

    # 2. Gaussian KL
    kl_div = gaussian_kl_divergence(golden["mu_q"], golden["log_std_q"], golden["mu_p"], golden["log_std_p"])
    err = float(jnp.max(jnp.abs(kl_div - golden["kl_div_ref"])))
    assert err < 1e-5, f"kl_div max abs error {err:.2e}"

    # 3. RSSM KL Balancing
    kl_balanced = rssm_kl_balancing(golden["mu_q"], golden["log_std_q"], golden["mu_p"], golden["log_std_p"], beta_prior=0.8, beta_post=0.2)
    err = float(jnp.max(jnp.abs(kl_balanced - golden["kl_balanced_ref"])))
    assert err < 1e-5, f"kl_balanced max abs error {err:.2e}"

    # 4. RND Curiosity
    rnd_rew = rnd_intrinsic_reward(golden["target_features"], golden["predictor_features"])
    err = float(jnp.max(jnp.abs(rnd_rew - golden["rnd_rewards_ref"])))
    assert err < 1e-5, f"rnd_rewards max abs error {err:.2e}"

    # 5. TD-MPC2 Rollout Return
    tdmpc_ret = tdmpc_latent_rollout_return(golden["latent_rewards"], golden["terminal_q"], gamma=0.99)
    err = float(jnp.max(jnp.abs(tdmpc_ret - golden["tdmpc_return_ref"])))
    assert err < 1e-5, f"tdmpc_return max abs error {err:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
