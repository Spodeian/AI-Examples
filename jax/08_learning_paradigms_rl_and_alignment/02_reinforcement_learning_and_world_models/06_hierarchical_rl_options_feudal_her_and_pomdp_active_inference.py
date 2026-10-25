"""Hierarchical RL Options Feudal Her and POMDP Active Inference — JAX reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.safetensors

``python jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.dot(x, weight) + bias


def her_goal_relabeling(
    states: jax.Array,
    goals: jax.Array,
    threshold: float = 0.1,
) -> jax.Array:
    """Hindsight Experience Replay (HER) sparse binary goal achievement reward."""
    dists = jnp.linalg.norm(states - goals, axis=-1)
    return (dists <= threshold).astype(jnp.float32)


def feudal_directional_reward(
    delta_s: jax.Array,
    subgoal_g: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """FeUdal Networks cosine directional reward."""
    dot_product = jnp.sum(delta_s * subgoal_g, axis=-1)
    norm_delta = jnp.linalg.norm(delta_s, axis=-1)
    norm_g = jnp.linalg.norm(subgoal_g, axis=-1)
    denom = jnp.clip(norm_delta * norm_g, a_min=eps)
    return dot_product / denom


def option_critic_termination_loss(
    beta_omega: jax.Array,
    q_omega: jax.Array,
    v_s: jax.Array,
    deliberation_cost: float = 0.01,
) -> jax.Array:
    """Option-Critic termination gradient loss with deliberation cost."""
    advantage_cost = q_omega - v_s + deliberation_cost
    return jnp.mean(beta_omega * advantage_cost)


def active_inference_free_energy(
    q_mean: jax.Array,
    q_log_std: jax.Array,
    p_mean: jax.Array,
    p_log_std: jax.Array,
    obs: jax.Array,
    recon: jax.Array,
    obs_noise_var: float = 1.0,
) -> jax.Array:
    """Variational Free Energy F in Active Inference / POMDP."""
    var_q = jnp.exp(2.0 * q_log_std)
    var_p = jnp.exp(2.0 * p_log_std)
    diff = q_mean - p_mean
    kl = 0.5 * jnp.sum((var_q + diff ** 2) / var_p - 1.0 + 2.0 * (p_log_std - q_log_std), axis=-1)

    diff_obs = obs - recon
    nll = 0.5 * jnp.sum((diff_obs ** 2) / obs_noise_var + math.log(2.0 * math.pi * obs_noise_var), axis=-1)
    return kl + nll


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.asarray(v) for k, v in raw.items()}

    # 1. Linear anchor
    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear anchor max abs error {err:.2e}"

    # 2. HER Relabeling
    her_r = her_goal_relabeling(golden["states"], golden["goals"], threshold=0.1)
    err = float(jnp.max(jnp.abs(her_r - golden["her_rewards_ref"])))
    assert err < 1e-5, f"her_rewards max abs error {err:.2e}"

    # 3. FeUdal Direction
    feudal_r = feudal_directional_reward(golden["delta_s"], golden["subgoal_g"])
    err = float(jnp.max(jnp.abs(feudal_r - golden["feudal_rewards_ref"])))
    assert err < 1e-5, f"feudal_rewards max abs error {err:.2e}"

    # 4. Option-Critic Termination Loss
    opt_l = option_critic_termination_loss(golden["beta_omega"], golden["q_omega"], golden["v_s"], deliberation_cost=0.01)
    err = float(jnp.max(jnp.abs(opt_l - golden["option_loss_ref"])))
    assert err < 1e-5, f"option_loss max abs error {err:.2e}"

    # 5. Active Inference Free Energy
    free_energy = active_inference_free_energy(
        golden["q_mean"],
        golden["q_log_std"],
        golden["p_mean"],
        golden["p_log_std"],
        golden["obs"],
        golden["recon"],
        obs_noise_var=1.0,
    )
    err = float(jnp.max(jnp.abs(free_energy - golden["free_energy_ref"])))
    assert err < 1e-5, f"free_energy max abs error {err:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
