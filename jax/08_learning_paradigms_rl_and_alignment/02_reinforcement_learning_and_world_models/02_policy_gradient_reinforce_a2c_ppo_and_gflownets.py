"""Policy Gradient Reinforce A2c PPO and GFlowNets — JAX reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.safetensors

``python jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.dot(x, weight) + bias


def ppo_clip_loss(
    log_probs: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    clip_eps: float = 0.2,
) -> jax.Array:
    """PPO-Clip policy surrogate loss:
    r_t = exp(log_prob - old_log_prob)
    L_clip = - mean(min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t))
    """
    ratio = jnp.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    clipped_surrogate = jnp.minimum(surr1, surr2)
    return -jnp.mean(clipped_surrogate)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> jax.Array:
    """Generalized Advantage Estimation (GAE) via scan/loop."""
    t_len = rewards.shape[0]

    def scan_fn(carry: jax.Array, idx: int) -> tuple[jax.Array, jax.Array]:
        last_gae = carry
        t = t_len - 1 - idx
        v_next = jnp.where(t == t_len - 1, next_value, values[t + 1])
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * not_done * v_next - values[t]
        adv = delta + gamma * lambda_ * not_done * last_gae
        return adv, adv

    _, rev_advantages = jax.lax.scan(scan_fn, jnp.array(0.0, dtype=jnp.float32), jnp.arange(t_len))
    return jnp.flip(rev_advantages, axis=0)


def gflownet_trajectory_balance_loss(
    log_z: jax.Array,
    log_pf_trajectory: jax.Array,
    log_pb_trajectory: jax.Array,
    reward: jax.Array,
) -> jax.Array:
    """GFlowNet Trajectory Balance (TB) objective."""
    sum_log_pf = jnp.sum(log_pf_trajectory, axis=-1)
    sum_log_pb = jnp.sum(log_pb_trajectory, axis=-1)
    log_r = jnp.log(jnp.clip(reward, a_min=1e-8))

    numerator = log_z + sum_log_pf
    denominator = log_r + sum_log_pb
    tb_error = numerator - denominator
    return jnp.mean(tb_error ** 2)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.asarray(v) for k, v in raw.items()}

    # 1. Linear anchor
    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear anchor max abs error {err:.2e}"

    # 2. PPO Clip Loss
    ppo_loss = ppo_clip_loss(golden["log_probs"], golden["old_log_probs"], golden["advantages"], clip_eps=0.2)
    err = float(jnp.max(jnp.abs(ppo_loss - golden["ppo_loss_ref"])))
    assert err < 1e-5, f"ppo_loss max abs error {err:.2e}"

    # 3. GAE
    gae_adv = compute_gae(
        golden["rewards"],
        golden["values"],
        golden["dones"],
        golden["next_value"],
        gamma=0.99,
        lambda_=0.95,
    )
    err = float(jnp.max(jnp.abs(gae_adv - golden["gae_advantages_ref"])))
    assert err < 1e-5, f"gae_advantages max abs error {err:.2e}"

    # 4. GFlowNet TB Loss
    tb_loss = gflownet_trajectory_balance_loss(
        golden["log_z"],
        golden["log_pf"],
        golden["log_pb"],
        golden["reward"],
    )
    err = float(jnp.max(jnp.abs(tb_loss - golden["tb_loss_ref"])))
    assert err < 1e-5, f"tb_loss max abs error {err:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
