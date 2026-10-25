"""Value Based RL Dqn Rainbow and Qr Dqn — JAX reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.safetensors

``python jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.dot(x, weight) + bias


def dueling_q_values(value: jax.Array, advantage: jax.Array) -> jax.Array:
    """Dueling Network Q-value aggregation:
    Q(s, a) = V(s) + (A(s, a) - (1 / |A|) * sum_{a'} A(s, a')).
    """
    mean_adv = jnp.mean(advantage, axis=-1, keepdims=True)
    return value + (advantage - mean_adv)


def double_dqn_targets(
    rewards: jax.Array,
    dones: jax.Array,
    q_next_online: jax.Array,
    q_next_target: jax.Array,
    gamma: float = 0.99,
) -> jax.Array:
    """Double DQN Bellman target calculation:
    a* = argmax_{a'} Q_online(s', a')
    y = r + gamma * (1 - done) * Q_target(s', a*)
    """
    best_actions = jnp.argmax(q_next_online, axis=-1)  # [B]
    batch_indices = jnp.arange(rewards.shape[0])
    q_target_selected = q_next_target[batch_indices, best_actions]  # [B]
    return rewards + gamma * (1.0 - dones) * q_target_selected


def per_loss(
    q_eval: jax.Array,
    targets: jax.Array,
    is_weights: jax.Array,
) -> jax.Array:
    """Prioritized Experience Replay (PER) weighted Huber / MSE loss:
    loss = (1 / B) * sum_i w_i * (q_eval_i - y_i)^2
    """
    td_errors = q_eval - targets
    weighted_sq_errors = is_weights * (td_errors ** 2)
    return jnp.mean(weighted_sq_errors)


def quantile_huber_loss(
    current_quantiles: jax.Array,
    target_quantiles: jax.Array,
    tau: jax.Array,
    kappa: float = 1.0,
) -> jax.Array:
    """Quantile Regression DQN (QR-DQN) pairwise quantile Huber loss:
    current_quantiles: [B, N]
    target_quantiles: [B, N]
    tau: [N]
    """
    # diff: [B, N, N] where diff[b, i, j] = target[b, j] - current[b, i]
    diff = jnp.expand_dims(target_quantiles, axis=1) - jnp.expand_dims(current_quantiles, axis=2)
    abs_diff = jnp.abs(diff)

    huber = jnp.where(
        abs_diff <= kappa,
        0.5 * (diff ** 2),
        kappa * (abs_diff - 0.5 * kappa),
    )

    tau_expanded = jnp.reshape(tau, (1, -1, 1))
    indicator = (diff < 0.0).astype(jnp.float32)
    weight = jnp.abs(tau_expanded - indicator)

    elementwise_loss = weight * huber / kappa
    return jnp.mean(elementwise_loss)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.asarray(v) for k, v in raw.items()}

    # 1. Linear anchor
    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear anchor max abs error {err:.2e}"

    # 2. Dueling Q
    q_duel = dueling_q_values(golden["value"], golden["advantage"])
    err = float(jnp.max(jnp.abs(q_duel - golden["dueling_q_ref"])))
    assert err < 1e-5, f"dueling_q max abs error {err:.2e}"

    # 3. Double DQN
    ddqn_targets = double_dqn_targets(
        golden["rewards"],
        golden["dones"],
        golden["q_next_online"],
        golden["q_next_target"],
        gamma=0.99,
    )
    err = float(jnp.max(jnp.abs(ddqn_targets - golden["ddqn_targets_ref"])))
    assert err < 1e-5, f"ddqn_targets max abs error {err:.2e}"

    # 4. PER Loss
    per_l = per_loss(golden["q_eval"], golden["targets"], golden["is_weights"])
    err = float(jnp.max(jnp.abs(per_l - golden["per_loss_ref"])))
    assert err < 1e-5, f"per_loss max abs error {err:.2e}"

    # 5. QR-DQN Loss
    qr_l = quantile_huber_loss(
        golden["current_quantiles"],
        golden["target_quantiles"],
        golden["tau"],
        kappa=1.0,
    )
    err = float(jnp.max(jnp.abs(qr_l - golden["qr_loss_ref"])))
    assert err < 1e-5, f"qr_loss max abs error {err:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
