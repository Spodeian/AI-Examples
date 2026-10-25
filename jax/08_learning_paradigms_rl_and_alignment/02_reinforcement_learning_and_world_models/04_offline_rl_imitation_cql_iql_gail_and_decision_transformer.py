"""Offline RL Imitation Cql IQL Gail and Decision Transformer — JAX reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.safetensors

``python jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.dot(x, weight) + bias


def cql_penalty(q_values: jax.Array, dataset_actions: jax.Array) -> jax.Array:
    """Conservative Q-Learning (CQL) conservative gap penalty."""
    logsumexp_q = jax.nn.logsumexp(q_values, axis=-1)
    b_idx = jnp.arange(q_values.shape[0])
    data_q = q_values[b_idx, dataset_actions]
    return jnp.mean(logsumexp_q - data_q)


def iql_expectile_loss(diff: jax.Array, tau: float = 0.7) -> jax.Array:
    """Implicit Q-Learning (IQL) asymmetric expectile regression loss."""
    weight = jnp.where(diff > 0, tau, 1.0 - tau)
    return jnp.mean(weight * (diff ** 2))


def iql_actor_loss(
    pred_action: jax.Array,
    target_action: jax.Array,
    adv: jax.Array,
    beta: float = 3.0,
    max_weight: float = 100.0,
) -> jax.Array:
    """Advantage-Weighted Regression (AWR) actor extraction in IQL."""
    weights = jnp.clip(jnp.exp(beta * adv), a_max=max_weight)
    sq_err = jnp.sum((pred_action - target_action) ** 2, axis=-1)
    return jnp.mean(weights * sq_err)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.asarray(v) for k, v in raw.items()}

    # 1. Linear anchor
    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear anchor max abs error {err:.2e}"

    # 2. CQL penalty
    cql_p = cql_penalty(golden["q_values"], golden["dataset_actions"])
    err = float(jnp.max(jnp.abs(cql_p - golden["cql_penalty_ref"])))
    assert err < 1e-5, f"cql_penalty max abs error {err:.2e}"

    # 3. IQL expectile loss
    iql_exp = iql_expectile_loss(golden["diff"], tau=0.7)
    err = float(jnp.max(jnp.abs(iql_exp - golden["iql_expectile_ref"])))
    assert err < 1e-5, f"iql_expectile max abs error {err:.2e}"

    # 4. IQL actor loss
    iql_act = iql_actor_loss(golden["pred_action"], golden["target_action"], golden["adv"], beta=3.0, max_weight=100.0)
    err = float(jnp.max(jnp.abs(iql_act - golden["iql_actor_ref"])))
    assert err < 1e-5, f"iql_actor max abs error {err:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
