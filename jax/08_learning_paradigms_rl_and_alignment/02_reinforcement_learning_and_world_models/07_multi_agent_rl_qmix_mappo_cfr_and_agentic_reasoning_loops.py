"""Multi Agent RL QMIX Mappo Cfr and Agentic Reasoning Loops — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.
Run ``python jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def qmix_monotonic_mixing(
    agent_qs: jnp.ndarray,
    w1_raw: jnp.ndarray,
    b1: jnp.ndarray,
    w2_raw: jnp.ndarray,
    b2: jnp.ndarray,
) -> jnp.ndarray:
    w1 = jnp.abs(w1_raw)
    w2 = jnp.abs(w2_raw)
    hidden = jax.nn.elu(jnp.matmul(agent_qs, w1) + b1)
    q_tot = jnp.matmul(hidden, w2) + b2
    return jnp.squeeze(jnp.squeeze(q_tot, axis=-1), axis=-1)


def mappo_centralized_gae(
    values: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    next_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> jnp.ndarray:
    b, t = values.shape
    advs = []
    gae = jnp.zeros(b, dtype=values.dtype)

    for step in reversed(range(t)):
        if step == t - 1:
            next_v = next_value
        else:
            next_v = values[:, step + 1]

        non_terminal = 1.0 - dones[:, step]
        delta = rewards[:, step] + gamma * next_v * non_terminal - values[:, step]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advs.append(gae)

    return jnp.stack(advs[::-1], axis=1)


def cfr_regret_matching(
    cumulative_regrets: jnp.ndarray,
    eps: float = 1e-9,
) -> jnp.ndarray:
    positive_regrets = jnp.maximum(cumulative_regrets, 0.0)
    sum_positive = jnp.sum(positive_regrets, axis=-1, keepdims=True)
    num_actions = cumulative_regrets.shape[-1]
    uniform_prob = jnp.full_like(cumulative_regrets, 1.0 / num_actions)

    strategy = jnp.where(sum_positive > eps, positive_regrets / (sum_positive + eps), uniform_prob)
    return strategy


def agentic_consensus_metric(
    embeddings: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    b, n, d = embeddings.shape
    norms = jnp.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    normed = embeddings / jnp.maximum(norms, eps)
    sim_matrix = jnp.matmul(normed, jnp.swapaxes(normed, 1, 2))
    total_sim = jnp.sum(sim_matrix, axis=(-2, -1)) - n
    return total_sim / (n * (n - 1))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. QMIX Monotonic Mixing
    q_tot = qmix_monotonic_mixing(g["agent_qs"], g["w1_raw"], g["b1"], g["w2_raw"], g["b2"])
    err_qmix = float(jnp.max(jnp.abs(q_tot - g["qmix_q_tot_ref"])))
    assert err_qmix < 1e-5, f"qmix error {err_qmix:.2e}"

    # 3. MAPPO Centralized GAE
    adv = mappo_centralized_gae(g["values"], g["rewards"], g["dones"], g["next_value"])
    err_mappo = float(jnp.max(jnp.abs(adv - g["mappo_advantages_ref"])))
    assert err_mappo < 1e-5, f"mappo error {err_mappo:.2e}"

    # 4. CFR Regret Matching
    strat = cfr_regret_matching(g["cumulative_regrets"])
    err_cfr = float(jnp.max(jnp.abs(strat - g["cfr_strategy_ref"])))
    assert err_cfr < 1e-5, f"cfr error {err_cfr:.2e}"

    # 5. Agentic Consensus Metric
    consensus = agentic_consensus_metric(g["embeddings"])
    err_consensus = float(jnp.max(jnp.abs(consensus - g["consensus_ref"])))
    assert err_consensus < 1e-5, f"consensus error {err_consensus:.2e}"

    print("08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
