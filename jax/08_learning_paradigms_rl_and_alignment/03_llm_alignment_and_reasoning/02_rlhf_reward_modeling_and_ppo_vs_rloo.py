"""RLHF Reward Modeling and PPO vs Rloo — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.
Run ``python jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def bradley_terry_reward_loss(
    rewards_chosen: jnp.ndarray,
    rewards_rejected: jnp.ndarray,
) -> jnp.ndarray:
    diff = rewards_chosen - rewards_rejected
    return jnp.mean(jax.nn.softplus(-diff))


def rlhf_token_level_rewards(
    log_probs_policy: jnp.ndarray,
    log_probs_ref: jnp.ndarray,
    terminal_scores: jnp.ndarray,
    kl_coeff: float = 0.05,
) -> jnp.ndarray:
    token_kl = log_probs_policy - log_probs_ref
    rewards = -kl_coeff * token_kl
    return rewards.at[:, -1].add(terminal_scores)


def rloo_leave_one_out_advantages(
    scores: jnp.ndarray,
) -> jnp.ndarray:
    b, k = scores.shape
    total_scores = jnp.sum(scores, axis=-1, keepdims=True)
    baselines = (total_scores - scores) / float(k - 1)
    return scores - baselines


def rloo_policy_gradient_loss(
    advantages: jnp.ndarray,
    log_probs_sum: jnp.ndarray,
) -> jnp.ndarray:
    return -jnp.mean(advantages * log_probs_sum)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. Bradley-Terry Loss
    bt_loss = bradley_terry_reward_loss(g["rewards_chosen"], g["rewards_rejected"])
    err_bt = float(jnp.abs(bt_loss - g["bt_loss_ref"]))
    assert err_bt < 1e-5, f"bradley-terry loss error {err_bt:.2e}"

    # 3. RLHF Token Rewards
    rew = rlhf_token_level_rewards(g["log_probs_policy"], g["log_probs_ref"], g["terminal_scores"], kl_coeff=0.05)
    err_rew = float(jnp.max(jnp.abs(rew - g["rlhf_rewards_ref"])))
    assert err_rew < 1e-5, f"rlhf token rewards error {err_rew:.2e}"

    # 4. RLOO Advantages
    adv = rloo_leave_one_out_advantages(g["scores"])
    err_adv = float(jnp.max(jnp.abs(adv - g["rloo_adv_ref"])))
    assert err_adv < 1e-5, f"rloo advantages error {err_adv:.2e}"

    # 5. RLOO Loss
    loss = rloo_policy_gradient_loss(g["rloo_adv_ref"], g["log_probs_sum"])
    err_loss = float(jnp.abs(loss - g["rloo_loss_ref"]))
    assert err_loss < 1e-5, f"rloo loss error {err_loss:.2e}"

    print("08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
