"""GRPO Dapo and Verifiable Reward Reasoning — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.
Run ``python jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def grpo_group_advantages(
    rewards: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    mean = jnp.mean(rewards, axis=-1, keepdims=True)
    std = jnp.std(rewards, axis=-1, keepdims=True)
    return (rewards - mean) / (std + eps)


def k3_kl_divergence(
    log_probs_policy: jnp.ndarray,
    log_probs_ref: jnp.ndarray,
) -> jnp.ndarray:
    delta = log_probs_policy - log_probs_ref
    return jnp.exp(-delta) - 1.0 + delta


def grpo_clipped_loss(
    log_probs_policy: jnp.ndarray,
    log_probs_old: jnp.ndarray,
    advantages: jnp.ndarray,
    kl_div: jnp.ndarray,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> jnp.ndarray:
    ratio = jnp.exp(log_probs_policy - log_probs_old)
    adv = advantages[..., None]
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    ppo_obj = jnp.minimum(surr1, surr2)
    token_obj = ppo_obj - beta * kl_div
    return -jnp.mean(token_obj)


def composite_verifiable_reward(
    accuracy_flags: jnp.ndarray,
    format_flags: jnp.ndarray,
    acc_weight: float = 1.0,
    format_weight: float = 0.5,
) -> jnp.ndarray:
    return acc_weight * accuracy_flags + format_weight * format_flags


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. Composite Verifiable Rewards
    rewards = composite_verifiable_reward(g["accuracy_flags"], g["format_flags"], acc_weight=1.0, format_weight=0.5)
    err_rew = float(jnp.max(jnp.abs(rewards - g["rewards_ref"])))
    assert err_rew < 1e-5, f"rewards error {err_rew:.2e}"

    # 3. GRPO Group Advantages
    adv = grpo_group_advantages(g["rewards_ref"])
    err_adv = float(jnp.max(jnp.abs(adv - g["advantages_ref"])))
    assert err_adv < 1e-5, f"advantages error {err_adv:.2e}"

    # 4. K3 KL Divergence
    kl = k3_kl_divergence(g["log_probs_policy"], g["log_probs_ref"])
    err_kl = float(jnp.max(jnp.abs(kl - g["kl_div_ref"])))
    assert err_kl < 1e-5, f"kl error {err_kl:.2e}"

    # 5. GRPO Clipped Loss
    loss = grpo_clipped_loss(g["log_probs_policy"], g["log_probs_old"], g["advantages_ref"], g["kl_div_ref"], clip_eps=0.2, beta=0.04)
    err_loss = float(jnp.abs(loss - g["grpo_loss_ref"]))
    assert err_loss < 1e-5, f"grpo loss error {err_loss:.2e}"

    print("08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
