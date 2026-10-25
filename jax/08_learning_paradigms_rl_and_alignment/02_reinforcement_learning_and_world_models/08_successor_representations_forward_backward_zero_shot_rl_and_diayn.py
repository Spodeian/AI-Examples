"""Successor Representations Forward Backward Zero Shot RL and Diayn — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.
Run ``python jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def successor_features_td_error(
    psi_current: jnp.ndarray,
    features: jnp.ndarray,
    psi_next: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
) -> jnp.ndarray:
    non_terminal = (1.0 - dones)[:, None]
    target = features + gamma * non_terminal * psi_next
    return target - psi_current


def gpi_transfer_value(
    psi_policies: jnp.ndarray,
    reward_weights: jnp.ndarray,
) -> jnp.ndarray:
    q_candidates = jnp.matmul(psi_policies, reward_weights)
    return jnp.max(q_candidates, axis=-1)


def fb_zero_shot_value(
    forward_map: jnp.ndarray,
    task_embedding: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.sum(forward_map * task_embedding, axis=-1)


def diayn_skill_reward_and_loss(
    logits: jnp.ndarray,
    skill_labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    num_skills = logits.shape[-1]
    log_prior = -math.log(num_skills)
    b = logits.shape[0]
    batch_idx = jnp.arange(b)
    skill_log_probs = log_probs[batch_idx, skill_labels]
    rewards = skill_log_probs - log_prior
    loss = -jnp.mean(skill_log_probs)
    return rewards, loss


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. Successor Features TD Error
    sf_td = successor_features_td_error(g["psi_current"], g["features"], g["psi_next"], g["dones"], gamma=0.99)
    err_sf = float(jnp.max(jnp.abs(sf_td - g["sf_td_error_ref"])))
    assert err_sf < 1e-5, f"successor features td error {err_sf:.2e}"

    # 3. GPI Transfer Value
    gpi_val = gpi_transfer_value(g["psi_policies"], g["reward_weights"])
    err_gpi = float(jnp.max(jnp.abs(gpi_val - g["gpi_values_ref"])))
    assert err_gpi < 1e-5, f"gpi transfer error {err_gpi:.2e}"

    # 4. FB Zero-Shot Value
    fb_val = fb_zero_shot_value(g["forward_map"], g["task_embedding"])
    err_fb = float(jnp.max(jnp.abs(fb_val - g["fb_values_ref"])))
    assert err_fb < 1e-5, f"fb zero shot error {err_fb:.2e}"

    # 5. DIAYN Skill Reward and Loss
    diayn_r, diayn_l = diayn_skill_reward_and_loss(g["logits"], g["skill_labels"])
    err_diayn_r = float(jnp.max(jnp.abs(diayn_r - g["diayn_rewards_ref"])))
    err_diayn_l = float(jnp.abs(diayn_l - g["diayn_loss_ref"]))
    assert err_diayn_r < 1e-5, f"diayn reward error {err_diayn_r:.2e}"
    assert err_diayn_l < 1e-5, f"diayn loss error {err_diayn_l:.2e}"

    print("08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
