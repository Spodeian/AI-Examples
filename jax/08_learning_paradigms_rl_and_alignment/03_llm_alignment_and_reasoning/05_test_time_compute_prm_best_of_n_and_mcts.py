"""Test Time Compute PRM Best of N and Mcts — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts. Run ``python jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def prm_product_aggregation(step_probs: jnp.ndarray, step_mask: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Process Reward Model product aggregation over valid reasoning steps."""
    log_probs = jnp.log(jnp.clip(step_probs, a_min=eps)) * step_mask
    return jnp.exp(jnp.sum(log_probs, axis=-1))


def prm_min_aggregation(step_probs: jnp.ndarray, step_mask: jnp.ndarray) -> jnp.ndarray:
    """Process Reward Model min aggregation (weakest link principle)."""
    masked_probs = jnp.where(step_mask > 0.5, step_probs, jnp.ones_like(step_probs))
    return jnp.min(masked_probs, axis=-1)


def best_of_n_rerank(
    orm_scores: jnp.ndarray,
    prm_scores: jnp.ndarray,
    alpha: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Combines ORM and PRM scores and identifies top-ranked candidate."""
    combined = alpha * orm_scores + (1.0 - alpha) * prm_scores
    best_idx = jnp.argmax(combined).astype(jnp.float32)[None]
    return combined, best_idx


def puct_node_selection(
    q_values: jnp.ndarray,
    priors: jnp.ndarray,
    visit_counts: jnp.ndarray,
    c_puct: float = 1.414,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """AlphaZero-style Predictor Upper Confidence bounds for Trees (PUCT)."""
    total_visits = jnp.sum(visit_counts)
    exploration = c_puct * priors * jnp.sqrt(total_visits) / (1.0 + visit_counts)
    puct_scores = q_values + exploration
    selected_action = jnp.argmax(puct_scores).astype(jnp.float32)[None]
    return puct_scores, selected_action


def mcts_backpropagation(
    q_values: jnp.ndarray,
    visit_counts: jnp.ndarray,
    action: int,
    leaf_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Incremental running-average backpropagation update along search trajectory."""
    new_visits = visit_counts.at[action].add(1.0)
    updated_q_val = q_values[action] + (leaf_value - q_values[action]) / new_visits[action]
    new_q = q_values.at[action].set(updated_q_val)
    return new_q, new_visits


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # PRM
    prm_prod = prm_product_aggregation(g["step_probs"], g["step_mask"])
    err_prod = float(jnp.max(jnp.abs(prm_prod - g["prm_prod_ref"])))
    assert err_prod < 1e-5, f"PRM product error {err_prod:.2e}"

    prm_min = prm_min_aggregation(g["step_probs"], g["step_mask"])
    err_min = float(jnp.max(jnp.abs(prm_min - g["prm_min_ref"])))
    assert err_min < 1e-5, f"PRM min error {err_min:.2e}"

    # Best-of-N
    comb, b_idx = best_of_n_rerank(g["orm_scores"], g["prm_min_ref"], alpha=0.5)
    err_comb = float(jnp.max(jnp.abs(comb - g["combined_scores_ref"])))
    assert err_comb < 1e-5, f"Best-of-N combined scores error {err_comb:.2e}"
    err_bidx = float(jnp.max(jnp.abs(b_idx - g["best_idx_ref"])))
    assert err_bidx < 1e-5, f"Best-of-N index error {err_bidx:.2e}"

    # PUCT
    puct, sel_a = puct_node_selection(g["q_values"], g["priors"], g["visit_counts"], c_puct=1.414)
    err_puct = float(jnp.max(jnp.abs(puct - g["puct_scores_ref"])))
    assert err_puct < 1e-5, f"PUCT error {err_puct:.2e}"
    err_sel = float(jnp.max(jnp.abs(sel_a - g["selected_action_ref"])))
    assert err_sel < 1e-5, f"Selected action error {err_sel:.2e}"

    # MCTS backprop
    action = int(g["selected_action_ref"][0])
    q_up, v_up = mcts_backpropagation(g["q_values"], g["visit_counts"], action, 0.90)
    err_q = float(jnp.max(jnp.abs(q_up - g["q_updated_ref"])))
    assert err_q < 1e-5, f"MCTS Q-update error {err_q:.2e}"
    err_v = float(jnp.max(jnp.abs(v_up - g["visits_updated_ref"])))
    assert err_v < 1e-5, f"MCTS visits update error {err_v:.2e}"

    print("08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
