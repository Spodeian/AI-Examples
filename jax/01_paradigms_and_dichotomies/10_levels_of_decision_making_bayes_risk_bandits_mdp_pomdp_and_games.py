"""Functional JAX/NumPy Implementation of Levels of Decision Making (Topic 10).

Functional implementations of:
- L0: Bayes Risk Decision Rule & Chow's Reject Option
- L1: Contextual Bandits Off-Policy Evaluation (IPS & Doubly Robust)
- L2: Closed-Loop MDP Bellman Optimality Backup
- L3: POMDP Bayesian Belief-State Filtering
- L5: Differentiable Mechanism Design / Auction Payment
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from safetensors.numpy import load_file

try:
    import jax
    import jax.numpy as jnp
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
    _ = jax
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


def softmax(x: Any, axis: int = -1) -> Any:
    """Numerically stable softmax."""
    x_max = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)


def bayes_decision_rule(
    probs: Any,
    cost_matrix: Any,
    c_reject: float = 0.5,
) -> Tuple[Any, Any, Any]:
    """Computes expected risk per action and flags Chow rejection."""
    p = jnp.asarray(probs, dtype=jnp.float32)
    c = jnp.asarray(cost_matrix, dtype=jnp.float32)
    expected_risk = jnp.dot(p, c.T)
    best_actions = jnp.argmin(expected_risk, axis=-1).astype(jnp.float32)
    min_risk = jnp.min(expected_risk, axis=-1)
    reject_mask = (min_risk > c_reject).astype(jnp.float32)
    return expected_risk, best_actions, reject_mask


def bandits_ope(
    target_probs: Any,
    logged_propensities: Any,
    rewards: Any,
    baseline_q: Any,
) -> Tuple[Any, Any]:
    """Computes IPS weights and Doubly Robust counterfactual reward estimates."""
    tp = jnp.asarray(target_probs, dtype=jnp.float32)
    lp = jnp.asarray(logged_propensities, dtype=jnp.float32)
    rew = jnp.asarray(rewards, dtype=jnp.float32)
    q = jnp.asarray(baseline_q, dtype=jnp.float32)

    ips_weights = tp / lp
    doubly_robust = q + ips_weights * (rew - q)
    return ips_weights, doubly_robust


def mdp_bellman_operator(
    v_current: Any,
    transitions: Any,
    rewards: Any,
    gamma: float = 0.9,
) -> Tuple[Any, Any]:
    """Applies a single Bellman optimality backup step on the state value function."""
    v = jnp.asarray(v_current, dtype=jnp.float32)
    p = jnp.asarray(transitions, dtype=jnp.float32)
    r = jnp.asarray(rewards, dtype=jnp.float32)

    future_v = jnp.einsum("sak,k->sa", p, v)
    q_values = r + gamma * future_v
    v_next = jnp.max(q_values, axis=-1)
    return q_values, v_next


def pomdp_belief_filter(
    prior_belief: Any,
    transition: Any,
    observation_likelihood: Any,
) -> Any:
    """Applies Bayesian belief prediction and update step given an observation."""
    b = jnp.asarray(prior_belief, dtype=jnp.float32)
    t = jnp.asarray(transition, dtype=jnp.float32)
    obs = jnp.asarray(observation_likelihood, dtype=jnp.float32)

    b_pred = jnp.dot(b, t)
    b_unnorm = obs * b_pred
    return b_unnorm / jnp.sum(b_unnorm)


def differentiable_auction(
    bids: Any,
    tau: float = 0.5,
) -> Tuple[Any, Any]:
    """Computes soft allocations and second-price auction payments."""
    b = jnp.asarray(bids, dtype=jnp.float32)
    allocations = softmax(b / tau, axis=-1)
    min_bid = jnp.min(b, axis=-1, keepdims=True)
    payments = allocations * min_bid
    return allocations, payments


def linear_anchor_forward(
    x: Any,
    weight: Any,
    bias: Any,
) -> Any:
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_arr = jnp.asarray(weight, dtype=jnp.float32)
    b_arr = jnp.asarray(bias, dtype=jnp.float32)
    return jnp.dot(x_arr, w_arr.T) + b_arr


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    raw = load_file(checkpoint_path)
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in raw.items()}


if __name__ == "__main__":
    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.safetensors"
    )
    tensors = load_checkpoint(str(ckpt))

    # Linear anchor
    out_anchor = linear_anchor_forward(tensors["x_anchor"], tensors["weight"], tensors["bias"])
    assert jnp.allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5), "Anchor mismatch"

    # L0 Bayes
    l0_risk, l0_acts, l0_rej = bayes_decision_rule(tensors["l0_probs"], tensors["l0_cost_matrix"])
    assert jnp.allclose(l0_risk, tensors["l0_expected_risk"], atol=1e-5), "L0 risk mismatch"
    assert jnp.allclose(l0_acts, tensors["l0_best_actions"], atol=1e-5), "L0 acts mismatch"
    assert jnp.allclose(l0_rej, tensors["l0_reject_mask"], atol=1e-5), "L0 reject mismatch"

    # L1 Bandits OPE
    l1_w, l1_dr = bandits_ope(
        tensors["l1_target_policy_probs"],
        tensors["l1_logged_propensities"],
        tensors["l1_rewards"],
        tensors["l1_baseline_q"],
    )
    assert jnp.allclose(l1_w, tensors["l1_ips_weights"], atol=1e-5), "L1 weights mismatch"
    assert jnp.allclose(l1_dr, tensors["l1_doubly_robust_estimates"], atol=1e-5), "L1 DR mismatch"

    # L2 MDP Bellman
    l2_q, l2_v = mdp_bellman_operator(
        tensors["l2_v_init"], tensors["l2_transitions"], tensors["l2_rewards"]
    )
    assert jnp.allclose(l2_q, tensors["l2_q_values"], atol=1e-5), "L2 Q mismatch"
    assert jnp.allclose(l2_v, tensors["l2_v_backup"], atol=1e-5), "L2 V mismatch"

    # L3 POMDP Belief
    l3_post = pomdp_belief_filter(
        tensors["l3_prior_belief"], tensors["l3_transition"], tensors["l3_observation_likelihood"]
    )
    assert jnp.allclose(l3_post, tensors["l3_posterior_belief"], atol=1e-5), "L3 post mismatch"

    # L5 Auction
    l5_alloc, l5_pay = differentiable_auction(tensors["l5_bids"])
    assert jnp.allclose(l5_alloc, tensors["l5_allocations"], atol=1e-5), "L5 alloc mismatch"
    assert jnp.allclose(l5_pay, tensors["l5_payments"], atol=1e-5), "L5 pay mismatch"

    print("JAX Decision Hierarchy operations successfully verified with zero errors!")
