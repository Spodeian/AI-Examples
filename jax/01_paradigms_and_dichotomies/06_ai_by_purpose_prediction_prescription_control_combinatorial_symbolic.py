"""Functional JAX implementation for Category 01 Topic 06:
06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic
"""

from __future__ import annotations

from typing import Any

try:
    import jax.numpy as jnp

    import jax
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
    _ = jax
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


def cost_predictor_forward(
    x: Any,
    w_cost1: Any,
    b_cost1: Any,
    w_cost2: Any,
    b_cost2: Any,
) -> Any:
    """Computes softplus(ReLU(x @ w1 + b1) @ w2 + b2)."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w1 = jnp.asarray(w_cost1, dtype=jnp.float32)
    b1 = jnp.asarray(b_cost1, dtype=jnp.float32)
    w2 = jnp.asarray(w_cost2, dtype=jnp.float32)
    b2 = jnp.asarray(b_cost2, dtype=jnp.float32)

    h = jnp.maximum(jnp.dot(x_arr, w1) + b1, 0.0)
    out = jnp.dot(h, w2) + b2
    # Numerically stable softplus: log(1 + exp(out))
    return jnp.logaddexp(out, 0.0)


def solve_lp_simplex(cost_vectors: Any) -> Any:
    """Exact simplex vertex: one_hot(argmin_i c_i)."""
    c_arr = jnp.asarray(cost_vectors, dtype=jnp.float32)
    opt_idx = jnp.argmin(c_arr, axis=-1)
    cost_dim = c_arr.shape[-1]
    if HAS_JAX:
        return jax.nn.one_hot(opt_idx, num_classes=cost_dim, dtype=jnp.float32)
    # NumPy fallback
    return (opt_idx[..., None] == jnp.arange(cost_dim)).astype(jnp.float32)


def decision_regret(c_pred: Any, c_star: Any) -> Any:
    """Computes downstream decision regret: c*^T w*(c_pred) - c*^T w*(c*)."""
    cp = jnp.asarray(c_pred, dtype=jnp.float32)
    cs = jnp.asarray(c_star, dtype=jnp.float32)

    w_pred = solve_lp_simplex(cp)
    w_star = solve_lp_simplex(cs)

    cost_actual = jnp.sum(cs * w_pred, axis=-1)
    cost_oracle = jnp.sum(cs * w_star, axis=-1)
    return cost_actual - cost_oracle


def spo_plus_loss(c_pred: Any, c_star: Any) -> Any:
    """Computes convex SPO+ surrogate loss."""
    cp = jnp.asarray(c_pred, dtype=jnp.float32)
    cs = jnp.asarray(c_star, dtype=jnp.float32)

    w_star = solve_lp_simplex(cs)
    cost_oracle = jnp.sum(cs * w_star, axis=-1)

    v_vec = cs - 2.0 * cp
    max_term = jnp.max(v_vec, axis=-1)
    lin_term = 2.0 * jnp.sum(cp * w_star, axis=-1) - cost_oracle
    return max_term + lin_term


def causal_predict_tau(x: Any, w_tau: Any, b_tau: Any) -> Any:
    """Predicts treatment effect tau(x) = x @ w_tau + b_tau."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    wt = jnp.asarray(w_tau, dtype=jnp.float32)
    bt = jnp.asarray(b_tau, dtype=jnp.float32)
    return jnp.dot(x_arr, wt) + bt


def causal_forward(
    x: Any,
    treatment_t: Any,
    w_tau: Any,
    b_tau: Any,
    w_mu0: Any,
    b_mu0: Any,
) -> Any:
    """Computes factual outcome y = mu_0(x) + T * tau(x)."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    t_arr = jnp.asarray(treatment_t, dtype=jnp.float32)
    w_m = jnp.asarray(w_mu0, dtype=jnp.float32)
    b_m = jnp.asarray(b_mu0, dtype=jnp.float32)

    tau = causal_predict_tau(x_arr, w_tau, b_tau)
    mu0 = jnp.dot(x_arr, w_m) + b_m
    return mu0 + t_arr * tau


def symbolic_forward(x: Any, w_eql: Any, b_eql: Any) -> Any:
    """Computes EQL functional primitives: [x0, sin(x1), cos(x2), x0*x3] @ w + b."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w = jnp.asarray(w_eql, dtype=jnp.float32)
    b = jnp.asarray(b_eql, dtype=jnp.float32)

    f0 = x_arr[:, 0:1]
    f1 = jnp.sin(x_arr[:, 1:2])
    f2 = jnp.cos(x_arr[:, 2:3])
    f3 = x_arr[:, 0:1] * x_arr[:, 3:4]
    features = jnp.concatenate([f0, f1, f2, f3], axis=-1)
    return jnp.dot(features, w) + b
