"""Functional JAX implementation for Category 01 Topic 02:
02_tree_vs_kernel_vs_neural_inductive_biases
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


def soft_decision_tree_forward(
    x: Any,
    routing_weights: Any,
    thresholds: Any,
    temperature: Any,
    leaf_weights: Any,
) -> Any:
    """Evaluates depth-2 differentiable soft decision tree in JAX / functional array form.

    Args:
        x: Input tensor [batch, in_dim].
        routing_weights: Split routing matrix [3, in_dim].
        thresholds: Node split thresholds [3].
        temperature: Sigmoid inverse temperature scalar beta.
        leaf_weights: Leaf prediction vectors [4, out_dim].

    Returns:
        Predicted output [batch, out_dim].
    """
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_route = jnp.asarray(routing_weights, dtype=jnp.float32)
    thresh = jnp.asarray(thresholds, dtype=jnp.float32)
    temp = jnp.asarray(temperature, dtype=jnp.float32)
    w_leaf = jnp.asarray(leaf_weights, dtype=jnp.float32)

    # Routing logits: beta * (x @ W_route.T - thresholds)
    logits = temp * (jnp.dot(x_arr, w_route.T) - thresh)
    p_right = 1.0 / (1.0 + jnp.exp(-logits))  # [batch, 3]
    p_left = 1.0 - p_right

    mu0 = p_left[:, 0:1] * p_left[:, 1:2]
    mu1 = p_left[:, 0:1] * p_right[:, 1:2]
    mu2 = p_right[:, 0:1] * p_left[:, 2:3]
    mu3 = p_right[:, 0:1] * p_right[:, 2:3]
    leaf_probs = jnp.concatenate([mu0, mu1, mu2, mu3], axis=-1)  # [batch, 4]

    return jnp.dot(leaf_probs, w_leaf)


def rbf_kernel_ridge_forward(
    x: Any,
    support_vectors: Any,
    dual_coeffs: Any,
    gamma: Any,
    bias: Any,
) -> Any:
    """Evaluates RBF kernel ridge predictor in JAX / functional array form.

    Args:
        x: Input tensor [batch, in_dim].
        support_vectors: Support vectors [support, in_dim].
        dual_coeffs: Dual expansion coefficients [support, out_dim].
        gamma: Inverse squared bandwidth scalar.
        bias: Additive intercept vector [out_dim].

    Returns:
        Predicted output [batch, out_dim].
    """
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    support = jnp.asarray(support_vectors, dtype=jnp.float32)
    coeffs = jnp.asarray(dual_coeffs, dtype=jnp.float32)
    g = jnp.asarray(gamma, dtype=jnp.float32)
    b = jnp.asarray(bias, dtype=jnp.float32)

    x_sq = jnp.sum(x_arr**2, axis=-1, keepdims=True)  # [batch, 1]
    s_sq = jnp.sum(support**2, axis=-1, keepdims=True).T  # [1, support]
    dists_sq = jnp.maximum(x_sq + s_sq - 2.0 * jnp.dot(x_arr, support.T), 0.0)
    sim = jnp.exp(-g * dists_sq)
    return jnp.dot(sim, coeffs) + b


def mlp_forward(
    x: Any,
    w1: Any,
    b1: Any,
    w2: Any,
    b2: Any,
) -> Any:
    """Evaluates 2-layer MLP with ReLU activation in JAX / functional array form.

    Args:
        x: Input tensor [batch, in_dim].
        w1: Input projection weight [in_dim, hidden_dim].
        b1: Hidden bias [hidden_dim].
        w2: Output projection weight [hidden_dim, out_dim].
        b2: Output bias [out_dim].

    Returns:
        Predicted output [batch, out_dim].
    """
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w1_arr = jnp.asarray(w1, dtype=jnp.float32)
    b1_arr = jnp.asarray(b1, dtype=jnp.float32)
    w2_arr = jnp.asarray(w2, dtype=jnp.float32)
    b2_arr = jnp.asarray(b2, dtype=jnp.float32)

    h = jnp.maximum(jnp.dot(x_arr, w1_arr) + b1_arr, 0.0)
    return jnp.dot(h, w2_arr) + b2_arr
