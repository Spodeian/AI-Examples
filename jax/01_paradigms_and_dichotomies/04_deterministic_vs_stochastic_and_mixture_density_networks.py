"""Functional JAX implementation for Category 01 Topic 04:
04_deterministic_vs_stochastic_and_mixture_density_networks
"""

from __future__ import annotations

import math
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


def deterministic_forward(
    x: Any,
    w_h: Any,
    b_h: Any,
    w_det: Any,
    b_det: Any,
) -> Any:
    """Computes deterministic conditional mean: y = ReLU(x @ w_h + b_h) @ w_det + b_det."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_h_arr = jnp.asarray(w_h, dtype=jnp.float32)
    b_h_arr = jnp.asarray(b_h, dtype=jnp.float32)
    w_det_arr = jnp.asarray(w_det, dtype=jnp.float32)
    b_det_arr = jnp.asarray(b_det, dtype=jnp.float32)

    hidden = jnp.maximum(jnp.dot(x_arr, w_h_arr) + b_h_arr, 0.0)
    return jnp.dot(hidden, w_det_arr) + b_det_arr


def mdn_forward(
    x: Any,
    w_h: Any,
    b_h: Any,
    w_pi: Any,
    b_pi: Any,
    w_mu: Any,
    b_mu: Any,
    w_sigma: Any,
    b_sigma: Any,
) -> tuple[Any, Any, Any]:
    """Computes Mixture Density Network parameters (pi, mu, sigma)."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_h_arr = jnp.asarray(w_h, dtype=jnp.float32)
    b_h_arr = jnp.asarray(b_h, dtype=jnp.float32)
    w_pi_arr = jnp.asarray(w_pi, dtype=jnp.float32)
    b_pi_arr = jnp.asarray(b_pi, dtype=jnp.float32)
    w_mu_arr = jnp.asarray(w_mu, dtype=jnp.float32)
    b_mu_arr = jnp.asarray(b_mu, dtype=jnp.float32)
    w_sigma_arr = jnp.asarray(w_sigma, dtype=jnp.float32)
    b_sigma_arr = jnp.asarray(b_sigma, dtype=jnp.float32)

    hidden = jnp.maximum(jnp.dot(x_arr, w_h_arr) + b_h_arr, 0.0)

    pi_logits = jnp.dot(hidden, w_pi_arr) + b_pi_arr
    # Stable softmax
    max_logits = jnp.max(pi_logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(pi_logits - max_logits)
    pi = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    mu = jnp.dot(hidden, w_mu_arr) + b_mu_arr
    log_sigma = jnp.dot(hidden, w_sigma_arr) + b_sigma_arr
    sigma = jnp.exp(log_sigma)

    return pi, mu, sigma


def mdn_log_prob(
    x: Any,
    y: Any,
    w_h: Any,
    b_h: Any,
    w_pi: Any,
    b_pi: Any,
    w_mu: Any,
    b_mu: Any,
    w_sigma: Any,
    b_sigma: Any,
) -> Any:
    """Computes log-likelihood log p(y|x) under Gaussian Mixture."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    y_arr = jnp.asarray(y, dtype=jnp.float32)
    w_h_arr = jnp.asarray(w_h, dtype=jnp.float32)
    b_h_arr = jnp.asarray(b_h, dtype=jnp.float32)
    w_pi_arr = jnp.asarray(w_pi, dtype=jnp.float32)
    b_pi_arr = jnp.asarray(b_pi, dtype=jnp.float32)
    w_mu_arr = jnp.asarray(w_mu, dtype=jnp.float32)
    b_mu_arr = jnp.asarray(b_mu, dtype=jnp.float32)
    w_sigma_arr = jnp.asarray(w_sigma, dtype=jnp.float32)
    b_sigma_arr = jnp.asarray(b_sigma, dtype=jnp.float32)

    hidden = jnp.maximum(jnp.dot(x_arr, w_h_arr) + b_h_arr, 0.0)
    pi_logits = jnp.dot(hidden, w_pi_arr) + b_pi_arr

    # Stable log-softmax
    max_logits = jnp.max(pi_logits, axis=-1, keepdims=True)
    log_pi = pi_logits - max_logits - jnp.log(jnp.sum(jnp.exp(pi_logits - max_logits), axis=-1, keepdims=True))

    mu = jnp.dot(hidden, w_mu_arr) + b_mu_arr
    log_sigma = jnp.dot(hidden, w_sigma_arr) + b_sigma_arr
    sigma = jnp.exp(log_sigma)

    diff = y_arr - mu
    log_gaussian = -0.5 * math.log(2.0 * math.pi) - log_sigma - 0.5 * jnp.square(diff / sigma)

    # LogSumExp over components
    component_terms = log_pi + log_gaussian
    max_terms = jnp.max(component_terms, axis=-1, keepdims=True)
    return jnp.squeeze(max_terms, axis=-1) + jnp.log(jnp.sum(jnp.exp(component_terms - max_terms), axis=-1))


def quantile_forward(
    x: Any,
    w_h: Any,
    b_h: Any,
    w_quantile: Any,
    b_quantile: Any,
) -> Any:
    """Computes direct quantile predictions."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_h_arr = jnp.asarray(w_h, dtype=jnp.float32)
    b_h_arr = jnp.asarray(b_h, dtype=jnp.float32)
    w_q_arr = jnp.asarray(w_quantile, dtype=jnp.float32)
    b_q_arr = jnp.asarray(b_quantile, dtype=jnp.float32)

    hidden = jnp.maximum(jnp.dot(x_arr, w_h_arr) + b_h_arr, 0.0)
    return jnp.dot(hidden, w_q_arr) + b_q_arr


def pinball_loss(
    y_pred: Any,
    y_true: Any,
    quantiles: Any,
) -> Any:
    """Computes pinball / check loss for quantile regression."""
    yp_arr = jnp.asarray(y_pred, dtype=jnp.float32)
    yt_arr = jnp.asarray(y_true, dtype=jnp.float32)
    q_arr = jnp.asarray(quantiles, dtype=jnp.float32)

    u = yt_arr - yp_arr
    return jnp.maximum(q_arr * u, (q_arr - 1.0) * u)
