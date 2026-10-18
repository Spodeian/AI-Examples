"""Functional JAX implementation for Category 01 Topic 03:
03_discrete_vs_continuous_latents_and_gradient_estimators
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


def gaussian_encode(
    x: Any,
    w_mu: Any,
    b_mu: Any,
    w_logvar: Any,
    b_logvar: Any,
) -> tuple[Any, Any]:
    """Computes Gaussian encoder mean and log-variance."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_m = jnp.asarray(w_mu, dtype=jnp.float32)
    b_m = jnp.asarray(b_mu, dtype=jnp.float32)
    w_lv = jnp.asarray(w_logvar, dtype=jnp.float32)
    b_lv = jnp.asarray(b_logvar, dtype=jnp.float32)

    mu = jnp.dot(x_arr, w_m) + b_m
    logvar = jnp.dot(x_arr, w_lv) + b_lv
    return mu, logvar


def gaussian_reparameterize(
    mu: Any,
    logvar: Any,
    eps: Any,
) -> Any:
    """Continuous pathwise reparameterization: z = mu + exp(0.5 * logvar) * eps."""
    mu_arr = jnp.asarray(mu, dtype=jnp.float32)
    lv_arr = jnp.asarray(logvar, dtype=jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=jnp.float32)

    std = jnp.exp(0.5 * lv_arr)
    return mu_arr + std * eps_arr


def gaussian_kl_divergence(
    mu: Any,
    logvar: Any,
) -> Any:
    """Analytical KL divergence between diagonal Gaussian and unit Gaussian N(0, I)."""
    mu_arr = jnp.asarray(mu, dtype=jnp.float32)
    lv_arr = jnp.asarray(logvar, dtype=jnp.float32)

    return -0.5 * jnp.sum(1.0 + lv_arr - mu_arr**2 - jnp.exp(lv_arr), axis=-1)


def linear_decode(
    z: Any,
    w_dec: Any,
    b_dec: Any,
) -> Any:
    """Linear decoder projection: z @ w_dec + b_dec."""
    z_arr = jnp.asarray(z, dtype=jnp.float32)
    w_d = jnp.asarray(w_dec, dtype=jnp.float32)
    b_d = jnp.asarray(b_dec, dtype=jnp.float32)

    return jnp.dot(z_arr, w_d) + b_d


def gumbel_softmax_sample(
    x: Any,
    w_logits: Any,
    b_logits: Any,
    gumbel_noise: Any,
    temperature: Any,
) -> Any:
    """Relaxed Gumbel-Softmax categorical sampling."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_l = jnp.asarray(w_logits, dtype=jnp.float32)
    b_l = jnp.asarray(b_logits, dtype=jnp.float32)
    g = jnp.asarray(gumbel_noise, dtype=jnp.float32)
    tau = jnp.asarray(temperature, dtype=jnp.float32)

    logits = jnp.dot(x_arr, w_l) + b_l
    unnorm = (logits + g) / tau
    # Numerically stable softmax
    max_val = jnp.max(unnorm, axis=-1, keepdims=True)
    exp_unnorm = jnp.exp(unnorm - max_val)
    return exp_unnorm / jnp.sum(exp_unnorm, axis=-1, keepdims=True)


def bernoulli_reinforce_score(
    probs: Any,
    b: Any,
    reward: Any,
    baseline: Any = 0.0,
) -> Any:
    """Score function / REINFORCE gradient estimator for Bernoulli latents."""
    p = jnp.asarray(probs, dtype=jnp.float32)
    b_arr = jnp.asarray(b, dtype=jnp.float32)
    r = jnp.asarray(reward, dtype=jnp.float32)
    base = jnp.asarray(baseline, dtype=jnp.float32)

    adv = r - base
    if adv.ndim == 1:
        adv = jnp.expand_dims(adv, axis=-1)
    return adv * (b_arr - p)
