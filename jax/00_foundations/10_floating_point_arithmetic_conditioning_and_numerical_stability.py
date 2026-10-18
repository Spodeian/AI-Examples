"""Functional JAX/Equinox implementation for Topic 10:
10_floating_point_arithmetic_conditioning_and_numerical_stability
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


def welford_online_mean_variance(x: Any) -> tuple[Any, Any]:
    """Single-pass numerically stable online mean and population variance via Welford's algorithm."""
    x_np = np.asarray(x)
    n, d = x_np.shape
    mean = np.zeros(d, dtype=np.float32)
    m2 = np.zeros(d, dtype=np.float32)

    for i in range(n):
        xi = x_np[i]
        delta = xi - mean
        mean = mean + delta / float(i + 1)
        delta2 = xi - mean
        m2 = m2 + delta * delta2

    var = m2 / float(n)
    if HAS_JAX and isinstance(x, Array):
        return jnp.asarray(mean), jnp.asarray(var)
    return mean, var


def log_sum_exp_stable(x: Any, axis: int = -1) -> Any:
    """Numerically stable Log-Sum-Exp avoiding overflow/underflow."""
    if HAS_JAX and isinstance(x, Array):
        m = jnp.max(x, axis=axis, keepdims=True)
        return jnp.squeeze(m, axis=axis) + jnp.log(jnp.sum(jnp.exp(x - m), axis=axis))
    x_np = np.asarray(x)
    m = np.max(x_np, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x_np - m), axis=axis))


def matrix_condition_number(a: Any) -> float:
    """Computes condition number kappa(A) via singular value decomposition."""
    a_np = np.asarray(a)
    s = np.linalg.svd(a_np, compute_uv=False)
    s_max = float(s[0])
    s_min = float(s[-1])
    if s_min == 0.0:
        return float("inf")
    return s_max / s_min


def neumaier_sum(x: Any, axis: int = 0) -> Any:
    """Neumaier compensated summation recovering lost low-order bits."""
    x_np = np.asarray(x)
    n = x_np.shape[axis]
    s = np.zeros(x_np.shape[1], dtype=np.float32)
    c = np.zeros(x_np.shape[1], dtype=np.float32)

    for i in range(n):
        xi = x_np[i]
        t = s + xi
        mask = np.abs(s) >= np.abs(xi)
        comp = np.where(mask, (s - t) + xi, (xi - t) + s)
        c = c + comp
        s = t

    res = s + c
    if HAS_JAX and isinstance(x, Array):
        return jnp.asarray(res)
    return res


def stable_linear_forward(x: Any, weight: Any, bias: Any) -> Any:
    """Stateless linear forward evaluation: x @ weight + bias."""
    if HAS_JAX and isinstance(x, Array):
        return jnp.dot(x, weight) + bias
    return np.dot(x, weight) + bias


def stable_linear_lse_forward(x: Any, weight: Any, bias: Any) -> Any:
    """Stateless forward linear evaluation with log-sum-exp reduction."""
    logits = stable_linear_forward(x, weight, bias)
    return log_sum_exp_stable(logits, axis=-1)
