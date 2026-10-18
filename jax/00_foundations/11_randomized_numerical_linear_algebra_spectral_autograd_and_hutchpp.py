"""Functional JAX/Equinox implementation for Topic 11:
11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp
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


def modified_gram_schmidt_qr(y: Any) -> tuple[Any, Any]:
    """Computes thin QR decomposition Y = Q R via Modified Gram-Schmidt with strictly positive diagonal R."""
    y_np = np.asarray(y, dtype=np.float32)
    m, num_cols = y_np.shape
    q = y_np.copy()
    r = np.zeros((num_cols, num_cols), dtype=np.float32)

    for j in range(num_cols):
        v = q[:, j].copy()
        for i in range(j):
            rij = np.dot(q[:, i], v)
            r[i, j] = rij
            v = v - rij * q[:, i]
        rjj = float(np.linalg.norm(v))
        r[j, j] = rjj
        q[:, j] = v / (rjj + 1e-12)

    if HAS_JAX and isinstance(y, Array):
        return jnp.asarray(q), jnp.asarray(r)
    return q, r


def randomized_range_finder(a: Any, omega: Any) -> Any:
    """Forms an orthonormal basis Q spanning the dominant range of A via sketch Y = A Omega."""
    a_arr = jnp.asarray(a) if HAS_JAX else np.asarray(a)
    om_arr = jnp.asarray(omega) if HAS_JAX else np.asarray(omega)
    y = a_arr @ om_arr
    q, _ = modified_gram_schmidt_qr(y)
    return q


def hutch_plus_plus_trace(a: Any, s_sketch: Any, g_mc: Any) -> Any:
    """Evaluates Hutch++ stochastic trace estimator combining low-rank sketch with residual Monte Carlo."""
    a_arr = np.asarray(a, dtype=np.float32)
    s_arr = np.asarray(s_sketch, dtype=np.float32)
    g_arr = np.asarray(g_mc, dtype=np.float32)

    # 1. Low-rank subspace sketch
    y = a_arr @ s_arr
    q, _ = modified_gram_schmidt_qr(y)

    # 2. Exact trace of projection on Q: tr(Q^T A Q)
    q_t_a_q = q.T @ a_arr @ q
    tr_proj = float(np.trace(q_t_a_q))

    # 3. Residual Monte Carlo on (I - Q Q^T) A
    a_g = a_arr @ g_arr
    proj_g = q @ (q.T @ a_g)
    residual_g = a_g - proj_g

    m_val = float(g_arr.shape[1])
    quad_forms = np.sum(g_arr * residual_g, axis=0)
    tr_res = float(np.sum(quad_forms) / m_val)

    total_tr = np.array([tr_proj + tr_res], dtype=np.float32)
    if HAS_JAX and isinstance(a, Array):
        return jnp.asarray(total_tr)
    return total_tr


def lorentzian_broadened_f_matrix(singular_values: Any, epsilon: float = 1e-3) -> Any:
    """Constructs skew-symmetric Lorentzian-broadened singular value coupling matrix F_ij.

    F_ij = (sigma_j - sigma_i) / ((sigma_j - sigma_i)^2 + epsilon^2) for i != j, 0 for i == j.
    """
    s = np.asarray(singular_values, dtype=np.float32)
    n_dim = s.shape[0]
    s_i = s[:, np.newaxis]
    s_j = s[np.newaxis, :]
    delta = s_j - s_i

    f = delta / (delta**2 + (epsilon**2))
    f = f * (1.0 - np.eye(n_dim, dtype=np.float32))

    if HAS_JAX and isinstance(singular_values, Array):
        return jnp.asarray(f)
    return f


def power_iteration_spectral_norm(
    w: Any,
    v_init: Any,
    num_iters: int = 5,
) -> tuple[Any, Any, Any]:
    """Evaluates dominant singular value (spectral norm) and singular vectors via Power Iteration."""
    w_arr = np.asarray(w, dtype=np.float32)
    v = np.asarray(v_init, dtype=np.float32)
    v = v / (float(np.linalg.norm(v)) + 1e-12)
    u = np.zeros(w_arr.shape[0], dtype=np.float32)

    for _ in range(num_iters):
        u_unnorm = w_arr @ v
        u = u_unnorm / (float(np.linalg.norm(u_unnorm)) + 1e-12)
        v_unnorm = w_arr.T @ u
        v = v_unnorm / (float(np.linalg.norm(v_unnorm)) + 1e-12)

    sigma = float(np.dot(u, w_arr @ v))
    sigma_arr = np.array([sigma], dtype=np.float32)

    if HAS_JAX and isinstance(w, Array):
        return jnp.asarray(sigma_arr), jnp.asarray(u), jnp.asarray(v)
    return sigma_arr, u, v


class FunctionalSpectralLinear:
    """Stateless functional linear module for JAX/NumPy runtimes."""

    def __init__(self, weight: Any, bias: Any) -> None:
        self.weight = weight
        self.bias = bias

    def __call__(self, x: Any) -> Any:
        return x @ self.weight + self.bias
