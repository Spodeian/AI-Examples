"""Topic 01: Linear Neuron and Invertible Transforms (JAX / Functional Track).

Implements the fundamental univariate and multivariate affine projection in a pure,
stateless functional form compatible with JAX transformations:
    jax.jit  (XLA compilation)
    jax.grad (Reverse-mode automatic differentiation)
    jax.vmap (Batched vectorization)

Features:
- Pure functional parameters container (NamedTuple pytree).
- Exact analytical inverse and Jacobian log-determinant.
- Functional coordinate transforms (Min-Max, Z-Score, Tanh).
- Closed-form OLS normal equations solver via jnp.linalg.pinv.
"""

from __future__ import annotations

from typing import Any, NamedTuple

try:
    import jax.numpy as jnp

    import jax
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
except (ImportError, AttributeError):
    # Fallback to numpy for environments without native JAX
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


class LinearNeuronParams(NamedTuple):
    """Stateless PyTree container for linear neuron parameters."""

    weight: Array  # [in_dim, out_dim]
    bias: Array  # [out_dim]


class TransformState(NamedTuple):
    """Stateless container for coordinate transform statistics."""

    scale: Array
    shift: Array


def init_linear_params(in_dim: int, out_dim: int, key: Any = None) -> LinearNeuronParams:
    """Initializes linear weights and biases deterministically."""
    if HAS_JAX and key is not None:
        w_key, b_key = jax.random.split(key)  # type: ignore[attr-defined]
        w = jax.random.normal(w_key, (in_dim, out_dim)) * (1.0 / (in_dim**0.5))  # type: ignore[attr-defined]
        b = jax.random.normal(b_key, (out_dim,)) * 0.01  # type: ignore[attr-defined]
    else:
        # Deterministic default weights
        w = jnp.ones((in_dim, out_dim), dtype=jnp.float32) * 2.0
        b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return LinearNeuronParams(weight=w, bias=b)


def forward(params: LinearNeuronParams, x: Array) -> Array:
    """Pure functional affine forward pass: y = x @ W + b."""
    return jnp.matmul(x, params.weight) + params.bias


def inverse(params: LinearNeuronParams, y: Array) -> Array:
    """Exact analytic inverse pass: x = (y - b) @ W^{-1}."""
    centered = y - params.bias
    # Solve system: W.T @ x.T = centered.T
    return jnp.linalg.solve(params.weight.T, centered.T).T


def log_abs_det_jacobian(params: LinearNeuronParams) -> float:
    """Computes log-determinant of Jacobian: log |det(W)|."""
    sign, logdet = jnp.linalg.slogdet(params.weight)
    return float(logdet)


def condition_number(params: LinearNeuronParams) -> float:
    """Computes spectral condition number kappa(W) = sigma_max / sigma_min."""
    s = jnp.linalg.svd(params.weight, compute_uv=False)
    return float(s[0] / s[-1])


# Functional Coordinate Transforms
def min_max_transform(x: Array, eps: float = 1e-8) -> tuple[Array, TransformState]:
    """Scales x to range [-1.0, 1.0]."""
    d_min = jnp.min(x, axis=0, keepdims=True)
    d_max = jnp.max(x, axis=0, keepdims=True)
    diff = jnp.maximum(d_max - d_min, eps)
    normed = 2.0 * (x - d_min) / diff - 1.0
    return normed, TransformState(scale=diff, shift=d_min)


def undo_min_max(t: Array, state: TransformState) -> Array:
    """Inverts Min-Max transform."""
    return (t + 1.0) * 0.5 * state.scale + state.shift


def z_score_transform(x: Array, eps: float = 1e-7) -> tuple[Array, TransformState]:
    """Standardizes x to zero mean and unit variance."""
    mu = jnp.mean(x, axis=0, keepdims=True)
    sigma = jnp.maximum(jnp.std(x, axis=0, keepdims=True), eps)
    return (x - mu) / sigma, TransformState(scale=sigma, shift=mu)


def undo_z_score(t: Array, state: TransformState) -> Array:
    """Inverts Z-Score transform."""
    return t * state.scale + state.shift


def tanh_transform(x: Array, scale: float = 1.0, shift: float = 0.0) -> Array:
    """Non-linear bounded invertible coordinate projection."""
    return jnp.tanh(scale * (x - shift))


def undo_tanh(t: Array, scale: float = 1.0, shift: float = 0.0, eps: float = 1e-6) -> Array:
    """Inverts bounded tanh projection."""
    clipped = jnp.clip(t, -1.0 + eps, 1.0 - eps)
    return jnp.arctanh(clipped) / scale + shift


def solve_ols_jax(x: Array, y: Array) -> tuple[Array, Array]:
    """Solves Ordinary Least Squares via Moore-Penrose pseudoinverse."""
    batch_size = x.shape[0]
    ones = jnp.ones((batch_size, 1), dtype=x.dtype)
    x_design = jnp.concatenate([x, ones], axis=1)

    theta = jnp.matmul(
        jnp.linalg.pinv(jnp.matmul(x_design.T, x_design)),
        jnp.matmul(x_design.T, y),
    )
    return theta[:-1, :], theta[-1, :]


if __name__ == "__main__":
    print(f"JAX Track Available: {HAS_JAX}")

    # Initialize 2D invertible system
    w_mat = jnp.array([[2.0, 0.5], [0.3, 1.5]], dtype=jnp.float32)
    b_vec = jnp.array([0.2, -0.4], dtype=jnp.float32)
    params = LinearNeuronParams(weight=w_mat, bias=b_vec)

    # Synthetic test points
    x_val = jnp.array([[-1.0, 2.0], [0.0, 0.5], [1.5, -0.5], [2.0, 1.0]], dtype=jnp.float32)

    # 1. Forward and Inverse Round-Trip Invariant
    y_val = forward(params, x_val)
    x_rec = inverse(params, y_val)
    recon_err = float(jnp.max(jnp.abs(x_val - x_rec)))
    print(f"Reconstruction Error (L_inf): {recon_err:.2e}")
    assert recon_err < 1e-5, f"Reconstruction error {recon_err} exceeds 1e-5"

    # 2. Conditioning and Log-Det
    cond = condition_number(params)
    logdet = log_abs_det_jacobian(params)
    print(f"Condition Number kappa(W): {cond:.4f}")
    print(f"Log-Determinant log |det(W)|: {logdet:.4f}")

    # 3. Z-Score Invertibility
    x_normed, s_z = z_score_transform(x_val)
    x_unnormed = undo_z_score(x_normed, s_z)
    z_err = float(jnp.max(jnp.abs(x_val - x_unnormed)))
    assert z_err < 1e-5, f"Z-score inversion error {z_err} exceeds 1e-5"
    print("Z-score inversion verified.")

    # 4. If JAX installed, verify jit compilation and gradients
    if HAS_JAX:
        jit_forward = jax.jit(forward)  # type: ignore[attr-defined]
        y_jit = jit_forward(params, x_val)
        assert jnp.allclose(y_val, y_jit)

        def loss_fn(p: LinearNeuronParams, x_in: Array, y_target: Array) -> Array:
            pred = forward(p, x_in)
            return jnp.mean((pred - y_target) ** 2)

        grads = jax.grad(loss_fn)(params, x_val, y_val)  # type: ignore[attr-defined]
        print("JAX jit and grad verified successfully.")

    print("[PASS] JAX Linear Neuron module validated successfully!")
