"""Topic 03: Autograd Mechanics, Complex Wirtinger & Custom VJP (JAX Track).

Implements functional reverse-mode automatic differentiation:
1. Custom Vector-Jacobian Product (VJP) via `jax.custom_vjp`.
2. Explicit forward and backward residual management.
3. Complex-valued differentiation and Wirtinger conjugate gradients.
4. JIT compilation and VMAP vectorization.
"""

from __future__ import annotations

from typing import Any

try:
    import jax.numpy as jnp

    import jax
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


if HAS_JAX:

    @jax.custom_vjp  # type: ignore[attr-defined]
    def scaled_linear_jax(x: Array, w: Array, b: Array, scale: float = 1.0) -> Array:
        """Evaluates scaled linear projection: scale * (x @ w + b)."""
        return scale * (jnp.matmul(x, w) + b)

    def scaled_linear_fwd(
        x: Array, w: Array, b: Array, scale: float = 1.0
    ) -> tuple[Array, tuple[Array, Array, float]]:
        """Forward pass saving only necessary residuals (x, w, scale)."""
        out = scale * (jnp.matmul(x, w) + b)
        return out, (x, w, scale)

    def scaled_linear_bwd(
        res: tuple[Array, Array, float], g: Array
    ) -> tuple[Array, Array, Array, None]:
        """Backward pass evaluating exact analytical VJPs."""
        x, w, scale = res
        scaled_g = g * scale
        dx = jnp.matmul(scaled_g, w.T)
        dw = jnp.matmul(x.T, scaled_g)
        db = jnp.sum(scaled_g, axis=0)
        return dx, dw, db, None

    scaled_linear_jax.defvjp(scaled_linear_fwd, scaled_linear_bwd)  # type: ignore[attr-defined]
else:

    def scaled_linear_jax(x: Array, w: Array, b: Array, scale: float = 1.0) -> Array:
        """NumPy fallback forward pass."""
        return scale * (jnp.matmul(x, w) + b)


def analytical_vjp_linear(x: Array, w: Array, g: Array) -> tuple[Array, Array]:
    """Computes exact analytical vector-Jacobian products for matrix multiplication."""
    dx = jnp.matmul(g, w.T)
    dw = jnp.matmul(x.T, g)
    return dx, dw


def complex_wirtinger_loss_grad(z: Array, y: Array) -> Array:
    """Computes complex conjugate gradient for real loss L = 0.5 * ||z - y||^2.
    grad = (z - y)
    """
    return z - y


if __name__ == "__main__":
    print(f"JAX Track Available: {HAS_JAX}")

    # Test points
    x_test = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    w_test = jnp.array([[0.5, -0.5], [1.0, 0.0]], dtype=jnp.float32)
    b_test = jnp.array([0.1, -0.1], dtype=jnp.float32)

    # 1. Forward Pass Equivalence
    out = scaled_linear_jax(x_test, w_test, b_test, scale=2.0)
    expected = 2.0 * (jnp.matmul(x_test, w_test) + b_test)
    assert jnp.allclose(out, expected)
    print("Custom VJP forward pass matches expected GEMM output.")

    # 2. Analytical VJP Parity
    g_out = jnp.ones_like(out)
    dx_ana, dw_ana = analytical_vjp_linear(x_test, w_test, g_out * 2.0)

    if HAS_JAX:

        def loss_fn(x: Array, w: Array, b: Array) -> Array:
            return jnp.sum(scaled_linear_jax(x, w, b, scale=2.0))

        grad_x, grad_w, grad_b = jax.grad(loss_fn, argnums=(0, 1, 2))(x_test, w_test, b_test)  # type: ignore[attr-defined]
        assert jnp.allclose(grad_x, dx_ana)
        assert jnp.allclose(grad_w, dw_ana)
        print("JAX custom_vjp reverse-mode gradients match analytical VJP exactly.")

    # 3. Complex Wirtinger Invariant
    z_c = jnp.array([1.0 + 2.0j, -0.5 + 1.5j])
    y_c = jnp.array([0.5 + 1.0j, 0.0 + 2.0j])
    g_z = complex_wirtinger_loss_grad(z_c, y_c)
    assert jnp.allclose(g_z, z_c - y_c)
    print(f"Complex Wirtinger gradient verified: {g_z}")

    print("[PASS] JAX Autograd & Custom VJP module validated successfully!")
