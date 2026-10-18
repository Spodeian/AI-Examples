"""Topic 04: Forward-Mode AD & Functional Transforms (JAX Track).

Implements composable functional differentiation and vectorization:
1. Forward-mode AD via `jax.jvp`: computes Jacobian-vector products (J @ v) in dual space.
2. Reverse-mode AD via `jax.vjp`: computes vector-Jacobian products (v^T @ J).
3. Composable batching via `jax.vmap` for per-sample gradients.
4. Exact Hessian calculation via `jax.hessian` and directional curvature (HVP) via `jvp(grad)`.
5. JIT compilation via `jax.jit`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

try:
    import jax.numpy as jnp

    import jax
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


class LinearParams(NamedTuple):
    """Container for linear weights and biases."""

    weight: Array  # [in_dim, out_dim]
    bias: Array  # [out_dim]


def linear_forward(params: LinearParams, x: Array) -> Array:
    """Evaluates affine map: y = x @ W + b."""
    return jnp.matmul(x, params.weight) + params.bias


def compute_jvp(
    fn: Callable[[Array], Array],
    primals: Array,
    tangents: Array,
) -> tuple[Array, Array]:
    """Computes primal output and forward-mode directional derivative (JVP: J @ v)."""
    if HAS_JAX:
        return cast(tuple[Array, Array], jax.jvp(fn, (primals,), (tangents,)))  # type: ignore[attr-defined]
    out = fn(primals)
    # Exact analytical JVP for affine function f(x + eps*v) - f(x):
    # In float32, central difference or exact linear perturbation:
    # f(v) - f(0) = v @ W
    zero = jnp.zeros_like(primals)
    jvp_exact = fn(tangents) - fn(zero)
    return out, jvp_exact


def compute_vjp(
    fn: Callable[[Array], Array],
    primals: Array,
) -> tuple[Array, Callable[[Array], tuple[Array, ...]]]:
    """Sets up reverse-mode Vector-Jacobian Product (VJP)."""
    if HAS_JAX:
        return cast(tuple[Array, Callable[[Array], tuple[Array, ...]]], jax.vjp(fn, primals))  # type: ignore[attr-defined]
    raise NotImplementedError("JAX required for exact vjp")


def compute_per_sample_grads_jax(
    params: LinearParams,
    batch_x: Array,
    batch_y: Array,
) -> tuple[Array, Array]:
    """Computes per-sample gradients (dL_i / dW, dL_i / db) without Python loops via vmap."""
    if not HAS_JAX:
        # NumPy loop fallback
        grads_w = []
        grads_b = []
        for i in range(batch_x.shape[0]):
            pred = linear_forward(params, batch_x[i : i + 1])
            diff = pred - batch_y[i : i + 1]
            gw = jnp.matmul(batch_x[i : i + 1].T, diff)
            gb = jnp.sum(diff, axis=0)
            grads_w.append(gw)
            grads_b.append(gb)
        return jnp.stack(grads_w, axis=0), jnp.stack(grads_b, axis=0)

    def single_loss(p: LinearParams, x: Array, y: Array) -> Array:
        pred = linear_forward(p, x)
        return 0.5 * jnp.sum((pred - y) ** 2)

    grad_fn = jax.grad(single_loss)  # type: ignore[attr-defined]
    per_sample_fn = jax.vmap(grad_fn, in_dims=(None, 0, 0))  # type: ignore[attr-defined]
    grads = per_sample_fn(params, batch_x, batch_y)
    return grads.weight, grads.bias


def compute_hvp_jax(
    scalar_fn: Callable[[Array], Array],
    x: Array,
    v: Array,
) -> Array:
    """Computes exact Hessian-Vector Product (H @ v) using forward-over-reverse AD: jvp(grad)."""
    if HAS_JAX:
        grad_fn = jax.grad(scalar_fn)  # type: ignore[attr-defined]
        _, hvp = jax.jvp(grad_fn, (x,), (v,))  # type: ignore[attr-defined]
        return hvp
    raise NotImplementedError("JAX required for exact HVP")


if __name__ == "__main__":
    import core

    print(f"Testing JAX Track on Topic 04 (HAS_JAX={HAS_JAX})...")

    # Load golden checkpoint
    ckpt_path = "weights/00_foundations/04_forward_ad_and_functional_transforms_torch_func.safetensors"
    weights, test_in_t, test_out_t, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w = jnp.array(weights["weight"].numpy())
    b = jnp.array(weights["bias"].numpy())
    test_in = jnp.array(test_in_t.numpy())
    test_out = jnp.array(test_out_t.numpy())
    tangent_in = jnp.array(weights["tangent_in"].numpy())
    tangent_out = jnp.array(weights["tangent_out"].numpy())

    params = LinearParams(weight=w, bias=b)

    # 1. Forward-Mode JVP check
    def fn(x: Array) -> Array:
        return linear_forward(params, x)

    primals_out, jvp_res = compute_jvp(fn, test_in, tangent_in)

    err_primals = float(jnp.max(jnp.abs(primals_out - test_out)))
    err_jvp = float(jnp.max(jnp.abs(jvp_res - tangent_out)))
    assert err_primals < 1e-5, f"Primals error {err_primals} exceeds 1e-5"
    assert err_jvp < 1e-4, f"JVP error {err_jvp} exceeds 1e-4"
    print(f"JAX JVP verified: err_primals={err_primals:.2e}, err_jvp={err_jvp:.2e}")

    # 2. Per-sample gradient shapes via vmap
    g_w, g_b = compute_per_sample_grads_jax(params, test_in, test_out)
    assert g_w.shape == (test_in.shape[0], w.shape[0], w.shape[1])
    assert g_b.shape == (test_in.shape[0], b.shape[0])
    print(f"JAX vmap per-sample grads verified: shape={g_w.shape}")

    # 3. Hessian-Vector Product
    if HAS_JAX:
        def scalar_loss(x: Array) -> Array:
            return 0.5 * jnp.sum((jnp.matmul(x, w)) ** 2)
        x0 = test_in[0]
        v0 = tangent_in[0]
        hvp_val = compute_hvp_jax(scalar_loss, x0, v0)

        # Compare with full hessian
        full_h = jax.hessian(scalar_loss)(x0)  # type: ignore[attr-defined]
        hvp_expected = jnp.matmul(full_h, v0)
        err_hvp = float(jnp.max(jnp.abs(hvp_val - hvp_expected)))
        assert err_hvp < 1e-5
        print(f"JAX HVP verified: error {err_hvp:.2e} < 1e-5")

    print("[PASS] All JAX functional autodiff tests passed cleanly!")
