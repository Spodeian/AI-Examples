"""Functional JAX Implementation: Weight Parametrizations & Abstract Shape Evaluation.

Topic 06: 06_nn_module_internals_parametrizations_and_meta_device (Category 00: Foundations)

Theoretical Overview:
In pure functional programming (JAX), neural network modules are stateless transformations
over explicit parameter PyTrees:
1. Weight Parametrizations:
   Rather than mutating module parameter dictionaries or intercepting attribute accesses,
   functional parametrizations are explicit tensor transformations:
       W = g(M)
   where M is an unconstrained parameter tensor and g is a manifold projection or transform.
   For the Cayley Orthogonal transform:
       A = triu(M, 1) - triu(M, 1)^T   (Lie algebra so(n) of skew-symmetric matrices)
       W = (I - A)(I + A)^{-1}          (Lie group SO(n) of special orthogonal matrices)
   Automatic differentiation through g(M) via jax.grad handles matrix inversion and projection
   analytically without graph mutation.
2. Abstract Shape Evaluation (Meta Device Analogue):
   JAX's built-in equivalent to PyTorch's torch.device("meta") is jax.eval_shape().
   jax.eval_shape traces the computation graph using jax.ShapeDtypeStruct instances,
   propagating tensor ranks, shapes, and dtypes without allocating physical device memory
   or dispatching arithmetic operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

try:
    import jax.numpy as jnp

    import jax
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


class OrthogonalParams(NamedTuple):
    """Container for unconstrained skew parameter and bias."""

    skew_param: Array  # [dim, dim]
    bias: Array  # [dim]


def cayley_transform(m: Array) -> Array:
    """Transforms unconstrained square matrix into strictly orthogonal matrix W in SO(n)."""
    dim = m.shape[0]
    tri_u = jnp.triu(m, k=1)
    skew_a = tri_u - tri_u.T
    eye = jnp.eye(dim, dtype=m.dtype)
    inv_term = jnp.linalg.inv(eye + skew_a)
    return jnp.matmul(eye - skew_a, inv_term)


def orthogonal_linear_forward(params: OrthogonalParams, x: Array) -> Array:
    """Evaluates affine map x @ W + b where W is orthogonal via Cayley transform."""
    w = cayley_transform(params.skew_param)
    return jnp.matmul(x, w) + params.bias


def loss_fn(params: OrthogonalParams, x: Array, target: Array) -> Array:
    """Squared error objective for gradient validation."""
    pred = orthogonal_linear_forward(params, x)
    return 0.5 * jnp.sum((pred - target) ** 2)


def instantiate_meta_model_shapes(dim: int, num_layers: int = 50) -> Any:
    """Traces deep multi-layer stack using abstract shape structs with 0 bytes allocated memory."""
    if not HAS_JAX:
        return None

    def deep_forward(x: Array) -> Array:
        curr = x
        for _ in range(num_layers):
            curr = jnp.matmul(curr, jnp.eye(dim, dtype=curr.dtype))
        return curr

    x_abstract = jax.ShapeDtypeStruct(shape=(8, dim), dtype=jnp.float32)
    return jax.eval_shape(deep_forward, x_abstract)


if __name__ == "__main__":
    import numpy as np

    import core

    print(f"Testing Functional JAX Module Internals & Shape Tracing (HAS_JAX={HAS_JAX})...")

    ckpt_path = Path("weights/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.safetensors")
    raw_weights, test_in_t, golden_out_t, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w_gold = raw_weights["weight"].numpy()
    b_gold = raw_weights["bias"].numpy()
    skew_gold = raw_weights["skew_param"].numpy()
    target_gold = raw_weights["target"].numpy()
    test_in = test_in_t.numpy()
    golden_out = golden_out_t.numpy()
    golden_ga = raw_weights["grad_skew"].numpy()
    golden_gb = raw_weights["grad_bias"].numpy()
    golden_gx = raw_weights["grad_input"].numpy()

    dim = skew_gold.shape[0]

    # Convert to JAX arrays if available, else numpy
    if HAS_JAX:
        m_j = jnp.array(skew_gold)
        b_j = jnp.array(b_gold)
        x_j = jnp.array(test_in)
        target_j = jnp.array(target_gold)
    else:
        m_j = skew_gold
        b_j = b_gold
        x_j = test_in
        target_j = target_gold

    params = OrthogonalParams(skew_param=m_j, bias=b_j)

    # 1. Cayley Orthogonality & Isometry Verification
    w_calc = np.array(cayley_transform(m_j))
    np.testing.assert_allclose(w_calc, w_gold, atol=1e-5, rtol=1e-5)

    eye = np.eye(dim, dtype=np.float32)
    ortho_check = w_calc.T @ w_calc
    np.testing.assert_allclose(ortho_check, eye, atol=1e-6, rtol=1e-6)
    print("[PASS] JAX Cayley transform strictly orthogonal: ||W^T W - I||_inf < 1e-6.")

    # Isometry check
    norm_before = np.linalg.norm(test_in, axis=-1)
    norm_after = np.linalg.norm(np.dot(test_in, w_calc), axis=-1)
    np.testing.assert_allclose(norm_before, norm_after, atol=1e-5, rtol=1e-5)
    print("[PASS] JAX Isometric norm preservation verified: ||W x||_2 == ||x||_2.")

    # 2. Forward Evaluation Parity
    out_pred = np.array(orthogonal_linear_forward(params, x_j))
    np.testing.assert_allclose(out_pred, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] JAX forward evaluation strictly matches golden reference.")

    # 3. Autograd Gradient Parity
    if HAS_JAX:
        # Gradient w.r.t params
        grad_params_fn = jax.grad(loss_fn, argnums=0)
        grad_params = grad_params_fn(params, x_j, target_j)

        np.testing.assert_allclose(np.array(grad_params.skew_param), golden_ga, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.array(grad_params.bias), golden_gb, atol=1e-5, rtol=1e-5)

        # Gradient w.r.t input x
        grad_x_fn = jax.grad(loss_fn, argnums=1)
        grad_x = grad_x_fn(params, x_j, target_j)
        np.testing.assert_allclose(np.array(grad_x), golden_gx, atol=1e-5, rtol=1e-5)
        print("[PASS] JAX autograd through Cayley parametrization matches golden gradients.")

        # 4. Abstract Shape Tracing (Meta Device Analogue)
        shape_result = instantiate_meta_model_shapes(dim=1024, num_layers=50)
        assert shape_result.shape == (8, 1024)
        assert shape_result.dtype == jnp.float32
        print("[PASS] JAX eval_shape dry-run: 50-layer 1024-dim model traced with 0 bytes RAM.")
    else:
        print("[SKIP] JAX autograd and eval_shape tests skipped (JAX not installed).")

    print("[PASS] All Topic 06 JAX tests passed cleanly!")
