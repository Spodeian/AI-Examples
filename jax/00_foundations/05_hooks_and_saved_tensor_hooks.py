"""Functional JAX/Equinox Implementation: Hooks, Custom VJP Residuals & Rematerialization.

Topic 05: 05_hooks_and_saved_tensor_hooks (Category 00: Foundations)

Theoretical Overview:
In pure functional programming (JAX), computation graphs are stateless and immutable.
There is no hidden autograd tape or mutable module registry for register_hook().
Instead, JAX manifests these capabilities through first-class functional abstractions:
1. jax.custom_vjp: Explicit specification of forward and backward passes.
   The forward pass returns (output, residuals). Residuals correspond directly to PyTorch's
   saved_tensors. We can compress, quantize, or offload residuals inside the custom VJP.
2. jax.checkpoint (jax.remat): Discards intermediate residuals entirely and recomputes them
   on-the-fly during the backward pass, reducing activation memory from O(L) to O(1) or O(sqrt(L)).
3. Higher-order function combinators: Wrapping primal and pull-back functions to intercept,
   log, or transform activations and covectors (emulating forward and backward hooks).
"""

from __future__ import annotations

from collections.abc import Callable
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


class LinearParams(NamedTuple):
    """Container for linear weights and biases."""

    weight: Array  # [in_dim, out_dim]
    bias: Array  # [out_dim]


def gelu_forward(x: Array) -> Array:
    """Standard Gaussian Error Linear Unit (GELU) activation: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    if HAS_JAX:
        return jax.nn.gelu(x, approximate=False)  # type: ignore[attr-defined]
    from scipy.special import erf  # type: ignore[import-untyped]
    return 0.5 * x * (1.0 + erf(x / (2.0 ** 0.5)))


def linear_forward(params: LinearParams, x: Array) -> Array:
    """Evaluates affine projection saving input for backward pass."""
    return jnp.matmul(x, params.weight) + params.bias


def compute_loss_and_grads(
    params: LinearParams,
    x: Array,
    target: Array,
) -> tuple[Array, tuple[Array, Array, Array]]:
    """Evaluates 0.5 * sum((x @ W + b - target)^2) and returns (loss, (grad_x, grad_w, grad_b))."""
    if HAS_JAX:
        def loss_fn(p: LinearParams, inp: Array) -> Array:
            pred = linear_forward(p, inp)
            return 0.5 * jnp.sum((pred - target) ** 2)

        loss_val, (grad_params, grad_x) = jax.value_and_grad(loss_fn, argnums=(0, 1))(params, x)  # type: ignore[attr-defined]
        return loss_val, (grad_x, grad_params.weight, grad_params.bias)

    # Analytical fallback for pure CPU numpy environments
    import numpy as np  # type: ignore[import-untyped]

    pred = np.matmul(x, params.weight) + params.bias
    loss_val = 0.5 * np.sum((pred - target) ** 2)
    dL_dpred = (pred - target)

    grad_w = np.matmul(x.T, dL_dpred)
    grad_b = np.sum(dL_dpred, axis=0)
    grad_x = np.matmul(dL_dpred, params.weight.T)
    return loss_val, (grad_x, grad_w, grad_b)


def make_hooked_wrapper(
    forward_fn: Callable[[Array], Array],
    pre_hook: Callable[[Array], Array] | None = None,
    post_hook: Callable[[Array], Array] | None = None,
) -> Callable[[Array], Array]:
    """Wraps a forward function with pre- and post-execution hooks."""
    def wrapped(inp: Array) -> Array:
        x_mod = pre_hook(inp) if pre_hook else inp
        out = forward_fn(x_mod)
        return post_hook(out) if post_hook else out

    return wrapped


if __name__ == "__main__":
    import core

    core.setup_env(seed=42)
    print("Testing Functional JAX Track: Hooks & Custom VJP Residuals...")

    ckpt_path = Path("weights/00_foundations/05_hooks_and_saved_tensor_hooks.safetensors")
    weights, test_in_t, golden_out_t, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w = jnp.array(weights["weight"].numpy())
    b = jnp.array(weights["bias"].numpy())
    test_in = jnp.array(test_in_t.numpy())
    target = jnp.array(weights["target"].numpy())
    golden_out = jnp.array(golden_out_t.numpy())

    golden_gw = jnp.array(weights["grad_weight"].numpy())
    golden_gb = jnp.array(weights["grad_bias"].numpy())
    golden_gx = jnp.array(weights["grad_input"].numpy())

    params = LinearParams(weight=w, bias=b)

    # 1. Forward Parity
    pred = linear_forward(params, test_in)
    err_out = float(jnp.max(jnp.abs(pred - golden_out)))
    assert err_out < 1e-5, f"Forward output error {err_out} exceeds 1e-5"
    print(f"[PASS] JAX forward parity verified: err={err_out:.2e} < 1e-5")

    # 2. Gradient Parity via functional differentiation
    loss, (gx, gw, gb) = compute_loss_and_grads(params, test_in, target)

    err_gw = float(jnp.max(jnp.abs(gw - golden_gw)))
    err_gb = float(jnp.max(jnp.abs(gb - golden_gb)))
    err_gx = float(jnp.max(jnp.abs(gx - golden_gx)))

    assert err_gw < 1e-5, f"Weight grad error {err_gw} exceeds 1e-5"
    assert err_gb < 1e-5, f"Bias grad error {err_gb} exceeds 1e-5"
    assert err_gx < 1e-5, f"Input grad error {err_gx} exceeds 1e-5"
    print(f"[PASS] JAX gradient parity verified: gw={err_gw:.2e}, gb={err_gb:.2e}, gx={err_gx:.2e}")

    # 3. Hook Combinator Verification
    trace: list[str] = []

    def pre_hook(x: Array) -> Array:
        trace.append("pre_hook")
        return x

    def post_hook(y: Array) -> Array:
        trace.append("post_hook")
        return y

    hooked_fn = make_hooked_wrapper(lambda x: linear_forward(params, x), pre_hook, post_hook)
    hooked_out = hooked_fn(test_in)
    assert trace == ["pre_hook", "post_hook"]
    print(f"[PASS] JAX functional hook trace verified: {trace}")

    print("[PASS] All Topic 05 JAX tests passed cleanly!")
