"""Functional JAX Implementation: Tabular Scalers, Yeo-Johnson & Weight Folding.

Topic 07: 07_feature_engineering_scalers_and_tabular_preprocessing (Category 00: Foundations)

Theoretical Overview:
In stateless functional programming (JAX), preprocessing pipelines and scalers are pure
functions parameterized by immutable PyTrees:
1. Standardization:
   - Forward: z = (x - mu) / sigma
   - Exact Inverse: x = z * sigma + mu
2. Monotonic Power Transformations (Yeo-Johnson):
   Stabilizes variance and reshapes asymmetric, skewed tabular features toward standard Gaussian
   distributions across both positive and negative values.
3. Serving Weight Folding:
   Analytically fuses linear scaling (1 / sigma) and centering (-mu / sigma) into the downstream
   projection layer:
       W_fused = (1 / sigma)[:, None] * W_head
       b_fused = b_head - (mu / sigma) @ W_head
   This enables 1-step serving evaluation with 0 extra memory allocations.
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


class ScalerParams(NamedTuple):
    """Container for tabular mean and standard deviation."""

    mean: Array  # [dim]
    std: Array  # [dim]


class FusedModelParams(NamedTuple):
    """Container for fused serving weights and bias."""

    weight: Array  # [in_dim, out_dim]
    bias: Array  # [out_dim]


def standard_scale(x: Array, params: ScalerParams) -> Array:
    """Transforms raw features into zero-mean unit-variance space."""
    return (x - params.mean) / params.std


def inverse_standard_scale(z: Array, params: ScalerParams) -> Array:
    """Invertible reconstruction from standardized representation back to raw input."""
    return z * params.std + params.mean


def yeo_johnson_transform(x: Array, lmbda: float = 0.5) -> Array:
    """Evaluates the Yeo-Johnson power transform using JAX where/piecewise primitives."""
    pos_mask = x >= 0
    pos_base = jnp.maximum(x + 1.0, 1e-8)
    neg_base = jnp.maximum(-x + 1.0, 1e-8)

    pos_term = ((pos_base**lmbda) - 1.0) / lmbda if abs(lmbda) > 1e-6 else jnp.log(pos_base)
    neg_term = (
        -((neg_base ** (2.0 - lmbda)) - 1.0) / (2.0 - lmbda)
        if abs(lmbda - 2.0) > 1e-6
        else -jnp.log(neg_base)
    )

    return jnp.where(pos_mask, pos_term, neg_term)


def fold_weights(params: ScalerParams, w_head: Array, b_head: Array) -> FusedModelParams:
    """Folds standardization statistics into linear projection weights for fused graph execution."""
    inv_std = (1.0 / params.std)[:, None] if params.std.ndim == 1 else 1.0 / params.std
    w_fused = inv_std * w_head
    b_fused = b_head - jnp.matmul(params.mean / params.std, w_head)
    return FusedModelParams(weight=w_fused, bias=b_fused)


def fused_forward(params: FusedModelParams, x: Array) -> Array:
    """Evaluates fused serving prediction: y = x @ W_fused + b_fused."""
    return jnp.matmul(x, params.weight) + params.bias


def loss_fn(params: FusedModelParams, x: Array, target: Array) -> Array:
    """Computes squared loss for autograd gradient parity."""
    pred = fused_forward(params, x)
    return 0.5 * jnp.sum((pred - target) ** 2)


if __name__ == "__main__":
    import numpy as np

    import core

    print(f"Testing Functional JAX Tabular Scalers & Weight Folding (HAS_JAX={HAS_JAX})...")

    ckpt_path = Path("weights/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.safetensors")
    raw_weights, test_in_t, golden_out_t, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w_gold = raw_weights["weight"].numpy()
    b_gold = raw_weights["bias"].numpy()
    mean_gold = raw_weights["mean"].numpy()
    std_gold = raw_weights["std"].numpy()
    w_head_gold = raw_weights["w_head"].numpy()
    b_head_gold = raw_weights["b_head"].numpy()
    yj_gold = raw_weights["yj_reference"].numpy()
    target_gold = raw_weights["target"].numpy()
    test_in = test_in_t.numpy()
    golden_out = golden_out_t.numpy()

    golden_gw = raw_weights["grad_weight"].numpy()
    golden_gb = raw_weights["grad_bias"].numpy()
    golden_gx = raw_weights["grad_input"].numpy()

    # Convert to JAX arrays if available, else numpy
    if HAS_JAX:
        x_j = jnp.array(test_in)
        mean_j = jnp.array(mean_gold)
        std_j = jnp.array(std_gold)
        w_head_j = jnp.array(w_head_gold)
        b_head_j = jnp.array(b_head_gold)
        target_j = jnp.array(target_gold)
    else:
        x_j = test_in
        mean_j = mean_gold
        std_j = std_gold
        w_head_j = w_head_gold
        b_head_j = b_head_gold
        target_j = target_gold

    scaler_p = ScalerParams(mean=mean_j, std=std_j)

    # 1. StandardScaler Invertibility Check
    z_scaled = standard_scale(x_j, scaler_p)
    x_rec = np.array(inverse_standard_scale(z_scaled, scaler_p))
    np.testing.assert_allclose(x_rec, test_in, atol=1e-6, rtol=1e-6)
    print("[PASS] JAX StandardScaler: Exact invertible roundtrip verified: ||x - inv(z)||_inf < 1e-6.")

    # 2. Yeo-Johnson Transform Parity
    yj_calc = np.array(yeo_johnson_transform(x_j, lmbda=0.5))
    np.testing.assert_allclose(yj_calc, yj_gold, atol=1e-5, rtol=1e-5)
    print("[PASS] JAX Yeo-Johnson transform matches golden reference.")

    # 3. Serving Weight Folding & Forward Evaluation Parity
    fused_p = fold_weights(scaler_p, w_head_j, b_head_j)
    np.testing.assert_allclose(np.array(fused_p.weight), w_gold, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(fused_p.bias), b_gold, atol=1e-5, rtol=1e-5)

    y_pred = np.array(fused_forward(fused_p, x_j))
    np.testing.assert_allclose(y_pred, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] JAX Weight folding identity and forward parity verified.")

    # 4. Autograd Gradient Parity
    if HAS_JAX:
        grad_params_fn = jax.grad(loss_fn, argnums=0)
        grad_params = grad_params_fn(fused_p, x_j, target_j)

        np.testing.assert_allclose(np.array(grad_params.weight), golden_gw, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.array(grad_params.bias), golden_gb, atol=1e-5, rtol=1e-5)

        grad_x_fn = jax.grad(loss_fn, argnums=1)
        grad_x = grad_x_fn(fused_p, x_j, target_j)
        np.testing.assert_allclose(np.array(grad_x), golden_gx, atol=1e-5, rtol=1e-5)
        print("[PASS] JAX autograd gradients match golden reference.")

        # 5. Abstract Shape Tracing
        x_abstract = jax.ShapeDtypeStruct(shape=(16, 4), dtype=jnp.float32)
        shape_out = jax.eval_shape(lambda x: fused_forward(fused_p, x), x_abstract)
        assert shape_out.shape == (16, 2)
        print("[PASS] JAX eval_shape verified for fused tabular pipeline.")
    else:
        print("[SKIP] JAX autograd tests skipped (JAX not installed).")

    print("[PASS] All Topic 07 JAX tests passed cleanly!")
