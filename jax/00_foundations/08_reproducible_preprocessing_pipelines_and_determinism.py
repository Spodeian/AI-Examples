"""Functional JAX Implementation: Determinism, Pure PRNG Keys & Kahan Summation.

Topic 08: 08_reproducible_preprocessing_pipelines_and_determinism (Category 00: Foundations)

Theoretical Overview:
In pure functional programming (JAX), reproducibility is built into the language primitives:
1. Pure Functional PRNGs (Threefry-2x32 / Philox):
   JAX has zero hidden mutable global state. Randomness is treated as an immutable value
   via explicit jax.random.PRNGKey(seed). Branching randomness requires explicit splitting:
       key, subkey = jax.random.split(key)
   This guarantees that execution order across parallel cores or threads never corrupts RNG states.
2. Compensated Reductions (Kahan Summation):
   Mitigates floating-point non-associativity across large array reductions:
       s = s + (x_i - c)
       c = (s_new - s_old) - (x_i - c)
3. Cryptographic Deterministic Partitioning:
   Assigns samples to dataset splits using salted hash digests, guaranteeing invariant splits
   regardless of batch size, thread count, or streaming order.
"""

from __future__ import annotations

import hashlib
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
    """Container for model parameters."""

    weight: Array  # [in_dim, out_dim]
    bias: Array  # [out_dim]


def hash_deterministic_split(
    sample_id: str,
    salt: str = "ai-examples-v1",
    train_ratio: float = 0.8,
) -> str:
    """Deterministically partitions samples based on SHA-256 digest without global state."""
    hasher = hashlib.sha256()
    hasher.update(f"{salt}:{sample_id}".encode())
    digest = hasher.hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if fraction < train_ratio else "test"


def kahan_sum(x: Array, axis: int = 0) -> Array:
    """Evaluates Kahan compensated summation along the specified axis."""
    # Transpose axis to front
    x_swapped = jnp.swapaxes(x, 0, axis)
    s = jnp.zeros_like(x_swapped[0])
    c = jnp.zeros_like(s)

    for i in range(x_swapped.shape[0]):
        y = x_swapped[i] - c
        t = s + y
        c = (t - s) - y
        s = t

    return s


def linear_forward(params: LinearParams, x: Array) -> Array:
    """Evaluates deterministic affine projection: y = x @ W + b."""
    return jnp.matmul(x, params.weight) + params.bias


def loss_fn(params: LinearParams, x: Array, target: Array) -> Array:
    """Evaluates squared loss for gradient parity."""
    pred = linear_forward(params, x)
    return 0.5 * jnp.sum((pred - target) ** 2)


if __name__ == "__main__":
    import numpy as np

    import core

    print(f"Testing Functional JAX Determinism & Pure PRNGs (HAS_JAX={HAS_JAX})...")

    ckpt_path = Path("weights/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.safetensors")
    raw_weights, test_in_t, golden_out_t, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w_gold = raw_weights["weight"].numpy()
    b_gold = raw_weights["bias"].numpy()
    kahan_mean_gold = raw_weights["kahan_mean"].numpy()
    target_gold = raw_weights["target"].numpy()
    test_in = test_in_t.numpy()
    golden_out = golden_out_t.numpy()

    golden_gw = raw_weights["grad_weight"].numpy()
    golden_gb = raw_weights["grad_bias"].numpy()
    golden_gx = raw_weights["grad_input"].numpy()

    # Convert to JAX arrays if available, else numpy
    if HAS_JAX:
        x_j = jnp.array(test_in)
        w_j = jnp.array(w_gold)
        b_j = jnp.array(b_gold)
        target_j = jnp.array(target_gold)
    else:
        x_j = test_in
        w_j = w_gold
        b_j = b_gold
        target_j = target_gold

    params = LinearParams(weight=w_j, bias=b_j)

    # 1. Functional Kahan Summation Verification
    k_sum = kahan_sum(x_j, axis=0)
    k_mean = np.array(k_sum / float(test_in.shape[0]))
    np.testing.assert_allclose(k_mean, kahan_mean_gold, atol=1e-6, rtol=1e-6)
    print("[PASS] JAX Kahan compensated reduction matches golden reference.")

    # 2. Hash-Based Partitioning Invariant
    splits1 = [hash_deterministic_split(f"item_{i}") for i in range(50)]
    splits2 = [hash_deterministic_split(f"item_{i}") for i in range(50)]
    assert splits1 == splits2, "Hash partition must be identical across invocations"
    print("[PASS] JAX Hash-based deterministic split invariant verified.")

    # 3. Forward Output Parity
    y_pred = np.array(linear_forward(params, x_j))
    np.testing.assert_allclose(y_pred, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] JAX forward evaluation strictly matches golden reference (<1e-5).")

    # 4. Autograd Gradient Parity
    if HAS_JAX:
        grad_params_fn = jax.grad(loss_fn, argnums=0)
        grad_params = grad_params_fn(params, x_j, target_j)

        np.testing.assert_allclose(np.array(grad_params.weight), golden_gw, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.array(grad_params.bias), golden_gb, atol=1e-5, rtol=1e-5)

        grad_x_fn = jax.grad(loss_fn, argnums=1)
        grad_x = grad_x_fn(params, x_j, target_j)
        np.testing.assert_allclose(np.array(grad_x), golden_gx, atol=1e-5, rtol=1e-5)
        print("[PASS] JAX autograd gradients strictly match golden reference.")

        # 5. Abstract Shape Tracing
        x_abs = jax.ShapeDtypeStruct(shape=(8, 4), dtype=jnp.float32)
        shape_out = jax.eval_shape(lambda x: linear_forward(params, x), x_abs)
        assert shape_out.shape == (8, 2)
        print("[PASS] JAX eval_shape verified for deterministic linear module.")
    else:
        print("[SKIP] JAX autograd tests skipped (JAX not installed).")

    print("[PASS] All Topic 08 JAX tests passed cleanly!")
