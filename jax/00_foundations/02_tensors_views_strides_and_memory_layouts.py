"""Topic 02: Tensors, Views, Strides & Memory Layouts (JAX Track).

Implements tensor layout abstractions, functional sliding window extraction,
and Einstein summation within JAX's functional, immutable array model.

Features:
- Functional sliding windows via `sliding_window_view`.
- High-performance tensor contraction with `jnp.einsum`.
- JIT compilation and VMAP vectorization over sliding patches.
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


def linear_storage_index(
    shape: tuple[int, ...], strides: tuple[int, ...], coords: tuple[int, ...]
) -> int:
    """Calculates linear 1D memory offset: index = sum_k (coords[k] * strides[k])."""
    if len(coords) != len(shape):
        raise ValueError("Coordinate dimensionality mismatch")
    return sum(c * s for c, s in zip(coords, strides))


def standard_c_strides(shape: tuple[int, ...], itemsize: int = 1) -> tuple[int, ...]:
    """Generates standard contiguous Row-Major (C-style) strides for given shape."""
    if not shape:
        return ()
    strides = [itemsize]
    for dim in reversed(shape[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def extract_sliding_windows(x: Array, window_size: int, step: int = 1) -> Array:
    """Extracts rolling window views across axis=1 using strided functional slicing."""
    batch_size, length = x.shape
    num_windows = (length - window_size) // step + 1

    # In JAX/NumPy, sliding_window_view or vmap over slice indices
    if hasattr(jnp.lib, "stride_tricks") and hasattr(jnp.lib.stride_tricks, "sliding_window_view"):
        # sliding_window_view returns [B, num_windows, window_size]
        windows = jnp.lib.stride_tricks.sliding_window_view(x, window_shape=window_size, axis=1)
        return windows[:, ::step, :]
    else:
        # Fallback slice stack
        return jnp.stack(
            [x[:, i * step : i * step + window_size] for i in range(num_windows)], axis=1
        )


def batched_bilinear_einsum(x: Array, w: Array, y: Array) -> Array:
    """Computes quadratic form sum_i x_i W_ij y_j per batch element using jnp.einsum."""
    return jnp.einsum("bi,ij,bj->b", x, w, y)


def attention_scores_einsum(q: Array, k: Array) -> Array:
    """Computes dot-product multi-head attention scores QK^T."""
    return jnp.einsum("bhqd,bhkd->bhqk", q, k)


if __name__ == "__main__":
    print(f"JAX Track Available: {HAS_JAX}")

    # 1. Stride Arithmetic Verification
    shape_3d = (2, 3, 4)
    strides_c = standard_c_strides(shape_3d)
    assert strides_c == (12, 4, 1)

    coord = (1, 2, 3)
    idx = linear_storage_index(shape_3d, strides_c, coord)
    assert idx == 23
    print(f"Computed C-strides: {strides_c} | Linear index for {coord} = {idx}")

    # 2. Sliding Window Extraction
    signal = jnp.linspace(0.0, 10.0, 16)[None, :]  # [1, 16]
    windows = extract_sliding_windows(signal, window_size=4, step=2)
    assert windows.shape == (1, 7, 4)
    print(f"Extracted sliding window shape: {windows.shape}")

    # 3. Einsum Contraction Verification
    x_val = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    w_val = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)
    y_val = jnp.array([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float32)

    res = batched_bilinear_einsum(x_val, w_val, y_val)
    expected = jnp.sum(x_val @ w_val * y_val, axis=-1)
    assert jnp.allclose(res, expected)
    print("Bilinear einsum matches GEMM perfectly.")

    # 4. JIT compilation if JAX available
    if HAS_JAX:
        jit_einsum = jax.jit(batched_bilinear_einsum)  # type: ignore[attr-defined]
        res_jit = jit_einsum(x_val, w_val, y_val)
        assert jnp.allclose(res, res_jit)
        print("JAX JIT einsum verified.")

    print("[PASS] JAX Tensor Views & Strides module validated successfully!")
