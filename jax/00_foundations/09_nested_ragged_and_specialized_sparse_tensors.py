"""Functional JAX/Equinox implementation for Topic 09:
09_nested_ragged_and_specialized_sparse_tensors
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


def ragged_to_padded(
    values: Any,
    offsets: Any,
    padding_value: float = 0.0,
) -> tuple[Any, Any]:
    """Packs ragged flat array into dense padded array and boolean mask."""
    num_seqs = len(offsets) - 1
    seq_lens = [int(offsets[i + 1] - offsets[i]) for i in range(num_seqs)]
    max_len = max(seq_lens)
    dim = values.shape[-1]

    padded = np.full((num_seqs, max_len, dim), fill_value=padding_value, dtype=np.float32)
    mask = np.zeros((num_seqs, max_len), dtype=bool)

    for i in range(num_seqs):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        length = end - start
        padded[i, :length] = np.array(values[start:end])
        mask[i, :length] = True

    if HAS_JAX and isinstance(values, Array):
        return jnp.asarray(padded), jnp.asarray(mask)
    return padded, mask


def padded_to_ragged(
    padded: Any,
    mask: Any,
) -> tuple[Any, Any]:
    """Flattens a dense padded batch back into contiguous ragged token format."""
    num_seqs = padded.shape[0]
    lengths = [int(np.sum(mask[i])) for i in range(num_seqs)]
    offsets = np.zeros(num_seqs + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    total_tokens = int(offsets[-1])
    dim = padded.shape[-1]
    values = np.zeros((total_tokens, dim), dtype=np.float32)

    for i in range(num_seqs):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        values[start:end] = np.array(padded[i, : lengths[i]])

    if HAS_JAX and isinstance(padded, Array):
        return jnp.asarray(values), jnp.asarray(offsets)
    return values, offsets


def ragged_mean_pool(
    values: Any,
    offsets: Any,
) -> Any:
    """Computes mean reduction for each ragged sequence slice defined by offsets."""
    num_seqs = len(offsets) - 1
    dim = values.shape[-1]
    pooled = np.zeros((num_seqs, dim), dtype=np.float32)

    for i in range(num_seqs):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        length = float(end - start)
        if length > 0.0:
            slice_vals = np.array(values[start:end])
            pooled[i] = np.sum(slice_vals, axis=0) / length

    if HAS_JAX and isinstance(values, Array):
        return jnp.asarray(pooled)
    return pooled


def csr_spmv(
    crow_indices: Any,
    col_indices: Any,
    values: Any,
    x: Any,
    num_rows: int,
    num_cols: int,
) -> Any:
    """Evaluates Compressed Sparse Row matrix-vector/matrix product: Y = A @ X."""
    dim_out = x.shape[-1]
    y = np.zeros((num_rows, dim_out), dtype=np.float32)

    crow_np = np.asarray(crow_indices)
    col_np = np.asarray(col_indices)
    val_np = np.asarray(values)
    x_np = np.asarray(x)

    for r in range(num_rows):
        start = int(crow_np[r])
        end = int(crow_np[r + 1])
        row_sum = np.zeros(dim_out, dtype=np.float32)
        for idx in range(start, end):
            c = int(col_np[idx])
            v = val_np[idx]
            row_sum += v * x_np[c]
        y[r] = row_sum

    if HAS_JAX and isinstance(x, Array):
        return jnp.asarray(y)
    return y


def ragged_linear_forward(
    values: Any,
    weight: Any,
    bias: Any,
) -> Any:
    """Stateless forward token projection: values @ weight + bias."""
    if HAS_JAX and isinstance(values, Array):
        return jnp.dot(values, weight) + bias
    return np.dot(values, weight) + bias


def ragged_linear_pooled_forward(
    values: Any,
    offsets: Any,
    weight: Any,
    bias: Any,
) -> Any:
    """Stateless forward token projection followed by per-sequence mean reduction."""
    tokens = ragged_linear_forward(values, weight, bias)
    return ragged_mean_pool(tokens, offsets)
