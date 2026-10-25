"""Dynamic Shapes Mark Dynamic and Recompilation Guards — functional JAX implementation.

Topic: 12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards. Run ``python jax/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def dynamic_masked_pool(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Masked mean pooling across variable sequence lengths."""
    mask_expanded = jnp.expand_dims(mask, -1)
    sum_x = jnp.sum(x * mask_expanded, axis=1)
    lengths = jnp.sum(mask, axis=-1, keepdims=True)
    return sum_x / jnp.maximum(lengths, 1.0)


def bucket_allocation(seq_lens: jnp.ndarray, bucket_boundaries: jnp.ndarray) -> jnp.ndarray:
    """Assign sequence lengths to the smallest enclosing bucket boundary."""
    diffs = jnp.expand_dims(bucket_boundaries, 0) - jnp.expand_dims(seq_lens, 1)
    valid = diffs >= 0.0
    cost = jnp.where(valid, diffs, jnp.inf)
    idx = jnp.argmin(cost, axis=-1)
    return bucket_boundaries[idx]


def dynamic_rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Root Mean Square Normalization (Zhang & Sennrich 2019) across dynamic features."""
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return (x / jnp.sqrt(variance + float(eps))) * weight


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Masked pool
    mp = dynamic_masked_pool(g["pool_x"], g["pool_mask"])
    err = float(jnp.max(jnp.abs(mp - g["masked_pool_ref"])))
    assert err < 1e-5, f"masked_pool error {err:.2e}"

    # 3. Bucket allocation
    alloc = bucket_allocation(g["seq_lens"], g["buckets"])
    err = float(jnp.max(jnp.abs(alloc - g["allocated_buckets_ref"])))
    assert err < 1e-5, f"bucket_alloc error {err:.2e}"

    # 4. RMSNorm
    rms = dynamic_rms_norm(g["rms_x"], g["rms_w"], eps=1e-5)
    err = float(jnp.max(jnp.abs(rms - g["dynamic_rms_ref"])))
    assert err < 1e-5, f"rms error {err:.2e}"

    print("12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
