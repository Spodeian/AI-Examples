"""Tensor Parallelism Megatron Sequence and Async TP — functional JAX implementation.

Topic: 16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp. Run ``python jax/16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def megatron_tp_mlp(
    x: jnp.ndarray,
    w1_shards: jnp.ndarray,
    w2_shards: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Megatron-LM 1D Tensor Parallel MLP block."""
    world_size = w1_shards.shape[0]
    partials = []
    for r in range(world_size):
        h_r = jax.nn.gelu(x @ w1_shards[r], approximate=False)
        y_r = h_r @ w2_shards[r]
        partials.append(y_r)

    partial_shards = jnp.stack(partials, axis=0)
    y = jnp.sum(partial_shards, axis=0)
    return y, partial_shards


def sequence_parallel_mlp(
    x_seq_shards: jnp.ndarray,
    w1_shards: jnp.ndarray,
    w2_shards: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Megatron Sequence Parallelism: AllGather, TP MLP, ReduceScatter."""
    world_size = x_seq_shards.shape[0]
    x_full = jnp.concatenate([x_seq_shards[r] for r in range(world_size)], axis=1)
    y_full, _ = megatron_tp_mlp(x_full, w1_shards, w2_shards)
    chunks = jnp.split(y_full, world_size, axis=1)
    y_seq_shards = jnp.stack(chunks, axis=0)
    return y_seq_shards, x_full


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Megatron TP MLP
    tp_out, partials = megatron_tp_mlp(g["x"], g["w1_shards"], g["w2_shards"])
    err = float(jnp.max(jnp.abs(partials - g["partial_shards"])))
    assert err < 1e-5, f"partials error {err:.2e}"
    err = float(jnp.max(jnp.abs(tp_out - g["tp_mlp_out"])))
    assert err < 1e-5, f"tp_out error {err:.2e}"

    # 2. Sequence Parallelism MLP
    sp_out, x_full = sequence_parallel_mlp(g["x_seq_shards"], g["w1_shards"], g["w2_shards"])
    err = float(jnp.max(jnp.abs(x_full - g["x_full"])))
    assert err < 1e-5, f"x_full error {err:.2e}"
    err = float(jnp.max(jnp.abs(sp_out - g["sp_mlp_out_shards"])))
    assert err < 1e-5, f"sp_out error {err:.2e}"

    print("16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
