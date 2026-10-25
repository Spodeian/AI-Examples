"""Collectives and DDP Gradient Bucketing Mechanics — functional JAX implementation.

Topic: 16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics. Run ``python jax/16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ring_allreduce_simulate(rank_data: jnp.ndarray) -> jnp.ndarray:
    """Simulate 2*(W-1) Ring AllReduce collective across simulated ranks."""
    world_size, total_elements = rank_data.shape
    chunk_size = total_elements // world_size

    chunks = rank_data.reshape((world_size, world_size, chunk_size))

    # Phase 1: Scatter-Reduce
    for step in range(world_size - 1):
        send_chunks = chunks
        new_chunks = chunks
        for r in range(world_size):
            recv_chunk_idx = (r - step - 1) % world_size
            prev_rank = (r - 1) % world_size
            val = new_chunks[r, recv_chunk_idx] + send_chunks[prev_rank, recv_chunk_idx]
            new_chunks = new_chunks.at[r, recv_chunk_idx].set(val)
        chunks = new_chunks

    # Phase 2: AllGather
    for step in range(world_size - 1):
        send_chunks = chunks
        new_chunks = chunks
        for r in range(world_size):
            recv_chunk_idx = (r - step) % world_size
            prev_rank = (r - 1) % world_size
            val = send_chunks[prev_rank, recv_chunk_idx]
            new_chunks = new_chunks.at[r, recv_chunk_idx].set(val)
        chunks = new_chunks

    return chunks[0].reshape((total_elements,))


def ddp_bucketed_allreduce(
    rank_grads: jnp.ndarray,
    bucket_cap: int,
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """DDP gradient bucketing with asynchronous bucket-level reduction simulation."""
    world_size, num_params = rank_grads.shape
    num_buckets = (num_params + bucket_cap - 1) // bucket_cap
    bucket_results = []

    averaged_parts = []
    for b in range(num_buckets):
        start = b * bucket_cap
        end = min(start + bucket_cap, num_params)
        bucket_slice = rank_grads[:, start:end]
        reduced_bucket = jnp.sum(bucket_slice, axis=0) / float(world_size)
        bucket_results.append(reduced_bucket)
        averaged_parts.append(reduced_bucket)

    averaged_grads = jnp.concatenate(averaged_parts, axis=0)
    return averaged_grads, bucket_results


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Ring AllReduce
    ar_sum = ring_allreduce_simulate(g["rank_inputs"])
    err = float(jnp.max(jnp.abs(ar_sum - g["allreduce_sum"])))
    assert err < 1e-5, f"allreduce_sum error {err:.2e}"

    # 2. DDP Bucketed AllReduce
    ddp_grads, buckets = ddp_bucketed_allreduce(g["rank_grads"], 8)
    err = float(jnp.max(jnp.abs(ddp_grads - g["ddp_grads_out"])))
    assert err < 1e-5, f"ddp_grads error {err:.2e}"
    err = float(jnp.max(jnp.abs(buckets[0] - g["bucket_1_reduced"])))
    assert err < 1e-5, f"bucket 1 error {err:.2e}"
    err = float(jnp.max(jnp.abs(buckets[1] - g["bucket_2_reduced"])))
    assert err < 1e-5, f"bucket 2 error {err:.2e}"

    print("16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
