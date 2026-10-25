"""Device Mesh and DTensor Sharding Placements — functional JAX implementation.

Topic: 16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements. Run ``python jax/16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def dtensor_shard(tensor: jnp.ndarray, mesh_size: int, shard_dim: int) -> jnp.ndarray:
    """Shard a global tensor along shard_dim across mesh_size virtual ranks: Shard(dim)."""
    chunks = jnp.split(tensor, mesh_size, axis=shard_dim)
    return jnp.stack(chunks, axis=0)


def dtensor_allgather(shards: jnp.ndarray, gather_dim: int) -> jnp.ndarray:
    """AllGather local shards along gather_dim: Shard(dim) -> Replicate()."""
    mesh_size = shards.shape[0]
    unstacked = [shards[r] for r in range(mesh_size)]
    return jnp.concatenate(unstacked, axis=gather_dim)


def dtensor_column_parallel_matmul(x: jnp.ndarray, w_shards: jnp.ndarray) -> jnp.ndarray:
    """Column-parallel matmul."""
    mesh_size = w_shards.shape[0]
    outs = [x @ w_shards[r] for r in range(mesh_size)]
    return jnp.stack(outs, axis=0)


def dtensor_row_parallel_matmul(
    x_shards: jnp.ndarray,
    w_shards: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Row-parallel matmul."""
    mesh_size = x_shards.shape[0]
    partials = [x_shards[r] @ w_shards[r] for r in range(mesh_size)]
    partial_shards = jnp.stack(partials, axis=0)
    allreduced_output = jnp.sum(partial_shards, axis=0)
    return allreduced_output, partial_shards


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Shard & Gather roundtrip
    x_sharded = dtensor_shard(g["global_x"], 2, 0)
    err = float(jnp.max(jnp.abs(x_sharded - g["x_sharded_dim0"])))
    assert err < 1e-5, f"x_sharded error {err:.2e}"
    x_gathered = dtensor_allgather(x_sharded, 0)
    err = float(jnp.max(jnp.abs(x_gathered - g["x_gathered"])))
    assert err < 1e-5, f"x_gathered error {err:.2e}"

    # 2. Column-parallel matmul
    col_out = dtensor_column_parallel_matmul(g["global_x"], g["w_col_shards"])
    err = float(jnp.max(jnp.abs(col_out - g["col_out_shards"])))
    assert err < 1e-5, f"col_out error {err:.2e}"
    col_gathered = dtensor_allgather(col_out, 1)
    err = float(jnp.max(jnp.abs(col_gathered - g["col_out_gathered"])))
    assert err < 1e-5, f"col_gathered error {err:.2e}"

    # 3. Row-parallel matmul
    row_allred, partials = dtensor_row_parallel_matmul(g["x_row_shards"], g["w_row_shards"])
    err = float(jnp.max(jnp.abs(partials - g["partial_shards"])))
    assert err < 1e-5, f"partial_shards error {err:.2e}"
    err = float(jnp.max(jnp.abs(row_allred - g["row_out_allreduced"])))
    assert err < 1e-5, f"row_allred error {err:.2e}"

    print("16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
