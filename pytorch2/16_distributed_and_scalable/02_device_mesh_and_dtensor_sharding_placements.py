"""Device Mesh and DTensor Sharding Placements — PyTorch 2.x reference implementation.

Topic: 16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements
Golden weights: weights/16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements.safetensors

``python pytorch2/16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements.py`` verifies against the golden checkpoint;
``python pytorch2/16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dtensor_shard(tensor: torch.Tensor, mesh_size: int, shard_dim: int) -> torch.Tensor:
    """Shard a global tensor along shard_dim across mesh_size virtual ranks: Shard(dim).

    Args:
        tensor: Global tensor.
        mesh_size: Number of ranks in the mesh dimension.
        shard_dim: Dimension along which to partition.

    Returns:
        shards: Tensor of local shards of shape [mesh_size, ...local_shape...].
    """
    chunks = torch.chunk(tensor, mesh_size, dim=shard_dim)
    return torch.stack(chunks, dim=0)


def dtensor_allgather(shards: torch.Tensor, gather_dim: int) -> torch.Tensor:
    """AllGather local shards along gather_dim to reconstruct global tensor: Shard(dim) -> Replicate().

    Args:
        shards: Local shards tensor [mesh_size, ...local_shape...].
        gather_dim: Tensor dimension along which to concatenate.

    Returns:
        global_tensor: Reconstructed global tensor.
    """
    mesh_size = shards.shape[0]
    unstacked = [shards[r] for r in range(mesh_size)]
    return torch.cat(unstacked, dim=gather_dim)


def dtensor_column_parallel_matmul(x: torch.Tensor, w_shards: torch.Tensor) -> torch.Tensor:
    """Column-parallel matmul: x is Replicated, W is Sharded on columns (dim 1).

    Output is Sharded on columns: Y_shards[r] = x @ W_shards[r].
    """
    mesh_size = w_shards.shape[0]
    outs = [x @ w_shards[r] for r in range(mesh_size)]
    return torch.stack(outs, dim=0)


def dtensor_row_parallel_matmul(
    x_shards: torch.Tensor,
    w_shards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-parallel matmul: x is Sharded on cols (dim 1), W is Sharded on rows (dim 0).

    Local matmul yields Partial(Sum) shards: Y_local[r] = X_shards[r] @ W_shards[r].
    AllReduce across mesh produces Replicated output: Y = sum_r Y_local[r].
    """
    mesh_size = x_shards.shape[0]
    partials = [x_shards[r] @ w_shards[r] for r in range(mesh_size)]
    partial_shards = torch.stack(partials, dim=0)
    allreduced_output = torch.sum(partial_shards, dim=0)
    return allreduced_output, partial_shards


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    mesh_size = 2
    global_x = torch.randn(4, 8)
    w_matrix = torch.randn(8, 6)

    # 1. Shard & AllGather roundtrip
    x_sharded_dim0 = dtensor_shard(global_x, mesh_size=mesh_size, shard_dim=0)
    x_gathered = dtensor_allgather(x_sharded_dim0, gather_dim=0)

    # 2. Column-Parallel Matmul (W sharded on dim 1)
    w_col_shards = dtensor_shard(w_matrix, mesh_size=mesh_size, shard_dim=1)
    col_out_shards = dtensor_column_parallel_matmul(global_x, w_col_shards)
    col_out_gathered = dtensor_allgather(col_out_shards, gather_dim=1)

    # 3. Row-Parallel Matmul (X sharded on dim 1, W sharded on dim 0)
    x_row_shards = dtensor_shard(global_x, mesh_size=mesh_size, shard_dim=1)
    w_row_shards = dtensor_shard(w_matrix, mesh_size=mesh_size, shard_dim=0)
    row_out_allreduced, partial_shards = dtensor_row_parallel_matmul(x_row_shards, w_row_shards)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "global_x": global_x,
        "w_matrix": w_matrix,
        "x_sharded_dim0": x_sharded_dim0,
        "x_gathered": x_gathered,
        "w_col_shards": w_col_shards,
        "col_out_shards": col_out_shards,
        "col_out_gathered": col_out_gathered,
        "x_row_shards": x_row_shards,
        "w_row_shards": w_row_shards,
        "partial_shards": partial_shards,
        "row_out_allreduced": row_out_allreduced,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-5, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
