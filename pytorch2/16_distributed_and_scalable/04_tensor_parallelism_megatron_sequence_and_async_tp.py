"""Tensor Parallelism Megatron Sequence and Async TP — PyTorch 2.x reference implementation.

Topic: 16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp
Golden weights: weights/16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp.safetensors

``python pytorch2/16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp.py`` verifies against the golden checkpoint;
``python pytorch2/16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def megatron_tp_mlp(
    x: torch.Tensor,
    w1_shards: torch.Tensor,
    w2_shards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Megatron-LM 1D Tensor Parallel MLP block with Column-Parallel + Row-Parallel composition.

    Args:
        x: Replicated input activations [batch, seq_len, d_model].
        w1_shards: Column-parallel up-projection weights [world_size, d_model, d_ffn // world_size].
        w2_shards: Row-parallel down-projection weights [world_size, d_ffn // world_size, d_model].

    Returns:
        y: AllReduced output [batch, seq_len, d_model].
        partial_shards: Pre-AllReduce local outputs [world_size, batch, seq_len, d_model].
    """
    world_size = w1_shards.shape[0]
    partials = []
    for r in range(world_size):
        h_r = F.gelu(x @ w1_shards[r])
        y_r = h_r @ w2_shards[r]
        partials.append(y_r)

    partial_shards = torch.stack(partials, dim=0)
    y = torch.sum(partial_shards, dim=0)
    return y, partial_shards


def sequence_parallel_mlp(
    x_seq_shards: torch.Tensor,
    w1_shards: torch.Tensor,
    w2_shards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Megatron Sequence Parallelism: AllGather input along sequence dim, compute TP, ReduceScatter output.

    Args:
        x_seq_shards: Input sharded along sequence dim [world_size, batch, seq_len // world_size, d_model].
        w1_shards: Column-parallel weights [world_size, d_model, d_ffn // world_size].
        w2_shards: Row-parallel weights [world_size, d_ffn // world_size, d_model].

    Returns:
        y_seq_shards: Output sharded along sequence dim [world_size, batch, seq_len // world_size, d_model].
        x_full: Reconstructed full activation tensor [batch, seq_len, d_model].
    """
    world_size = x_seq_shards.shape[0]
    # Step 1: AllGather input along sequence dimension (dim 1)
    x_full = torch.cat([x_seq_shards[r] for r in range(world_size)], dim=1)

    # Step 2: Forward through Tensor Parallel MLP
    y_full, _ = megatron_tp_mlp(x_full, w1_shards, w2_shards)

    # Step 3: ReduceScatter along sequence dimension (dim 1)
    chunks = torch.chunk(y_full, world_size, dim=1)
    y_seq_shards = torch.stack(chunks, dim=0)
    return y_seq_shards, x_full


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    world_size = 2
    batch = 2
    seq_len = 4
    d_model = 8
    d_ffn = 12

    x = torch.randn(batch, seq_len, d_model)
    w1_shards = torch.randn(world_size, d_model, d_ffn // world_size) * 0.25
    w2_shards = torch.randn(world_size, d_ffn // world_size, d_model) * 0.25

    # 1. Megatron TP MLP
    tp_mlp_out, partial_shards = megatron_tp_mlp(x, w1_shards, w2_shards)

    # 2. Sequence Parallelism MLP
    chunks = torch.chunk(x, world_size, dim=1)
    x_seq_shards = torch.stack(chunks, dim=0)  # [2, 2, 2, 8]
    sp_mlp_out_shards, x_full = sequence_parallel_mlp(x_seq_shards, w1_shards, w2_shards)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "w1_shards": w1_shards,
        "w2_shards": w2_shards,
        "tp_mlp_out": tp_mlp_out,
        "partial_shards": partial_shards,
        "x_seq_shards": x_seq_shards,
        "sp_mlp_out_shards": sp_mlp_out_shards,
        "x_full": x_full,
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
