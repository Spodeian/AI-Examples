"""Collectives and DDP Gradient Bucketing Mechanics — PyTorch 2.x reference implementation.

Topic: 16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics
Golden weights: weights/16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics.safetensors

``python pytorch2/16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics.py`` verifies against the golden checkpoint;
``python pytorch2/16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ring_allreduce_simulate(rank_data: torch.Tensor) -> torch.Tensor:
    """Simulate 2*(W-1) Ring AllReduce collective across simulated ranks.

    Performs Scatter-Reduce followed by AllGather on rank_data [world_size, total_elements].

    Args:
        rank_data: Tensor of shape [world_size, total_elements].

    Returns:
        reduced: Element-wise sum across all ranks of shape [total_elements].
    """
    world_size, total_elements = rank_data.shape
    assert total_elements % world_size == 0, "total_elements must be divisible by world_size"
    chunk_size = total_elements // world_size

    # Split into chunks: [world_size, world_size, chunk_size]
    # chunks[r, c] is chunk c of rank r
    chunks = rank_data.view(world_size, world_size, chunk_size).clone()

    # Phase 1: Scatter-Reduce (W - 1 steps)
    for step in range(world_size - 1):
        send_chunks = chunks.clone()
        for r in range(world_size):
            send_chunk_idx = (r - step) % world_size
            recv_chunk_idx = (r - step - 1) % world_size
            prev_rank = (r - 1) % world_size
            # Rank r receives from prev_rank and accumulates
            chunks[r, recv_chunk_idx] += send_chunks[prev_rank, recv_chunk_idx]

    # Phase 2: AllGather (W - 1 steps)
    for step in range(world_size - 1):
        send_chunks = chunks.clone()
        for r in range(world_size):
            send_chunk_idx = (r - step + 1) % world_size
            recv_chunk_idx = (r - step) % world_size
            prev_rank = (r - 1) % world_size
            # Rank r receives fully reduced chunk from prev_rank
            chunks[r, recv_chunk_idx] = send_chunks[prev_rank, recv_chunk_idx]

    # All ranks now have the exact same reduced vector
    reduced = chunks[0].view(total_elements)
    return reduced


def ddp_bucketed_allreduce(
    rank_grads: torch.Tensor,
    bucket_cap: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """DDP gradient bucketing with asynchronous bucket-level reduction simulation.

    Args:
        rank_grads: Flattened gradients across ranks [world_size, num_params].
        bucket_cap: Capacity in number of elements per bucket.

    Returns:
        averaged_grads: AllReduced and averaged gradients [num_params].
        bucket_results: List of reduced bucket tensors.
    """
    world_size, num_params = rank_grads.shape
    num_buckets = (num_params + bucket_cap - 1) // bucket_cap
    bucket_results = []

    averaged_parts = []
    for b in range(num_buckets):
        start = b * bucket_cap
        end = min(start + bucket_cap, num_params)
        bucket_slice = rank_grads[:, start:end]
        # Sum across ranks and divide by world_size
        reduced_bucket = torch.sum(bucket_slice, dim=0) / float(world_size)
        bucket_results.append(reduced_bucket)
        averaged_parts.append(reduced_bucket)

    averaged_grads = torch.cat(averaged_parts, dim=0)
    return averaged_grads, bucket_results


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    world_size = 4
    total_elements = 8  # divisible by 4 (chunk_size = 2)
    rank_inputs = torch.randn(world_size, total_elements)

    # 1. Ring AllReduce
    allreduce_sum = ring_allreduce_simulate(rank_inputs)

    # 2. DDP Gradient Bucketing
    num_params = 16
    bucket_cap = 8
    rank_grads = torch.randn(world_size, num_params)
    ddp_grads_out, buckets = ddp_bucketed_allreduce(rank_grads, bucket_cap)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "rank_inputs": rank_inputs,
        "allreduce_sum": allreduce_sum,
        "rank_grads": rank_grads,
        "bucket_1_reduced": buckets[0],
        "bucket_2_reduced": buckets[1],
        "ddp_grads_out": ddp_grads_out,
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
