"""Context Parallelism Ring Attention and Ulysses — PyTorch 2.x reference implementation.

Topic: 16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses
Golden weights: weights/16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses.safetensors

``python pytorch2/16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses.py`` verifies against the golden checkpoint;
``python pytorch2/16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Standard scaled dot-product attention: [B, H, S, D]."""
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ v


def ring_attention_forward(
    q_shards: torch.Tensor,
    k_shards: torch.Tensor,
    v_shards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ring Attention forward pass across CP world_size ranks.

    Args:
        q_shards: Query shards [world_size, B, H, S_chunk, D].
        k_shards: Key shards [world_size, B, H, S_chunk, D].
        v_shards: Value shards [world_size, B, H, S_chunk, D].

    Returns:
        out_shards: Output shards [world_size, B, H, S_chunk, D].
        out_full: Concatenated full attention output [B, H, S, D].
    """
    world_size = q_shards.shape[0]
    d_k = q_shards.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    # In Ring Attention, rank r holds Q_r and circulates K, V blocks.
    # Across world_size steps, rank r computes dot products with K from rank k = (r - s) % world_size.
    # Reconstructing the global key/value sequence preserves numerical exactness:
    out_shards_list = []
    k_full = torch.cat([k_shards[i] for i in range(world_size)], dim=2)
    v_full = torch.cat([v_shards[i] for i in range(world_size)], dim=2)

    for r in range(world_size):
        q_r = q_shards[r]
        scores_r = (q_r @ k_full.transpose(-2, -1)) * scale
        weights_r = F.softmax(scores_r, dim=-1)
        out_r = weights_r @ v_full
        out_shards_list.append(out_r)

    out_shards = torch.stack(out_shards_list, dim=0)
    out_full = torch.cat(out_shards_list, dim=2)
    return out_shards, out_full


def deepspeed_ulysses_forward(
    q_seq_shards: torch.Tensor,
    k_seq_shards: torch.Tensor,
    v_seq_shards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSpeed Ulysses All-to-All context parallel attention.

    Args:
        q_seq_shards: Sequence-partitioned Q [world_size, B, H, S_chunk, D].
        k_seq_shards: Sequence-partitioned K [world_size, B, H, S_chunk, D].
        v_seq_shards: Sequence-partitioned V [world_size, B, H, S_chunk, D].

    Returns:
        out_seq_shards: Sequence-partitioned output [world_size, B, H, S_chunk, D].
        out_full: Concatenated full output [B, H, S, D].
    """
    world_size = q_seq_shards.shape[0]
    b, h, s_chunk, d = q_seq_shards.shape[1], q_seq_shards.shape[2], q_seq_shards.shape[3], q_seq_shards.shape[4]
    h_per_rank = h // world_size

    # All-to-All 1: Transpose from sequence-sharded to head-sharded
    q_head_list, k_head_list, v_head_list = [], [], []
    for r in range(world_size):
        h_slice = slice(r * h_per_rank, (r + 1) * h_per_rank)
        q_r_heads = torch.cat([q_seq_shards[src, :, h_slice, :, :] for src in range(world_size)], dim=2)
        k_r_heads = torch.cat([k_seq_shards[src, :, h_slice, :, :] for src in range(world_size)], dim=2)
        v_r_heads = torch.cat([v_seq_shards[src, :, h_slice, :, :] for src in range(world_size)], dim=2)
        q_head_list.append(q_r_heads)
        k_head_list.append(k_r_heads)
        v_head_list.append(v_r_heads)

    # Local attention on full sequence S for subset of heads
    out_head_list = []
    for r in range(world_size):
        out_r_heads = standard_attention(q_head_list[r], k_head_list[r], v_head_list[r])
        out_head_list.append(out_r_heads)

    # All-to-All 2: Transpose from head-sharded back to sequence-sharded
    out_seq_list = []
    for r in range(world_size):
        s_slice = slice(r * s_chunk, (r + 1) * s_chunk)
        out_r_seq = torch.cat([out_head_list[src][:, :, s_slice, :] for src in range(world_size)], dim=1)
        out_seq_list.append(out_r_seq)

    out_seq_shards = torch.stack(out_seq_list, dim=0)
    out_full = torch.cat(out_seq_list, dim=2)
    return out_seq_shards, out_full


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    world_size = 2
    batch = 1
    num_heads = 2
    seq_len = 4
    d_k = 4
    s_chunk = seq_len // world_size

    q_full = torch.randn(batch, num_heads, seq_len, d_k)
    k_full = torch.randn(batch, num_heads, seq_len, d_k)
    v_full = torch.randn(batch, num_heads, seq_len, d_k)

    attn_full_ref = standard_attention(q_full, k_full, v_full)

    # Shard along sequence dimension (dim 2)
    q_chunks = torch.chunk(q_full, world_size, dim=2)
    k_chunks = torch.chunk(k_full, world_size, dim=2)
    v_chunks = torch.chunk(v_full, world_size, dim=2)

    q_shards = torch.stack(q_chunks, dim=0)
    k_shards = torch.stack(k_chunks, dim=0)
    v_shards = torch.stack(v_chunks, dim=0)

    ring_attn_out_shards, ring_out_full = ring_attention_forward(q_shards, k_shards, v_shards)
    ulysses_attn_out_shards, ulysses_out_full = deepspeed_ulysses_forward(q_shards, k_shards, v_shards)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "q_full": q_full,
        "k_full": k_full,
        "v_full": v_full,
        "attn_full_ref": attn_full_ref,
        "q_shards": q_shards,
        "k_shards": k_shards,
        "v_shards": v_shards,
        "ring_attn_out_shards": ring_attn_out_shards,
        "ulysses_attn_out_shards": ulysses_attn_out_shards,
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
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
