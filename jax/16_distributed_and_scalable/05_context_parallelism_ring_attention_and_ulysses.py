"""Context Parallelism Ring Attention and Ulysses — functional JAX implementation.

Topic: 16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses.
Run ``python jax/16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def standard_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Standard scaled dot-product attention in JAX: [B, H, S, D]."""
    d_k = q.shape[-1]
    scores = (q @ jnp.swapaxes(k, -2, -1)) / math.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v


def ring_attention_forward(
    q_shards: jnp.ndarray,
    k_shards: jnp.ndarray,
    v_shards: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ring Attention forward pass across CP world_size ranks in JAX."""
    world_size = q_shards.shape[0]
    d_k = q_shards.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    k_full = jnp.concatenate([k_shards[i] for i in range(world_size)], axis=2)
    v_full = jnp.concatenate([v_shards[i] for i in range(world_size)], axis=2)

    out_shards_list = []
    for r in range(world_size):
        q_r = q_shards[r]
        scores_r = (q_r @ jnp.swapaxes(k_full, -2, -1)) * scale
        weights_r = jax.nn.softmax(scores_r, axis=-1)
        out_r = weights_r @ v_full
        out_shards_list.append(out_r)

    out_shards = jnp.stack(out_shards_list, axis=0)
    out_full = jnp.concatenate(out_shards_list, axis=2)
    return out_shards, out_full


def deepspeed_ulysses_forward(
    q_seq_shards: jnp.ndarray,
    k_seq_shards: jnp.ndarray,
    v_seq_shards: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """DeepSpeed Ulysses All-to-All context parallel attention in JAX."""
    world_size = q_seq_shards.shape[0]
    b, h, s_chunk, d = q_seq_shards.shape[1], q_seq_shards.shape[2], q_seq_shards.shape[3], q_seq_shards.shape[4]
    h_per_rank = h // world_size

    # All-to-All 1: sequence-sharded to head-sharded
    q_head_list, k_head_list, v_head_list = [], [], []
    for r in range(world_size):
        h_start = r * h_per_rank
        h_end = (r + 1) * h_per_rank
        q_r_heads = jnp.concatenate([q_seq_shards[src, :, h_start:h_end, :, :] for src in range(world_size)], axis=2)
        k_r_heads = jnp.concatenate([k_seq_shards[src, :, h_start:h_end, :, :] for src in range(world_size)], axis=2)
        v_r_heads = jnp.concatenate([v_seq_shards[src, :, h_start:h_end, :, :] for src in range(world_size)], axis=2)
        q_head_list.append(q_r_heads)
        k_head_list.append(k_r_heads)
        v_head_list.append(v_r_heads)

    # Local attention on full sequence S for subset of heads
    out_head_list = []
    for r in range(world_size):
        out_r_heads = standard_attention(q_head_list[r], k_head_list[r], v_head_list[r])
        out_head_list.append(out_r_heads)

    # All-to-All 2: head-sharded back to sequence-sharded
    out_seq_list = []
    for r in range(world_size):
        s_start = r * s_chunk
        s_end = (r + 1) * s_chunk
        out_r_seq = jnp.concatenate([out_head_list[src][:, :, s_start:s_end, :] for src in range(world_size)], axis=1)
        out_seq_list.append(out_r_seq)

    out_seq_shards = jnp.stack(out_seq_list, axis=0)
    out_full = jnp.concatenate(out_seq_list, axis=2)
    return out_seq_shards, out_full


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Standard attention reference
    ref = standard_attention(g["q_full"], g["k_full"], g["v_full"])
    err = float(jnp.max(jnp.abs(ref - g["attn_full_ref"])))
    assert err < 1e-5, f"standard attention error {err:.2e}"

    # Ring Attention
    ring_shards, _ = ring_attention_forward(g["q_shards"], g["k_shards"], g["v_shards"])
    err = float(jnp.max(jnp.abs(ring_shards - g["ring_attn_out_shards"])))
    assert err < 1e-5, f"ring attention error {err:.2e}"

    # DeepSpeed Ulysses
    ulysses_shards, _ = deepspeed_ulysses_forward(g["q_shards"], g["k_shards"], g["v_shards"])
    err = float(jnp.max(jnp.abs(ulysses_shards - g["ulysses_attn_out_shards"])))
    assert err < 1e-5, f"deepspeed ulysses error {err:.2e}"

    print("16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
