"""Supervised Finetuning Chat Masks Neftune and Sequence Packing — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.
Run ``python jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def neftune_add_noise(
    embeddings: jnp.ndarray,
    noise: jnp.ndarray,
    alpha: float = 5.0,
) -> jnp.ndarray:
    b, seq_len, d_dim = embeddings.shape
    scale = alpha / math.sqrt(seq_len * d_dim)
    return embeddings + scale * noise


def masked_completion_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    loss_mask: jnp.ndarray,
    eps: float = 1e-9,
) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    b, l, v = logits.shape
    batch_idx = jnp.arange(b)[:, None]
    seq_idx = jnp.arange(l)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, targets]
    nll = -target_log_probs
    return jnp.sum(nll * loss_mask) / (jnp.sum(loss_mask) + eps)


def packed_block_diagonal_causal_mask(
    segment_ids: jnp.ndarray,
) -> jnp.ndarray:
    seq_len = segment_ids.shape[0]
    idx = jnp.arange(seq_len)
    causal = idx[None, :] <= idx[:, None]
    same_seg = segment_ids[None, :] == segment_ids[:, None]
    return (causal & same_seg).astype(jnp.float32)


def packed_position_ids(
    segment_ids: jnp.ndarray,
) -> jnp.ndarray:
    seq_len = int(segment_ids.shape[0])
    seg_list = [int(x) for x in segment_ids]
    pos_ids = []
    cur = 0
    for i in range(seq_len):
        if i > 0 and seg_list[i] != seg_list[i - 1]:
            cur = 0
        pos_ids.append(cur)
        cur += 1
    return jnp.array(pos_ids, dtype=jnp.int64)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. NEFTune Add Noise
    noisy = neftune_add_noise(g["embeddings"], g["noise"], alpha=5.0)
    err_nef = float(jnp.max(jnp.abs(noisy - g["neftune_noisy_ref"])))
    assert err_nef < 1e-5, f"neftune error {err_nef:.2e}"

    # 3. Masked Completion Cross-Entropy
    loss = masked_completion_cross_entropy(g["logits"], g["targets"], g["loss_mask"])
    err_loss = float(jnp.abs(loss - g["masked_loss_ref"]))
    assert err_loss < 1e-5, f"masked loss error {err_loss:.2e}"

    # 4. Packed Block-Diagonal Causal Mask
    mask = packed_block_diagonal_causal_mask(g["segment_ids"])
    err_mask = float(jnp.max(jnp.abs(mask - g["packed_mask_ref"])))
    assert err_mask < 1e-5, f"packed mask error {err_mask:.2e}"

    # 5. Packed Position IDs
    pos = packed_position_ids(g["segment_ids"])
    err_pos = float(jnp.max(jnp.abs(pos - g["packed_positions_ref"])))
    assert err_pos < 1e-5, f"packed position error {err_pos:.2e}"

    print("08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
