"""KV Cache Paged Radix H2O Snapkv and Yoco — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco. Run ``python jax/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def paged_kv_gather_and_attend(
    query: jnp.ndarray,
    key_blocks: jnp.ndarray,
    value_blocks: jnp.ndarray,
    block_table: jnp.ndarray,
    context_len: int,
    block_size: int = 4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch, _, dim = query.shape
    scale = 1.0 / math.sqrt(dim)

    attn_outs = []
    attn_weights_list = []

    for b in range(batch):
        k_list = []
        v_list = []
        for t in range(context_len):
            block_idx = int(block_table[b, t // block_size])
            offset = t % block_size
            k_list.append(key_blocks[block_idx, offset])
            v_list.append(value_blocks[block_idx, offset])

        k_seq = jnp.stack(k_list)  # [context_len, dim]
        v_seq = jnp.stack(v_list)  # [context_len, dim]

        q_b = query[b]  # [1, dim]
        scores = (q_b @ k_seq.T) * scale  # [1, context_len]
        probs = jax.nn.softmax(scores, axis=-1)  # [1, context_len]
        out_b = probs @ v_seq  # [1, dim]

        attn_outs.append(out_b)
        attn_weights_list.append(probs)

    return jnp.stack(attn_outs), jnp.stack(attn_weights_list)


def h2o_evict_mask(
    cumulative_scores: jnp.ndarray,
    local_window: int = 4,
    heavy_budget: int = 4,
) -> jnp.ndarray:
    batch, seq_len = cumulative_scores.shape
    keep_mask = jnp.zeros_like(cumulative_scores)

    recent_start = max(0, seq_len - local_window)
    # Mark local window as 1.0
    keep_mask = keep_mask.at[:, recent_start:].set(1.0)

    cand_len = recent_start
    if cand_len > 0 and heavy_budget > 0:
        cand_scores = cumulative_scores[:, :cand_len]
        actual_budget = min(heavy_budget, cand_len)
        for b in range(batch):
            top_indices = jnp.argsort(cand_scores[b])[-actual_budget:]
            keep_mask = keep_mask.at[b, top_indices].set(1.0)

    return keep_mask


def snapkv_select_mask(
    prompt_attention: jnp.ndarray,
    obs_window: int = 4,
    budget: int = 6,
) -> jnp.ndarray:
    batch, _, prompt_len = prompt_attention.shape
    importance = jnp.mean(prompt_attention, axis=1)  # [batch, prompt_len]

    retain_mask = jnp.zeros_like(importance)
    k = min(budget, prompt_len)

    for b in range(batch):
        top_idx = jnp.argsort(importance[b])[-k:]
        retain_mask = retain_mask.at[b, top_idx].set(1.0)

    return retain_mask


def yoco_cross_layer_forward(
    h_mid: jnp.ndarray,
    k_shared: jnp.ndarray,
    v_shared: jnp.ndarray,
    w_q: jnp.ndarray,
    w_out: jnp.ndarray,
) -> jnp.ndarray:
    _, _, dim = h_mid.shape
    scale = 1.0 / math.sqrt(dim)

    q = h_mid @ w_q
    scores = (q @ jnp.swapaxes(k_shared, -1, -2)) * scale
    attn = jax.nn.softmax(scores, axis=-1)
    context = attn @ v_shared
    h_out = h_mid + context @ w_out
    return h_out


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Paged Attention
    p_out, p_weights = paged_kv_gather_and_attend(
        g["query"], g["key_blocks"], g["value_blocks"], g["block_table"], context_len=10, block_size=4
    )
    err = float(jnp.max(jnp.abs(p_out - g["paged_out"])))
    assert err < 1e-5, f"paged_out error {err:.2e}"
    err = float(jnp.max(jnp.abs(p_weights - g["paged_weights"])))
    assert err < 1e-5, f"paged_weights error {err:.2e}"

    # 3. H2O Cache Eviction
    h_mask = h2o_evict_mask(g["cumulative_scores"], local_window=4, heavy_budget=3)
    err = float(jnp.max(jnp.abs(h_mask - g["h2o_mask"])))
    assert err < 1e-5, f"h2o_mask error {err:.2e}"

    # 4. SnapKV
    s_mask = snapkv_select_mask(g["prompt_attn"], obs_window=4, budget=6)
    err = float(jnp.max(jnp.abs(s_mask - g["snapkv_mask"])))
    assert err < 1e-5, f"snapkv_mask error {err:.2e}"

    # 5. YOCO
    y_out = yoco_cross_layer_forward(g["h_mid"], g["k_shared"], g["v_shared"], g["w_q"], g["w_out"])
    err = float(jnp.max(jnp.abs(y_out - g["yoco_out"])))
    assert err < 1e-5, f"yoco_out error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
