"""KV Cache Paged Radix H2O Snapkv and Yoco — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco
Golden weights: weights/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def paged_kv_gather_and_attend(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather non-contiguous KV cache pages and compute scaled dot-product attention.

    Args:
        query: Shape [batch, 1, dim].
        key_blocks: Shape [num_blocks, block_size, dim].
        value_blocks: Shape [num_blocks, block_size, dim].
        block_table: Logical to physical block mapping of shape [batch, max_blocks].
        context_len: Number of valid tokens in context.
        block_size: Number of tokens per physical memory page block.

    Returns:
        attn_out: Attention output of shape [batch, 1, dim].
        attn_weights: Attention distribution of shape [batch, 1, context_len].
    """
    batch, _, dim = query.shape
    scale = 1.0 / math.sqrt(dim)

    attn_outs = []
    attn_weights_list = []

    for b in range(batch):
        # Gather keys and values for context_len tokens
        k_list = []
        v_list = []
        for t in range(context_len):
            block_idx = block_table[b, t // block_size].item()
            offset = t % block_size
            k_list.append(key_blocks[block_idx, offset])
            v_list.append(value_blocks[block_idx, offset])

        k_seq = torch.stack(k_list)  # [context_len, dim]
        v_seq = torch.stack(v_list)  # [context_len, dim]

        # Query attention against gathered KV
        q_b = query[b]  # [1, dim]
        scores = (q_b @ k_seq.transpose(-1, -2)) * scale  # [1, context_len]
        probs = F.softmax(scores, dim=-1)  # [1, context_len]
        out_b = probs @ v_seq  # [1, dim]

        attn_outs.append(out_b)
        attn_weights_list.append(probs)

    return torch.stack(attn_outs), torch.stack(attn_weights_list)


def h2o_evict_mask(
    cumulative_scores: torch.Tensor,
    local_window: int = 4,
    heavy_budget: int = 4,
) -> torch.Tensor:
    """Heavy Hitter Oracle (H2O) cache retention mask computation.

    Unconditionally retains the most recent `local_window` tokens, and allocates
    `heavy_budget` slots to the historical tokens with highest cumulative attention scores.

    Args:
        cumulative_scores: Accumulated attention scores of shape [batch, seq_len].
        local_window: Number of recent tokens to unconditionally preserve.
        heavy_budget: Number of heavy hitter historical tokens to retain.

    Returns:
        keep_mask: Binary mask of shape [batch, seq_len] (1 = keep, 0 = evict).
    """
    batch, seq_len = cumulative_scores.shape
    keep_mask = torch.zeros_like(cumulative_scores)

    # Local recent tokens unconditionally kept
    recent_start = max(0, seq_len - local_window)
    keep_mask[:, recent_start:] = 1.0

    # Historical candidate window
    cand_len = recent_start
    if cand_len > 0 and heavy_budget > 0:
        cand_scores = cumulative_scores[:, :cand_len]
        actual_budget = min(heavy_budget, cand_len)
        _, top_indices = torch.topk(cand_scores, k=actual_budget, dim=-1, largest=True)
        for b in range(batch):
            keep_mask[b, top_indices[b]] = 1.0

    return keep_mask


def snapkv_select_mask(
    prompt_attention: torch.Tensor,
    obs_window: int = 4,
    budget: int = 6,
) -> torch.Tensor:
    """SnapKV: Observation window pooling for prompt KV cache compression.

    Pools attention weights emitted by the last `obs_window` prompt tokens across preceding positions,
    selecting the top `budget` most salient prefix tokens.

    Args:
        prompt_attention: Attention matrix of shape [batch, obs_window, prompt_len].
        obs_window: Size of observation suffix window.
        budget: Maximum number of prompt tokens to preserve.

    Returns:
        retain_mask: Binary selection mask of shape [batch, prompt_len].
    """
    batch, _, prompt_len = prompt_attention.shape
    importance = torch.mean(prompt_attention, dim=1)  # [batch, prompt_len]

    retain_mask = torch.zeros_like(importance)
    k = min(budget, prompt_len)
    _, top_idx = torch.topk(importance, k=k, dim=-1, largest=True)

    for b in range(batch):
        retain_mask[b, top_idx[b]] = 1.0

    return retain_mask


def yoco_cross_layer_forward(
    h_mid: torch.Tensor,
    k_shared: torch.Tensor,
    v_shared: torch.Tensor,
    w_q: torch.Tensor,
    w_out: torch.Tensor,
) -> torch.Tensor:
    """You Only Cache Once (YOCO) cross-decoder forward step.

    Queries a single global KV cache generated at the midpoint across subsequent decoder layers.

    Args:
        h_mid: Hidden state representation of shape [batch, seq_len, dim].
        k_shared: Shared global key cache of shape [batch, seq_len, dim].
        v_shared: Shared global value cache of shape [batch, seq_len, dim].
        w_q: Query projection weights of shape [dim, dim].
        w_out: Output projection weights of shape [dim, dim].

    Returns:
        h_out: Layer updated representation of shape [batch, seq_len, dim].
    """
    _, _, dim = h_mid.shape
    scale = 1.0 / math.sqrt(dim)

    q = h_mid @ w_q
    scores = (q @ k_shared.transpose(-1, -2)) * scale
    attn = F.softmax(scores, dim=-1)
    context = attn @ v_shared
    h_out = h_mid + context @ w_out
    return h_out


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. PagedAttention
    batch = 2
    dim = 8
    block_size = 4
    num_blocks = 8
    context_len = 10
    query = torch.randn(batch, 1, dim)
    key_blocks = torch.randn(num_blocks, block_size, dim) * 0.3
    value_blocks = torch.randn(num_blocks, block_size, dim) * 0.3
    block_table = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.int64)

    paged_out, paged_weights = paged_kv_gather_and_attend(
        query, key_blocks, value_blocks, block_table, context_len, block_size
    )

    # 2. H2O Cache Eviction
    # 2 batch items, seq_len 12, local_window 4, heavy_budget 3 -> keep 7 tokens, evict 5
    cumulative_scores = torch.rand(batch, 12)
    h2o_mask = h2o_evict_mask(cumulative_scores, local_window=4, heavy_budget=3)

    # 3. SnapKV Observation Window Compression
    # Observation window of 4 tokens looking at 16 prompt tokens, budget 6
    prompt_attn = F.softmax(torch.randn(batch, 4, 16), dim=-1)
    snapkv_mask = snapkv_select_mask(prompt_attn, obs_window=4, budget=6)

    # 4. YOCO Global KV Sharing
    h_mid = torch.randn(batch, 8, dim)
    k_shared = torch.randn(batch, 8, dim) * 0.2
    v_shared = torch.randn(batch, 8, dim) * 0.2
    w_q = torch.randn(dim, dim) * 0.15
    w_out = torch.randn(dim, dim) * 0.15
    yoco_out = yoco_cross_layer_forward(h_mid, k_shared, v_shared, w_q, w_out)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "query": query,
        "key_blocks": key_blocks,
        "value_blocks": value_blocks,
        "block_table": block_table.to(torch.float32),  # store as float32 in safetensors
        "paged_out": paged_out,
        "paged_weights": paged_weights,
        "cumulative_scores": cumulative_scores,
        "h2o_mask": h2o_mask,
        "prompt_attn": prompt_attn,
        "snapkv_mask": snapkv_mask,
        "h_mid": h_mid,
        "k_shared": k_shared,
        "v_shared": v_shared,
        "w_q": w_q,
        "w_out": w_out,
        "yoco_out": yoco_out,
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
