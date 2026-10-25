"""Sparse, Sliding Window, and Sink Attention — PyTorch 2.x reference implementation.

Topic: 05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention
Golden weights: weights/05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention.safetensors

``python pytorch2/05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def streaming_sink_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_sinks: int,
    window_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """StreamingLLM Attention with Attention Sinks and Sliding Window.
    
    q: [B, T, D]
    k: [B, S, D]
    v: [B, S, D]
    
    Allowed attention targets for token i:
    - Initial attention sinks: j < num_sinks
    - Local sliding window: i - j < window_size
    - Causal constraint: j <= i
    """
    B, T, D = q.shape
    S = k.shape[1]

    # Compute pairwise scores: [B, T, S]
    scores = torch.bmm(q, k.transpose(1, 2)) * scale

    # Construct sink + sliding window mask
    i_idx = torch.arange(T, device=q.device).unsqueeze(1)  # [T, 1]
    j_idx = torch.arange(S, device=q.device).unsqueeze(0)  # [1, S]

    is_sink = j_idx < num_sinks
    is_window = (i_idx - j_idx < window_size) & (i_idx >= j_idx)
    valid_mask = (is_sink | is_window) & (j_idx <= i_idx)

    mask_value = -1e9
    scores_masked = torch.where(valid_mask.unsqueeze(0), scores, torch.full_like(scores, mask_value))
    attn_weights = F.softmax(scores_masked, dim=-1)

    out = torch.bmm(attn_weights, v)
    return out, attn_weights


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    batch = 8
    in_dim = 4
    out_dim = 4
    test_in = torch.randn(batch, in_dim)
    w_linear = torch.randn(in_dim, out_dim) * 0.1
    b_linear = torch.randn(out_dim) * 0.1
    test_out = linear_anchor(test_in, w_linear, b_linear)

    # 2. Streaming Sink Attention
    B = 2
    T = 6
    S = 6
    D = 4
    num_sinks = 2
    window_size = 2
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, T, D) * 0.3
    k = torch.randn(B, S, D) * 0.3
    v = torch.randn(B, S, D) * 0.3
    num_sinks_tensor = torch.tensor([num_sinks])
    window_size_tensor = torch.tensor([window_size])
    scale_tensor = torch.tensor([scale])

    sink_out, sink_weights = streaming_sink_attention(q, k, v, num_sinks, window_size, scale)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "q": q,
        "k": k,
        "v": v,
        "num_sinks": num_sinks_tensor,
        "window_size": window_size_tensor,
        "scale": scale_tensor,
        "sink_out": sink_out,
        "sink_weights": sink_weights,
    }


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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
