"""Production Serving Dynamic Batcher PD Disaggregation and Streaming — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming
Golden weights: weights/17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def prefill_step(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    bo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill worker: compute-bound full causal self-attention over prompt sequence.
    
    Returns:
      y: [B, S, D] output activations
      k: [B, S, D] key cache
      v: [B, S, D] value cache
    """
    b, s, d = x.shape
    q = x @ wq
    k = x @ wk
    v = x @ wv

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    # Causal lower-triangular mask
    mask = torch.triu(torch.full((s, s), -10000.0, device=x.device), diagonal=1)
    attn = F.softmax(scores + mask, dim=-1)
    context = attn @ v
    y = context @ wo + bo
    return y, k, v


def decode_step(
    x_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    bo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode worker: memory-bound incremental step consuming KV-cache from prefill worker.
    
    Returns:
      y_token: [B, 1, D] next-token output activation
      new_k: [B, S+1, D] updated key cache
      new_v: [B, S+1, D] updated value cache
    """
    b, _, d = x_new.shape
    q_new = x_new @ wq
    k_new = x_new @ wk
    v_new = x_new @ wv

    new_k = torch.cat([k_cache, k_new], dim=1)
    new_v = torch.cat([v_cache, v_new], dim=1)

    scores = (q_new @ new_k.transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(scores, dim=-1)
    context = attn @ new_v
    y_token = context @ wo + bo
    return y_token, new_k, new_v


def schedule_dynamic_batch(seq_lengths: list[int], max_tokens_per_batch: int = 16) -> list[list[int]]:
    """Partitions incoming requests into chunked micro-batches adhering to token budget."""
    batches = []
    current_batch = []
    current_tokens = 0
    for idx, length in enumerate(seq_lengths):
        if current_tokens + length > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [idx]
            current_tokens = length
        else:
            current_batch.append(idx)
            current_tokens += length
    if current_batch:
        batches.append(current_batch)
    return batches


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    b, s, d = 2, 4, 4

    x_prefill = torch.randn(b, s, d)
    x_decode = torch.randn(b, 1, d)

    wq = torch.randn(d, d) * 0.2
    wk = torch.randn(d, d) * 0.2
    wv = torch.randn(d, d) * 0.2
    wo = torch.randn(d, d) * 0.2
    bo = torch.randn(d) * 0.1

    # Prefill step
    y_prefill, k_cache, v_cache = prefill_step(x_prefill, wq, wk, wv, wo, bo)

    # Decode step (simulating PD-disaggregation transfer of KV cache)
    y_decode, new_k, new_v = decode_step(x_decode, k_cache, v_cache, wq, wk, wv, wo, bo)

    # Full causal forward pass over combined sequence [x_prefill, x_decode]
    x_full = torch.cat([x_prefill, x_decode], dim=1)
    y_full, _, _ = prefill_step(x_full, wq, wk, wv, wo, bo)
    # Mathematical equivalence check: y_decode must equal y_full[:, -1:, :]
    assert torch.max(torch.abs(y_decode - y_full[:, -1:, :])) < 1e-5

    linear_w = torch.randn(d, d) * 0.1
    linear_b = torch.randn(d) * 0.1
    anchor_in = y_decode.squeeze(1)
    test_out = linear_anchor(anchor_in, linear_w, linear_b)

    tensors = {
        "x_prefill": x_prefill,
        "x_decode": x_decode,
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "bo": bo,
        "y_prefill": y_prefill,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "y_decode": y_decode,
        "new_k": new_k,
        "new_v": new_v,
        "y_full": y_full,
        "linear_w": linear_w,
        "linear_b": linear_b,
        "test_input": anchor_in.clone(),
        "weight": linear_w.clone(),
        "bias": linear_b.clone(),
        "test_output": test_out,
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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
