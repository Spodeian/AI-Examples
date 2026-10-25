"""Attention Variants: MHA, MQA, GQA, MLA, and DiffAttn — PyTorch 2.x reference implementation.

Topic: 05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn
Golden weights: weights/05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn.safetensors

``python pytorch2/05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def grouped_query_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped-Query Attention (GQA).
    
    q: [B, H, T, D]
    k: [B, G, S, D] (G groups, where H % G == 0)
    v: [B, G, S, D]
    
    Repeats k and v so each group of query heads attends to its assigned KV head.
    """
    B, H, T, D = q.shape
    G = k.shape[1]
    S = k.shape[2]
    group_size = H // G

    # Expand k and v: [B, G, 1, S, D] -> [B, G, group_size, S, D] -> [B, H, S, D]
    k_exp = k.unsqueeze(2).expand(B, G, group_size, S, D).reshape(B, H, S, D)
    v_exp = v.unsqueeze(2).expand(B, G, group_size, S, D).reshape(B, H, S, D)

    # Scaled dot-product: [B, H, T, D] @ [B, H, D, S] -> [B, H, T, S]
    scores = torch.matmul(q, k_exp.transpose(-1, -2)) * scale
    attn_weights = F.softmax(scores, dim=-1)

    out = torch.matmul(attn_weights, v_exp)  # [B, H, T, D]
    return out, attn_weights


def differential_attention(
    q1: torch.Tensor,
    k1: torch.Tensor,
    q2: torch.Tensor,
    k2: torch.Tensor,
    v: torch.Tensor,
    lambda_param: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differential Attention (DiffAttn).
    
    DiffAttn(Q, K, V) = (softmax(Q1 K1^T * scale) - lambda * softmax(Q2 K2^T * scale)) V
    Cancels noise and sharpens attention over relevant context.
    """
    scores1 = torch.matmul(q1, k1.transpose(-1, -2)) * scale
    scores2 = torch.matmul(q2, k2.transpose(-1, -2)) * scale

    a1 = F.softmax(scores1, dim=-1)
    a2 = F.softmax(scores2, dim=-1)

    diff_weights = a1 - lambda_param * a2
    out = torch.matmul(diff_weights, v)
    return out, diff_weights


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

    # 2. Grouped-Query Attention (H=4, G=2, group_size=2)
    B = 2
    H = 4
    G = 2
    T = 4
    S = 4
    D = 4
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, T, D) * 0.3
    k = torch.randn(B, G, S, D) * 0.3
    v = torch.randn(B, G, S, D) * 0.3
    scale_tensor = torch.tensor([scale])

    gqa_out, gqa_weights = grouped_query_attention(q, k, v, scale)

    # 3. Differential Attention
    lambda_val = 0.5
    lambda_tensor = torch.tensor([lambda_val])
    q1 = torch.randn(B, H, T, D) * 0.3
    k1 = torch.randn(B, H, S, D) * 0.3
    q2 = torch.randn(B, H, T, D) * 0.3
    k2 = torch.randn(B, H, S, D) * 0.3
    v_diff = torch.randn(B, H, S, D) * 0.3

    diff_out, diff_weights = differential_attention(q1, k1, q2, k2, v_diff, lambda_val, scale)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "q": q,
        "k": k,
        "v": v,
        "scale": scale_tensor,
        "gqa_out": gqa_out,
        "gqa_weights": gqa_weights,
        "lambda_val": lambda_tensor,
        "q1": q1,
        "k1": k1,
        "q2": q2,
        "k2": k2,
        "v_diff": v_diff,
        "diff_out": diff_out,
        "diff_weights": diff_weights,
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
