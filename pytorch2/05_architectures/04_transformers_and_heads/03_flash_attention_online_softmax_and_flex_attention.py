"""Flash Attention, Online Softmax, and Flex Attention — PyTorch 2.x reference implementation.

Topic: 05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention
Golden weights: weights/05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention.safetensors

``python pytorch2/05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def online_softmax_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tiled FlashAttention algorithm with Online Softmax accumulation.
    
    q: [B, T, D]
    k: [B, S, D]
    v: [B, S, D]
    block_size: chunk size along sequence S
    
    Returns (out, final_max, final_sum).
    """
    B, T, D = q.shape
    S = k.shape[1]

    # Initialize running accumulators
    out = torch.zeros(B, T, D, device=q.device, dtype=q.dtype)
    m = torch.full((B, T, 1), -float("inf"), device=q.device, dtype=q.dtype)
    l = torch.zeros(B, T, 1, device=q.device, dtype=q.dtype)

    num_blocks = (S + block_size - 1) // block_size

    for j in range(num_blocks):
        start = j * block_size
        end = min(start + block_size, S)

        k_block = k[:, start:end, :]  # [B, B_s, D]
        v_block = v[:, start:end, :]  # [B, B_s, D]

        # Block scores: [B, T, D] @ [B, D, B_s] -> [B, T, B_s]
        scores_j = torch.bmm(q, k_block.transpose(1, 2)) * scale

        # Online max update: m_new = max(m_old, max(scores_j))
        block_max = scores_j.max(dim=-1, keepdim=True).values  # [B, T, 1]
        m_new = torch.maximum(m, block_max)

        # Numerical rescale factors
        alpha = torch.exp(m - m_new)  # [B, T, 1]
        p_j = torch.exp(scores_j - m_new)  # [B, T, B_s]

        # Online sum update: l_new = alpha * l_old + sum(p_j)
        l_new = alpha * l + p_j.sum(dim=-1, keepdim=True)  # [B, T, 1]

        # Online output update: out = (alpha * l_old * out_old + p_j @ v_block) / l_new
        out = (alpha * l * out + torch.bmm(p_j, v_block)) / l_new
        m = m_new
        l = l_new

    return out, m, l


def standard_scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Standard global reference attention."""
    scores = torch.bmm(q, k.transpose(1, 2)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, v)


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

    # 2. FlashAttention Sequence Data
    B = 2
    T = 4
    S = 8
    D = 4
    block_size = 4
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, T, D) * 0.3
    k = torch.randn(B, S, D) * 0.3
    v = torch.randn(B, S, D) * 0.3
    scale_tensor = torch.tensor([scale])
    block_size_tensor = torch.tensor([block_size])

    flash_out, final_m, final_l = online_softmax_flash_attention(q, k, v, block_size, scale)
    std_out = standard_scaled_dot_product(q, k, v, scale)

    # Assert exact mathematical equivalence between tiled online softmax and global standard
    err_equiv = float(torch.max(torch.abs(flash_out - std_out)))
    assert err_equiv < 1e-6, f"FlashAttention equivalence failed: {err_equiv:.2e}"

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "q": q,
        "k": k,
        "v": v,
        "scale": scale_tensor,
        "block_size": block_size_tensor,
        "flash_out": flash_out,
        "final_m": final_m,
        "final_l": final_l,
        "std_out": std_out,
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
