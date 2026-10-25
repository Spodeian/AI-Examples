"""Transformer Archetypes: BERT, GPT, T5, ViT, Swin, Perceiver, and Jamba — PyTorch 2.x reference implementation.

Topic: 05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba
Golden weights: weights/05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba.safetensors

``python pytorch2/05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def perceiver_cross_attention(
    latents: torch.Tensor,
    inputs: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perceiver cross-attention bottleneck: compresses large inputs into small latent array.
    
    latents: [B, N_latents, D_latent]
    inputs: [B, M_inputs, D_input]
    
    q: [B, N, D_head]
    k, v: [B, M, D_head]
    A = softmax(q @ k.T * scale)  # [B, N, M]
    out = (A @ v) @ w_o           # [B, N, D_latent]
    """
    q = latents @ w_q  # [B, N, D_head]
    k = inputs @ w_k   # [B, M, D_head]
    v = inputs @ w_v   # [B, M, D_head]

    scores = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, N, M]
    attn_weights = F.softmax(scores, dim=-1)

    ctx = torch.bmm(attn_weights, v)  # [B, N, D_head]
    out = ctx @ w_o                    # [B, N, D_latent]
    return out, attn_weights


def causal_decoder_mask(
    scores: torch.Tensor,
) -> torch.Tensor:
    """Applies causal autoregressive mask (GPT archetype)."""
    T = scores.shape[-1]
    mask = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
    return torch.where(mask.unsqueeze(0), scores, torch.full_like(scores, -1e9))


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

    # 2. Perceiver Cross-Attention Bottleneck
    B = 2
    N_latents = 3   # 3 latent bottleneck slots
    M_inputs = 8    # 8 large input features
    D_latent = 4
    D_input = 4
    D_head = 4
    scale = 1.0 / math.sqrt(D_head)

    latents = torch.randn(B, N_latents, D_latent) * 0.3
    inputs = torch.randn(B, M_inputs, D_input) * 0.3
    w_q = torch.randn(D_latent, D_head) * 0.2
    w_k = torch.randn(D_input, D_head) * 0.2
    w_v = torch.randn(D_input, D_head) * 0.2
    w_o = torch.randn(D_head, D_latent) * 0.2
    scale_tensor = torch.tensor([scale])

    perceiver_out, perceiver_attn = perceiver_cross_attention(
        latents, inputs, w_q, w_k, w_v, w_o, scale
    )

    # 3. Causal Decoder Masking
    raw_scores = torch.randn(B, 4, 4)
    causal_scores = causal_decoder_mask(raw_scores)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "latents": latents,
        "inputs": inputs,
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "w_o": w_o,
        "scale": scale_tensor,
        "perceiver_out": perceiver_out,
        "perceiver_attn": perceiver_attn,
        "raw_scores": raw_scores,
        "causal_scores": causal_scores,
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
