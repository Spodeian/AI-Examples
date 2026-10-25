"""Capstone Modern LLM End to End Pipeline — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline
Golden weights: weights/17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Root Mean Square Normalization."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * gamma


def causal_attention_block(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
) -> torch.Tensor:
    """Causal self-attention layer with causal upper-triangular masking."""
    b, s, d = x.shape
    q = x @ wq
    k = x @ wk
    v = x @ wv

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    mask = torch.triu(torch.full((s, s), -10000.0, device=x.device), diagonal=1)
    attn = F.softmax(scores + mask, dim=-1)
    return (attn @ v) @ wo


def swiglu_ffn_block(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU feed-forward network block."""
    gate = F.silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down


def modern_llm_pipeline_forward(
    x: torch.Tensor,
    gamma1: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    gamma2: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    gamma_final: torch.Tensor,
    w_head: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end modern LLM forward pipeline with pre-RMSNorm and residual streams."""
    # Attention sub-layer
    normed_1 = rmsnorm(x, gamma1)
    attn = causal_attention_block(normed_1, wq, wk, wv, wo)
    h1 = x + attn

    # SwiGLU FFN sub-layer
    normed_2 = rmsnorm(h1, gamma2)
    ffn = swiglu_ffn_block(normed_2, w_gate, w_up, w_down)
    h_out = h1 + ffn

    # Output projection head
    h_final = rmsnorm(h_out, gamma_final)
    logits = h_final @ w_head
    return logits, h_out, attn, ffn


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    b, s, d, v = 2, 4, 4, 16

    x = torch.randn(b, s, d)
    gamma1 = torch.randn(d) * 0.1 + 1.0
    wq = torch.randn(d, d) * 0.2
    wk = torch.randn(d, d) * 0.2
    wv = torch.randn(d, d) * 0.2
    wo = torch.randn(d, d) * 0.2

    gamma2 = torch.randn(d) * 0.1 + 1.0
    w_gate = torch.randn(d, d) * 0.2
    w_up = torch.randn(d, d) * 0.2
    w_down = torch.randn(d, d) * 0.2

    gamma_final = torch.randn(d) * 0.1 + 1.0
    w_head = torch.randn(d, v) * 0.2

    logits, h_out, attn, ffn = modern_llm_pipeline_forward(
        x, gamma1, wq, wk, wv, wo, gamma2, w_gate, w_up, w_down, gamma_final, w_head
    )

    linear_w = torch.randn(d, d) * 0.1
    linear_b = torch.randn(d) * 0.1
    anchor_in = h_out.reshape(-1, d)
    test_out = linear_anchor(anchor_in, linear_w, linear_b)

    tensors = {
        "x": x,
        "gamma1": gamma1,
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "gamma2": gamma2,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "gamma_final": gamma_final,
        "w_head": w_head,
        "logits": logits,
        "h_out": h_out,
        "attn": attn,
        "ffn": ffn,
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
