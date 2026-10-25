"""PEFT Adapters Ia3 Prefix Prompt and Llama Adapter — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter
Golden weights: weights/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ia3_rescale(activation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(IA)^3 multiplicative scaling: a_scaled = a * scale.

    Args:
        activation: Intermediate activation [batch, seq_len, dim].
        scale: Learned vector [dim] or [1, 1, dim].
    """
    return activation * scale


def prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
) -> torch.Tensor:
    """Prefix-Tuning attention with prepended key and value prefixes.

    Args:
        q: Query tensor [batch, seq_len, dim].
        k: Base key tensor [batch, seq_len, dim].
        v: Base value tensor [batch, seq_len, dim].
        prefix_k: Learnable key prefix [batch, prefix_len, dim].
        prefix_v: Learnable value prefix [batch, prefix_len, dim].

    Returns:
        out: Context output tensor [batch, seq_len, dim].
    """
    k_cat = torch.cat([prefix_k, k], dim=1)
    v_cat = torch.cat([prefix_v, v], dim=1)
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k_cat.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_cat)


def llama_adapter_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """LLaMA-Adapter zero-gated prefix attention.

    Computes joint attention probabilities over [prefix; sequence], then scales
    the prefix contribution by learnable zero-initialized gate:
    out = Attn_seq @ V + gate * (Attn_prefix @ V_prefix)

    Args:
        q: Query tensor [batch, seq_len, dim].
        k: Base key tensor [batch, seq_len, dim].
        v: Base value tensor [batch, seq_len, dim].
        prefix_k: Adapter key prefix [batch, prefix_len, dim].
        prefix_v: Adapter value prefix [batch, prefix_len, dim].
        gate: Learnable scalar gate parameter [1].

    Returns:
        out: Zero-gated adapted output [batch, seq_len, dim].
    """
    k_cat = torch.cat([prefix_k, k], dim=1)
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k_cat.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)

    prefix_len = prefix_k.shape[1]
    attn_prefix = attn[:, :, :prefix_len]
    attn_seq = attn[:, :, prefix_len:]

    seq_out = torch.matmul(attn_seq, v)
    prefix_out = torch.matmul(attn_prefix, prefix_v)
    return seq_out + gate * prefix_out


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    batch = 2
    seq_len = 4
    prefix_len = 2
    dim = 8

    q = torch.randn(batch, seq_len, dim)
    k = torch.randn(batch, seq_len, dim)
    v = torch.randn(batch, seq_len, dim)

    prefix_k = torch.randn(batch, prefix_len, dim) * 0.5
    prefix_v = torch.randn(batch, prefix_len, dim) * 0.5

    # 1. (IA)^3 scaling
    ia3_scale = torch.randn(dim) * 0.2 + 1.0
    ia3_out = ia3_rescale(v, ia3_scale)

    # 2. Prefix Attention
    prefix_attn_out = prefix_attention(q, k, v, prefix_k, prefix_v)

    # 3. LLaMA-Adapter Zero-Gate Forward
    gate = torch.tensor([0.2], dtype=torch.float32)
    llama_adapter_out = llama_adapter_forward(q, k, v, prefix_k, prefix_v, gate)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "q": q,
        "k": k,
        "v": v,
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "ia3_scale": ia3_scale,
        "ia3_out": ia3_out,
        "prefix_attn_out": prefix_attn_out,
        "gate": gate,
        "llama_adapter_out": llama_adapter_out,
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
