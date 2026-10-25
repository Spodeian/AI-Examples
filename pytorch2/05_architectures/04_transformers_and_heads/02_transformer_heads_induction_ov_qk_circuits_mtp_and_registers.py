"""Transformer Heads: Induction, OV/QK Circuits, MTP, and Registers — PyTorch 2.x reference implementation.

Topic: 05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers
Golden weights: weights/05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers.safetensors

``python pytorch2/05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ov_qk_circuits(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates decoupled QK (attention pattern) and OV (value copy) circuits.
    
    Scores = (x_q @ w_q) @ (x_k @ w_k)^T * scale
    A = softmax(Scores, dim=-1)
    Out = A @ ((x_v @ w_v) @ w_o)
    """
    q = x_q @ w_q  # [B, T_q, D_head]
    k = x_k @ w_k  # [B, T_k, D_head]
    v = x_v @ w_v  # [B, T_k, D_head]
    ov = v @ w_o   # [B, T_k, D_model]

    scores = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, T_q, T_k]
    attn_weights = F.softmax(scores, dim=-1)

    out = torch.bmm(attn_weights, ov)  # [B, T_q, D_model]
    return out, attn_weights


def register_prepend_and_mtp(
    x_seq: torch.Tensor,
    registers: torch.Tensor,
    w_mtp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepends register tokens and computes Multi-Token Prediction (MTP) heads.
    
    x_seq: [B, T, D_model]
    registers: [B, Num_reg, D_model]
    w_mtp: [K_future, D_vocab, D_model]
    
    Returns augmented sequence and K-step ahead predicted logits.
    """
    # 1. Register prepend: [B, Num_reg + T, D_model]
    x_aug = torch.cat([registers, x_seq], dim=1)

    # 2. Multi-token prediction on last token:
    x_last = x_seq[:, -1, :]  # [B, D_model]
    # For each future token k: logits_k = x_last @ w_mtp[k].T
    mtp_logits = torch.einsum("bd,kvd->bkv", x_last, w_mtp)  # [B, K, D_vocab]

    return x_aug, mtp_logits


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

    # 2. QK and OV Circuits
    B = 2
    T = 4
    D_model = 4
    D_head = 4
    scale = 1.0 / math.sqrt(D_head)

    x_q = torch.randn(B, T, D_model) * 0.2
    x_k = torch.randn(B, T, D_model) * 0.2
    x_v = torch.randn(B, T, D_model) * 0.2
    w_q = torch.randn(D_model, D_head) * 0.2
    w_k = torch.randn(D_model, D_head) * 0.2
    w_v = torch.randn(D_model, D_head) * 0.2
    w_o = torch.randn(D_head, D_model) * 0.2
    scale_tensor = torch.tensor([scale])

    circuit_out, circuit_attn = ov_qk_circuits(x_q, x_k, x_v, w_q, w_k, w_v, w_o, scale)

    # 3. Registers & Multi-Token Prediction (K=2 future steps, D_vocab=6)
    Num_reg = 2
    K_future = 2
    D_vocab = 6
    registers = torch.randn(B, Num_reg, D_model) * 0.1
    w_mtp = torch.randn(K_future, D_vocab, D_model) * 0.2

    x_aug, mtp_logits = register_prepend_and_mtp(x_q, registers, w_mtp)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "x_q": x_q,
        "x_k": x_k,
        "x_v": x_v,
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "w_o": w_o,
        "scale": scale_tensor,
        "circuit_out": circuit_out,
        "circuit_attn": circuit_attn,
        "registers": registers,
        "w_mtp": w_mtp,
        "x_aug": x_aug,
        "mtp_logits": mtp_logits,
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
