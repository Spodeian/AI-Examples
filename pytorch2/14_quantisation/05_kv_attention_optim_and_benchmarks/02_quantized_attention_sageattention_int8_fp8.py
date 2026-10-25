"""Quantized Attention: SageAttention INT8/FP8 — PyTorch 2.x reference implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8
Golden weights: weights/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.safetensors

``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def key_mean_smoothing(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Key-mean smoothing: subtract sequence mean from Key matrix.

    Softmax invariance theorem:
    softmax(Q (K - K_mean)^T / sqrt(d)) = softmax((Q K^T - Q K_mean^T) / sqrt(d))
    Because Q K_mean^T is identical across all key columns for a given query row,
    it cancels out exactly in row-wise softmax!
    """
    k_mean = torch.mean(k, dim=0, keepdim=True)
    k_smoothed = k - k_mean
    return k_smoothed.clone(), k_mean.clone()


def quantize_symmetric_int8(
    x: torch.Tensor,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor INT8 quantization."""
    amax = torch.max(torch.abs(x))
    scale = torch.clamp(amax / 127.0, min=eps)
    x_int8 = torch.clamp(torch.round(x / scale), -127.0, 127.0)
    x_deq = x_int8 * scale
    return x_deq.clone(), x_int8.clone(), scale.reshape(1).clone()


def sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SageAttention forward pass: Key-mean smoothing -> INT8 QK^T -> Softmax -> INT8/FP8 PV."""
    d_k = q.shape[1]
    k_smoothed, k_mean = key_mean_smoothing(k)

    # 1. Symmetric INT8 quantization for Q and smoothed K
    q_deq, q_int8, s_q = quantize_symmetric_int8(q)
    k_deq, k_int8, s_k = quantize_symmetric_int8(k_smoothed)

    # 2. Integer GEMM: int8 x int8 -> int32 accumulator, rescaled to float
    int_acc = q_int8 @ k_int8.T
    scores = int_acc * ((s_q * s_k) / math.sqrt(float(d_k)))

    # 3. Softmax
    probs = torch.softmax(scores, dim=-1)

    # 4. Value quantization and projection
    v_deq, v_int8, s_v = quantize_symmetric_int8(v)
    out = probs @ v_deq

    return out.clone(), scores.clone(), probs.clone(), int_acc.clone(), k_smoothed.clone()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, seq_len, d_k, d_v = 4, 16, 16, 16

    q = torch.randn(b, d_k) * 0.8
    # Keys have large non-zero DC bias / mean shifts
    k = torch.randn(seq_len, d_k) * 0.4 + 2.5
    v = torch.randn(seq_len, d_v) * 0.5

    # 1. Key-mean smoothing
    k_smoothed, k_mean = key_mean_smoothing(k)

    # 2. INT8 quantization
    q_deq, q_int8, s_q = quantize_symmetric_int8(q)
    k_deq, k_int8, s_k = quantize_symmetric_int8(k_smoothed)
    v_deq, v_int8, s_v = quantize_symmetric_int8(v)

    # 3. SageAttention full forward pass
    out_sage, scores_sage, probs_sage, int_acc, _ = sage_attention(q, k, v)

    # 4. FP32 unquantized attention with smoothing (exact mathematical equivalence)
    scores_fp32_smooth = (q @ k_smoothed.T) / math.sqrt(float(d_k))
    probs_fp32_smooth = torch.softmax(scores_fp32_smooth, dim=-1)
    out_fp32_smooth = probs_fp32_smooth @ v

    # 5. FP32 unquantized attention without smoothing (original ground truth)
    scores_fp32_raw = (q @ k.T) / math.sqrt(float(d_k))
    probs_fp32_raw = torch.softmax(scores_fp32_raw, dim=-1)
    out_fp32_raw = probs_fp32_raw @ v

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "q": q,
        "k": k,
        "v": v,
        "k_smoothed": k_smoothed,
        "k_mean": k_mean,
        "q_deq": q_deq,
        "q_int8": q_int8,
        "s_q": s_q,
        "k_deq": k_deq,
        "k_int8": k_int8,
        "s_k": s_k,
        "v_deq": v_deq,
        "v_int8": v_int8,
        "s_v": s_v,
        "int_acc": int_acc,
        "scores_sage": scores_sage,
        "probs_sage": probs_sage,
        "out_sage": out_sage,
        "probs_fp32_smooth": probs_fp32_smooth,
        "probs_fp32_raw": probs_fp32_raw,
        "out_fp32_smooth": out_fp32_smooth,
        "out_fp32_raw": out_fp32_raw,
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
