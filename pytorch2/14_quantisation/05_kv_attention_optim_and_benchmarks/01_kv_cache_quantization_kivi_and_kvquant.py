"""KV Cache Quantization: KIVI and KVQuant — PyTorch 2.x reference implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant
Golden weights: weights/14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant.safetensors

``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def quantize_k_per_channel(
    k: torch.Tensor,
    bits: int = 2,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KIVI Key Quantization: per-channel asymmetric quantization along sequence dimension (dim=0)."""
    qmax = float((1 << bits) - 1)
    k_min = torch.min(k, dim=0, keepdim=True).values
    k_max = torch.max(k, dim=0, keepdim=True).values

    scale = torch.clamp((k_max - k_min) / qmax, min=eps)
    zero = torch.round(torch.clamp(-k_min / scale, 0.0, qmax))

    k_int = torch.clamp(torch.round(k / scale + zero), 0.0, qmax)
    k_deq = (k_int - zero) * scale
    return k_deq.clone(), k_int.clone(), scale.clone(), zero.clone()


def quantize_v_per_token(
    v: torch.Tensor,
    bits: int = 2,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KIVI Value Quantization: per-token asymmetric quantization along channel dimension (dim=1)."""
    qmax = float((1 << bits) - 1)
    v_min = torch.min(v, dim=1, keepdim=True).values
    v_max = torch.max(v, dim=1, keepdim=True).values

    scale = torch.clamp((v_max - v_min) / qmax, min=eps)
    zero = torch.round(torch.clamp(-v_min / scale, 0.0, qmax))

    v_int = torch.clamp(torch.round(v / scale + zero), 0.0, qmax)
    v_deq = (v_int - zero) * scale
    return v_deq.clone(), v_int.clone(), scale.clone(), zero.clone()


def kivi_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 4,
    bits: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KIVI Decoding Attention: quantize past tokens (2-bit K-channel / V-token) while keeping recent window FP32."""
    seq_len, d_k = k.shape
    if seq_len > window_size:
        k_past = k[: seq_len - window_size]
        v_past = v[: seq_len - window_size]
        k_past_deq, _, _, _ = quantize_k_per_channel(k_past, bits=bits)
        v_past_deq, _, _, _ = quantize_v_per_token(v_past, bits=bits)

        k_eff = torch.cat([k_past_deq, k[seq_len - window_size :]], dim=0)
        v_eff = torch.cat([v_past_deq, v[seq_len - window_size :]], dim=0)
    else:
        k_eff = k.clone()
        v_eff = v.clone()

    scores = (q @ k_eff.T) * (1.0 / math.sqrt(float(d_k)))
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v_eff
    return out.clone(), k_eff.clone(), v_eff.clone()


def kvquant_dense_sparse_key(
    k: torch.Tensor,
    outlier_channels: int = 2,
    bits: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KVQuant Outlier-Aware Key Quantization: keeps top outlier channels FP32, quantizes remaining 2-bit."""
    # Find channels with highest peak magnitude
    channel_peaks = torch.max(torch.abs(k), dim=0).values
    top_channels = torch.topk(channel_peaks, k=outlier_channels).indices

    mask = torch.zeros(k.shape[1], dtype=torch.bool, device=k.device)
    mask[top_channels] = True

    k_deq = k.clone()
    # Quantize non-outlier channels
    normal_channels = (~mask).nonzero(as_tuple=True)[0]
    k_normal = k[:, normal_channels]
    k_normal_deq, _, _, _ = quantize_k_per_channel(k_normal, bits=bits)
    k_deq[:, normal_channels] = k_normal_deq

    return k_deq.clone(), mask.float().clone(), channel_peaks.clone()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    seq_len, d_k, d_v = 16, 16, 16
    window_size = 4
    bits = 2

    q = torch.randn(1, d_k) * 0.8
    # Keys have prominent channel outliers (columns 2 and 5)
    k = torch.randn(seq_len, d_k) * 0.4
    k[:, 2] *= 4.5
    k[:, 5] *= 3.8

    # Values have token outliers (rows 3 and 11)
    v = torch.randn(seq_len, d_v) * 0.5
    v[3, :] *= 3.2
    v[11, :] *= 4.1

    # 1. KIVI Key per-channel quantization
    k_deq, k_int, k_scale, k_zero = quantize_k_per_channel(k, bits=bits)

    # 2. KIVI Value per-token quantization
    v_deq, v_int, v_scale, v_zero = quantize_v_per_token(v, bits=bits)

    # 3. KIVI Decoding Attention with sliding window
    out_kivi, k_eff, v_eff = kivi_attention_decode(q, k, v, window_size=window_size, bits=bits)

    # 4. KVQuant outlier channel decomposition
    k_kvquant, outlier_mask, channel_peaks = kvquant_dense_sparse_key(k, outlier_channels=2, bits=bits)

    # 5. Full-precision unquantized baseline attention
    scores_fp32 = (q @ k.T) * (1.0 / math.sqrt(float(d_k)))
    probs_fp32 = torch.softmax(scores_fp32, dim=-1)
    out_fp32 = probs_fp32 @ v

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "q": q,
        "k": k,
        "v": v,
        "k_deq": k_deq,
        "k_int": k_int,
        "k_scale": k_scale,
        "k_zero": k_zero,
        "v_deq": v_deq,
        "v_int": v_int,
        "v_scale": v_scale,
        "v_zero": v_zero,
        "k_eff": k_eff,
        "v_eff": v_eff,
        "out_kivi": out_kivi,
        "k_kvquant": k_kvquant,
        "outlier_mask": outlier_mask,
        "channel_peaks": channel_peaks,
        "out_fp32": out_fp32,
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
