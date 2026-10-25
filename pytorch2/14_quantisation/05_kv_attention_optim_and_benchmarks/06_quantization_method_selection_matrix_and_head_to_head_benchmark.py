"""Quantization Method Selection Matrix and Head to Head Benchmark — PyTorch 2.x reference implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark
Golden weights: weights/14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark.safetensors

``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def rtn_quantize(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Round-to-Nearest uniform symmetric per-row quantization."""
    levels = float((2 ** (bits - 1)) - 1)
    scale = torch.amax(torch.abs(w), dim=-1, keepdim=True) / levels
    q = torch.clamp(torch.round(w / (scale + 1e-8)), -levels, levels)
    return q * scale


def awq_quantize(w: torch.Tensor, x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Activation-Aware Weight Quantization with salient column protection."""
    levels = float((2 ** (bits - 1)) - 1)
    act_scale = torch.mean(torch.abs(x), dim=0, keepdim=True)
    s_awq = torch.clamp(act_scale ** 0.5, min=1e-3)
    w_scaled = w * s_awq
    s_w = torch.amax(torch.abs(w_scaled), dim=-1, keepdim=True) / levels
    q = torch.clamp(torch.round(w_scaled / (s_w + 1e-8)), -levels, levels)
    return (q * s_w) / s_awq


def bitnet_quantize(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 native ternary {-scale, 0, +scale} quantization."""
    scale = torch.mean(torch.abs(w))
    q = torch.clamp(torch.round(w / (scale + 1e-8)), -1.0, 1.0)
    return q * scale


def nf4_quantize(w: torch.Tensor) -> torch.Tensor:
    """NormalFloat 4-bit (NF4) quantile codebook quantization."""
    levels = NF4_LEVELS.to(w.device)
    scale = torch.amax(torch.abs(w), dim=-1, keepdim=True)
    norm_w = w / (scale + 1e-8)
    diffs = torch.abs(norm_w.unsqueeze(-1) - levels)
    idx = torch.argmin(diffs, dim=-1)
    return levels[idx] * scale


def compute_sqnr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Computes Signal-to-Quantization-Noise Ratio (SQNR) in dB."""
    sig_pwr = float(torch.mean(signal ** 2))
    noise_pwr = float(torch.mean(noise ** 2))
    return float(10.0 * torch.log10(torch.tensor(sig_pwr / (noise_pwr + 1e-12))))


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    w = torch.randn(64, 64) * 0.1
    x = torch.randn(16, 64)

    # 5 Key Representative Formats across Category 14
    w_rtn8 = rtn_quantize(w, bits=8)
    w_rtn4 = rtn_quantize(w, bits=4)
    w_awq4 = awq_quantize(w, x, bits=4)
    w_bitnet = bitnet_quantize(w)
    w_nf4 = nf4_quantize(w)

    # Output activations
    y_true = x @ w
    y_rtn8 = x @ w_rtn8
    y_rtn4 = x @ w_rtn4
    y_awq4 = x @ w_awq4
    y_bitnet = x @ w_bitnet
    y_nf4 = x @ w_nf4

    methods = [
        (w_rtn8, y_rtn8),
        (w_rtn4, y_rtn4),
        (w_awq4, y_awq4),
        (w_bitnet, y_bitnet),
        (w_nf4, y_nf4),
    ]

    weight_sqnrs = []
    act_sqnrs = []
    for w_q, y_q in methods:
        weight_sqnrs.append(compute_sqnr(w, w - w_q))
        act_sqnrs.append(compute_sqnr(y_true, y_true - y_q))

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "w": w,
        "x": x,
        "w_rtn8": w_rtn8,
        "w_rtn4": w_rtn4,
        "w_awq4": w_awq4,
        "w_bitnet": w_bitnet,
        "w_nf4": w_nf4,
        "y_true": y_true,
        "y_rtn8": y_rtn8,
        "y_rtn4": y_rtn4,
        "y_awq4": y_awq4,
        "y_bitnet": y_bitnet,
        "y_nf4": y_nf4,
        "weight_sqnrs": torch.tensor(weight_sqnrs, dtype=torch.float32),
        "act_sqnrs": torch.tensor(act_sqnrs, dtype=torch.float32),
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
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
