"""SmoothQuant and Outlier Migration W8A8 — PyTorch 2.x reference implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8
Golden weights: weights/14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8.safetensors

``python pytorch2/14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def compute_smoothquant_scale(
    x_calib: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Computes migration scale s_j = (max |X_j|)^alpha / (max |W_j|)^(1 - alpha)."""
    act_max = torch.max(torch.abs(x_calib), dim=0).values
    weight_max = torch.max(torch.abs(weight), dim=1).values
    act_max = torch.clamp(act_max, min=eps)
    weight_max = torch.clamp(weight_max, min=eps)

    s = (act_max ** alpha) / (weight_max ** (1.0 - alpha))
    s = torch.clamp(s, min=eps)
    return s


def quantize_per_token_int8(
    x: torch.Tensor,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric INT8 activation quantization."""
    scale = torch.max(torch.abs(x), dim=-1, keepdim=True).values / qmax
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q, scale


def quantize_per_channel_int8(
    w: torch.Tensor,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column symmetric INT8 weight quantization."""
    scale = torch.max(torch.abs(w), dim=0, keepdim=True).values / qmax
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    return q, scale


def w8a8_gemm_unfused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard W8A8 GEMM without smoothing (per-token activation, per-channel weight)."""
    q_x, s_x = quantize_per_token_int8(x)
    q_w, s_w = quantize_per_channel_int8(weight)
    deq_x = q_x * s_x
    deq_w = q_w * s_w
    out = deq_x @ deq_w
    if bias is not None:
        out = out + bias
    return out


def smoothquant_w8a8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SmoothQuant W8A8 GEMM: scales activations and weights, then performs INT8 GEMM."""
    x_smooth = x / scale.unsqueeze(0)
    w_smooth = weight * scale.unsqueeze(1)

    q_x, s_x = quantize_per_token_int8(x_smooth)
    q_w, s_w = quantize_per_channel_int8(w_smooth)

    deq_x = q_x * s_x
    deq_w = q_w * s_w
    out = deq_x @ deq_w
    if bias is not None:
        out = out + bias
    return out, x_smooth, w_smooth


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Calibration activations with prominent outlier features
    n_samples, d_in, d_out = 64, 16, 8
    x_calib = torch.randn(n_samples, d_in)
    # Channel 0 has an emergent 15x activation outlier; channel 1 has 8x
    x_calib[:, 0] *= 15.0
    x_calib[:, 1] *= 8.0

    w_orig = torch.randn(d_in, d_out) * 0.2
    b_orig = torch.randn(d_out) * 0.05

    # 1. Un-smoothed baseline W8A8
    out_unsmoothed = w8a8_gemm_unfused(x_calib[:8], w_orig, b_orig)

    # 2. SmoothQuant scale computation (alpha = 0.5)
    scale = compute_smoothquant_scale(x_calib, w_orig, alpha=0.5)

    # 3. SmoothQuant W8A8
    out_smooth, x_smooth, w_smooth = smoothquant_w8a8_gemm(
        x_calib[:8], w_orig, scale, b_orig
    )

    # 4. FP32 ground truth
    out_fp32 = x_calib[:8] @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "scale": scale,
        "out_unsmoothed": out_unsmoothed,
        "x_smooth": x_smooth,
        "w_smooth": w_smooth,
        "out_smooth": out_smooth,
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
