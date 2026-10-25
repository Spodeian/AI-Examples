"""LLM.int8() Mixed Precision Outlier Decomposition — PyTorch 2.x reference implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition
Golden weights: weights/14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition.safetensors

``python pytorch2/14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def find_outlier_channels(x: torch.Tensor, threshold: float = 6.0) -> torch.Tensor:
    """Identifies outlier channel indices where max activation magnitude >= threshold."""
    channel_max = torch.max(torch.abs(x), dim=0).values
    outlier_mask = channel_max >= threshold
    outlier_indices = torch.where(outlier_mask)[0]
    return outlier_indices


def quantize_row_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8 quantization."""
    scale = torch.max(torch.abs(x), dim=-1, keepdim=True).values / 127.0
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(x / scale), -128.0, 127.0)
    return q, scale


def quantize_col_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column symmetric INT8 quantization."""
    scale = torch.max(torch.abs(w), dim=0, keepdim=True).values / 127.0
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(w / scale), -128.0, 127.0)
    return q, scale


def llm_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    threshold: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixed-precision outlier decomposition: Y = X_outlier W_outlier + X_int8 W_int8 + bias."""
    d_in = x.shape[1]
    channel_max = torch.max(torch.abs(x), dim=0).values
    outlier_mask = channel_max >= threshold
    outlier_cols = torch.where(outlier_mask)[0]
    normal_cols = torch.where(~outlier_mask)[0]

    if len(outlier_cols) > 0:
        x_outlier = x[:, outlier_cols]
        w_outlier = weight[outlier_cols, :]
        y_fp = x_outlier @ w_outlier
    else:
        y_fp = torch.zeros(x.shape[0], weight.shape[1], dtype=x.dtype, device=x.device)

    if len(normal_cols) > 0:
        x_normal = x[:, normal_cols]
        w_normal = weight[normal_cols, :]
        q_x, s_x = quantize_row_int8(x_normal)
        q_w, s_w = quantize_col_int8(w_normal)
        deq_x = q_x * s_x
        deq_w = q_w * s_w
        y_int8 = deq_x @ deq_w
    else:
        y_int8 = torch.zeros(x.shape[0], weight.shape[1], dtype=x.dtype, device=x.device)

    y_total = y_fp + y_int8
    if bias is not None:
        y_total = y_total + bias

    return y_total, y_fp, y_int8, outlier_cols.float()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Calibration activations with 2 emergent outlier dimensions (magnitude > 6.0)
    b, d_in, d_out = 16, 16, 8
    x_calib = torch.randn(b, d_in)
    # Channel 2 and 7 are massive emergent outliers (14.0 and 9.5)
    x_calib[:, 2] = torch.sign(x_calib[:, 2]) * (torch.abs(x_calib[:, 2]) + 12.0)
    x_calib[:, 7] = torch.sign(x_calib[:, 7]) * (torch.abs(x_calib[:, 7]) + 8.0)

    w_orig = torch.randn(d_in, d_out) * 0.2
    b_orig = torch.randn(d_out) * 0.05

    # 1. Standard pure INT8 quantization (with no outlier handling - suffers underflow)
    q_x_pure, s_x_pure = quantize_row_int8(x_calib)
    q_w_pure, s_w_pure = quantize_col_int8(w_orig)
    out_pure_int8 = (q_x_pure * s_x_pure) @ (q_w_pure * s_w_pure) + b_orig

    # 2. LLM.int8() mixed-precision decomposition
    threshold = 6.0
    out_total, out_fp, out_int8, outlier_indices = llm_int8_linear(
        x_calib, w_orig, b_orig, threshold=threshold
    )

    # 3. Exact FP32 ground truth
    out_fp32 = x_calib @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "out_pure_int8": out_pure_int8,
        "out_total": out_total,
        "out_fp": out_fp,
        "out_int8": out_int8,
        "outlier_indices": outlier_indices,
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
