"""PT2E Export Quantization Workflow — PyTorch 2.x reference implementation.

Topic: 14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow
Golden weights: weights/14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow.safetensors

``python pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def conv2d_bn_fold(
    w_conv: torch.Tensor,
    b_conv: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_gamma: torch.Tensor,
    bn_beta: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algebraic Conv2d + BatchNorm2d fusion: W_f = W * (gamma / sqrt(var + eps))."""
    scale = bn_gamma / torch.sqrt(bn_var + eps)
    w_folded = w_conv * scale.view(-1, 1, 1, 1)
    b_folded = (b_conv - bn_mean) * scale + bn_beta
    return w_folded, b_folded


def linear_bn_fold(
    w_linear: torch.Tensor,
    b_linear: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_gamma: torch.Tensor,
    bn_beta: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algebraic Linear + BatchNorm1d fusion: W_f = W * (gamma / sqrt(var + eps))."""
    scale = bn_gamma / torch.sqrt(bn_var + eps)
    w_folded = w_linear * scale.unsqueeze(0)
    b_folded = (b_linear - bn_mean) * scale + bn_beta
    return w_folded, b_folded


def quantize_symmetric_int8(
    x: torch.Tensor,
    scale: float,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> torch.Tensor:
    """Simulated INT8 Quantize-Dequantize (Q/DQ) node."""
    if scale == 0.0:
        return torch.zeros_like(x)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


def pt2e_qdq_conv2d(
    x: torch.Tensor,
    w_folded: torch.Tensor,
    b_folded: torch.Tensor,
    s_x: float,
    s_w: float,
) -> torch.Tensor:
    """Low-level Q/DQ Conv2d node following PT2E lowering graph."""
    x_q = quantize_symmetric_int8(x, s_x)
    w_q = quantize_symmetric_int8(w_folded, s_w)
    return F.conv2d(x_q, w_q, b_folded)


def pt2e_qdq_linear(
    x: torch.Tensor,
    w_folded: torch.Tensor,
    b_folded: torch.Tensor,
    s_x: float,
    s_w: float,
) -> torch.Tensor:
    """Low-level Q/DQ Linear node following PT2E lowering graph."""
    x_q = quantize_symmetric_int8(x, s_x)
    w_q = quantize_symmetric_int8(w_folded, s_w)
    return x_q @ w_q + b_folded


def unfused_conv_bn_quantize(
    x: torch.Tensor,
    w_conv: torch.Tensor,
    b_conv: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_gamma: torch.Tensor,
    bn_beta: torch.Tensor,
    s_w: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pathological anti-pattern: quantizing Conv weights before unfused BatchNorm."""
    w_q = quantize_symmetric_int8(w_conv, s_w)
    conv_out = F.conv2d(x, w_q, b_conv)
    scale = bn_gamma / torch.sqrt(bn_var + eps)
    return (conv_out - bn_mean.view(1, -1, 1, 1)) * scale.view(1, -1, 1, 1) + bn_beta.view(1, -1, 1, 1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Conv-BN Folding & Quantization
    c_in, c_out, k = 4, 8, 3
    w_conv = torch.randn(c_out, c_in, k, k) * 0.1
    b_conv = torch.randn(c_out) * 0.1
    bn_mean = torch.randn(c_out) * 0.05
    bn_var = torch.rand(c_out) * 0.5 + 0.1
    bn_gamma = torch.rand(c_out) * 0.5 + 0.8
    bn_beta = torch.randn(c_out) * 0.05

    w_folded, b_folded = conv2d_bn_fold(w_conv, b_conv, bn_mean, bn_var, bn_gamma, bn_beta)

    # 2. Linear-BN Folding
    d_in, d_out = 4, 6
    w_lin = torch.randn(d_in, d_out) * 0.1
    b_lin = torch.randn(d_out) * 0.1
    lin_mean = torch.randn(d_out) * 0.05
    lin_var = torch.rand(d_out) * 0.5 + 0.1
    lin_gamma = torch.rand(d_out) * 0.5 + 0.8
    lin_beta = torch.randn(d_out) * 0.05

    w_lin_folded, b_lin_folded = linear_bn_fold(w_lin, b_lin, lin_mean, lin_var, lin_gamma, lin_beta)

    # 3. QDQ Execution
    x_conv = torch.randn(2, 4, 8, 8)
    s_x_conv = float(torch.max(torch.abs(x_conv))) / 127.0
    s_w_conv = float(torch.max(torch.abs(w_folded))) / 127.0
    out_qdq_conv = pt2e_qdq_conv2d(x_conv, w_folded, b_folded, s_x_conv, s_w_conv)

    x_lin = torch.randn(4, 4)
    s_x_lin = float(torch.max(torch.abs(x_lin))) / 127.0
    s_w_lin = float(torch.max(torch.abs(w_lin_folded))) / 127.0
    out_qdq_lin = pt2e_qdq_linear(x_lin, w_lin_folded, b_lin_folded, s_x_lin, s_w_lin)

    # 4. Unfused vs Folded Pathological Demonstration
    out_unfused = unfused_conv_bn_quantize(x_conv, w_conv, b_conv, bn_mean, bn_var, bn_gamma, bn_beta, s_w_conv)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "w_conv": w_conv,
        "b_conv": b_conv,
        "bn_mean": bn_mean,
        "bn_var": bn_var,
        "bn_gamma": bn_gamma,
        "bn_beta": bn_beta,
        "w_folded": w_folded,
        "b_folded": b_folded,
        "w_lin": w_lin,
        "b_lin": b_lin,
        "lin_mean": lin_mean,
        "lin_var": lin_var,
        "lin_gamma": lin_gamma,
        "lin_beta": lin_beta,
        "w_lin_folded": w_lin_folded,
        "b_lin_folded": b_lin_folded,
        "x_conv": x_conv,
        "out_qdq_conv": out_qdq_conv,
        "x_lin": x_lin,
        "out_qdq_lin": out_qdq_lin,
        "out_unfused": out_unfused,
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
