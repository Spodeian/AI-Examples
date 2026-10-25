"""TorchAO Modern API and Affine Quantized Tensor — PyTorch 2.x reference implementation.

Topic: 14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor
Golden weights: weights/14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor.safetensors

``python pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def affine_quantize_blockwise(
    weight: torch.Tensor,
    group_size: int = 16,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine Quantized Tensor (AQT) block-wise symmetric quantization."""
    orig_shape = weight.shape
    flat = weight.flatten()
    n_groups = len(flat) // group_size
    scales = []
    q_ints = []
    dequants = []

    for g in range(n_groups):
        chunk = flat[g * group_size : (g + 1) * group_size]
        amax = float(torch.max(torch.abs(chunk)))
        scale = amax / qmax if amax > 0.0 else 1.0
        q = torch.clamp(torch.round(chunk / scale), qmin, qmax)
        deq = q * scale
        scales.append(scale)
        q_ints.append(q)
        dequants.append(deq)

    scales_tensor = torch.tensor(scales, dtype=torch.float32)
    q_int_tensor = torch.cat(q_ints).reshape(orig_shape)
    dequant_tensor = torch.cat(dequants).reshape(orig_shape)
    return dequant_tensor, scales_tensor, q_int_tensor


def dynamic_w8a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic Activation INT8, Static Weight INT8 (W8A8) GEMM."""
    s_x = torch.max(torch.abs(x), dim=-1, keepdim=True).values / 127.0
    s_x = torch.where(s_x == 0.0, torch.ones_like(s_x), s_x)
    x_q = torch.clamp(torch.round(x / s_x), -128.0, 127.0)

    s_w = torch.max(torch.abs(weight), dim=0, keepdim=True).values / 127.0
    s_w = torch.where(s_w == 0.0, torch.ones_like(s_w), s_w)
    w_q = torch.clamp(torch.round(weight / s_w), -128.0, 127.0)

    y_int = x_q @ w_q
    y = y_int * (s_x * s_w) + bias
    return y, s_x, s_w


def int4_weight_only_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """INT4 Weight-Only (W4A16) block-wise quantization with FP32/FP16 activation."""
    w_deq, scales, q_int = affine_quantize_blockwise(weight, group_size=group_size, qmin=-8.0, qmax=7.0)
    y = x @ w_deq + bias
    return y, w_deq, scales


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Linear dimensions
    b, d_in, d_out = 4, 16, 8
    x_act = torch.randn(b, d_in)
    w_linear = torch.randn(d_in, d_out) * 0.2
    b_linear = torch.randn(d_out) * 0.1

    # 1. AQT INT8 Block-wise Quantization
    w_block_dequant, w_block_scales, w_block_int = affine_quantize_blockwise(
        w_linear, group_size=16, qmin=-128.0, qmax=127.0
    )
    out_w_only = x_act @ w_block_dequant + b_linear

    # 2. Dynamic W8A8 Linear
    w8a8_out, w8a8_s_x, w8a8_s_w = dynamic_w8a8_linear(x_act, w_linear, b_linear)

    # 3. INT4 Weight-Only Linear
    out_w4_only, w4_block_dequant, w4_block_scales = int4_weight_only_linear(
        x_act, w_linear, b_linear, group_size=16
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_act": x_act,
        "w_linear": w_linear,
        "b_linear": b_linear,
        "w_block_dequant": w_block_dequant,
        "w_block_scales": w_block_scales,
        "w_block_int": w_block_int,
        "out_w_only": out_w_only,
        "w8a8_out": w8a8_out,
        "w8a8_s_x": w8a8_s_x,
        "w8a8_s_w": w8a8_s_w,
        "w4_block_dequant": w4_block_dequant,
        "w4_block_scales": w4_block_scales,
        "out_w4_only": out_w4_only,
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
