"""Torch Library Custom Ops and Fake Kernels — PyTorch 2.x reference implementation.

Topic: 12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels
Golden weights: weights/12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels.safetensors

``python pytorch2/12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels.py`` verifies against the golden checkpoint;
``python pytorch2/12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def custom_soft_shrink(x: torch.Tensor, lambd: float = 0.5) -> torch.Tensor:
    """Custom L1 proximal shrinkage operator:
    shrink(x, lambd) = sign(x) * max(|x| - lambd, 0)
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - float(lambd), min=0.0)


def custom_soft_shrink_meta(x_shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Fake/Meta kernel: symbolically computes output shape and strides without storage."""
    return torch.empty(x_shape, dtype=dtype, device=device)


def custom_asymmetric_quantize(
    x: torch.Tensor,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-wise 8-bit asymmetric quantization simulated dequantization:
    s = clamp((max(x) - min(x)) / 255.0, min=eps)
    z = clamp(round(-min(x) / s), 0, 255)
    q = clamp(round(x / s) + z, 0, 255)
    x_hat = (q - z) * s
    """
    x_min = torch.amin(x, dim=-1, keepdim=True)
    x_max = torch.amax(x, dim=-1, keepdim=True)
    scale = torch.clamp((x_max - x_min) / 255.0, min=float(eps))
    zero_point = torch.clamp(torch.round(-x_min / scale), min=0.0, max=255.0)
    q = torch.clamp(torch.round(x / scale) + zero_point, min=0.0, max=255.0)
    x_hat = (q - zero_point) * scale
    return x_hat, scale, zero_point


def custom_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Depthwise 1D Convolution with no groups cross-talk:
    x: (B, C, L)
    weight: (C, K)
    bias: (C,)
    y[b, c, t] = bias[c] + sum_k x[b, c, t + k] * weight[c, k]
    """
    b, c, l = x.shape
    c_w, k = weight.shape
    assert c == c_w, "Channel dimension mismatch"
    out_len = l - k + 1
    y = torch.zeros(b, c, out_len, dtype=x.dtype, device=x.device)
    for t in range(out_len):
        slice_x = x[:, :, t : t + k]  # (B, C, K)
        prod = torch.sum(slice_x * weight.unsqueeze(0), dim=-1)  # (B, C)
        y[:, :, t] = prod + bias.unsqueeze(0)
    return y


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Custom Soft Shrinkage
    shrink_x = torch.randn(4, 5) * 2.0
    shrink_ref = custom_soft_shrink(shrink_x, lambd=0.5)

    # 2. Custom Asymmetric Quantization
    quant_x = torch.randn(4, 8) * 3.0 + 1.0
    quant_x_hat, quant_scale, quant_zero = custom_asymmetric_quantize(quant_x, eps=1e-7)

    # 3. Custom Depthwise Conv1D
    conv_x = torch.randn(2, 3, 6)
    conv_w = torch.randn(3, 3) * 0.2
    conv_b = torch.randn(3) * 0.1
    conv_ref = custom_depthwise_conv1d(conv_x, conv_w, conv_b)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "shrink_x": shrink_x,
        "shrink_ref": shrink_ref,
        "quant_x": quant_x,
        "quant_x_hat": quant_x_hat,
        "quant_scale": quant_scale,
        "quant_zero": quant_zero,
        "conv_x": conv_x,
        "conv_w": conv_w,
        "conv_b": conv_b,
        "conv_ref": conv_ref,
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
