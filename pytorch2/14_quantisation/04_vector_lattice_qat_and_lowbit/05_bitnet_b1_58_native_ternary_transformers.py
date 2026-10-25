"""BitNet b1.58 Native Ternary Transformers — PyTorch 2.x reference implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers
Golden weights: weights/14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers.safetensors

``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def weight_quant_bitnet(w: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BitNet b1.58 weight quantization: scales by mean(|W|) and clips to {-1, 0, +1}."""
    beta = torch.mean(torch.abs(w), dim=(0, 1), keepdim=True)
    scale = torch.clamp(beta, min=eps)
    w_ternary = torch.clamp(torch.round(w / scale), -1.0, 1.0)
    w_deq = w_ternary * scale
    return w_deq, w_ternary, beta


def activation_quant_bitnet(
    x: torch.Tensor,
    q_b: float = 127.0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BitNet b1.58 per-token activation quantization to 8-bit integers."""
    # gamma = max_j |X_ij| per token [batch, 1]
    gamma = torch.max(torch.abs(x), dim=1, keepdim=True).values
    scale = torch.clamp(gamma / q_b, min=eps)
    x_int8 = torch.clamp(torch.round(x / scale), -q_b, q_b)
    x_deq = x_int8 * scale
    return x_deq, x_int8, gamma


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Root Mean Square Layer Normalization (SubLN / RMSNorm)."""
    variance = torch.mean(x ** 2, dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def bitlinear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    rms_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BitLinear layer forward pass: SubLN -> INT8 act quant -> Ternary weight quant -> MatMul."""
    x_norm = rms_norm(x, rms_weight)
    x_q, _, _ = activation_quant_bitnet(x_norm, q_b=127.0)
    w_q, _, _ = weight_quant_bitnet(weight)
    y = x_q @ w_q
    if bias is not None:
        y = y + bias
    return y, x_q, w_q


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, d_in, d_out = 16, 16, 8
    x_raw = torch.randn(b, d_in) * 1.5
    w_orig = torch.randn(d_in, d_out) * 0.4
    b_orig = torch.randn(d_out) * 0.05
    rms_weight = torch.ones(d_in) * 1.05

    # 1. SubLN / RMSNorm
    x_norm = rms_norm(x_raw, rms_weight)

    # 2. Activation quantization
    x_q, x_int8, gamma = activation_quant_bitnet(x_norm, q_b=127.0)

    # 3. Ternary weight quantization
    w_q, w_ternary, beta = weight_quant_bitnet(w_orig)

    # 4. BitLinear full forward pass
    out_bitlinear, _, _ = bitlinear_forward(x_raw, w_orig, rms_weight, b_orig)

    # 5. Multiplication-free pure integer matmul output: (x_int8 @ w_ternary) * (gamma * beta / 127.0) + bias
    integer_acc = x_int8 @ w_ternary
    out_integer_scaled = integer_acc * (gamma * beta / 127.0) + b_orig

    # 6. FP32 ground truth forward pass
    out_fp32 = x_norm @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_raw": x_raw,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "rms_weight": rms_weight,
        "x_norm": x_norm,
        "x_q": x_q,
        "x_int8": x_int8,
        "gamma": gamma,
        "w_q": w_q,
        "w_ternary": w_ternary,
        "beta": beta,
        "integer_acc": integer_acc,
        "out_bitlinear": out_bitlinear,
        "out_integer_scaled": out_integer_scaled,
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
