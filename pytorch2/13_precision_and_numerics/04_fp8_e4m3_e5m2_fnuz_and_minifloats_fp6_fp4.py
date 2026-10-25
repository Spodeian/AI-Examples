"""FP8 E4M3 E5M2 FNUZ and Minifloats FP6 FP4 — PyTorch 2.x reference implementation.

Topic: 13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4
Golden weights: weights/13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4.safetensors

``python pytorch2/13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4.py`` verifies against the golden checkpoint;
``python pytorch2/13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def quantize_minifloat(
    x: torch.Tensor,
    e_bits: int,
    m_bits: int,
    bias: int,
    max_val: float,
) -> torch.Tensor:
    """Simulates arbitrary IEEE/OCP minifloat quantisation with subnormals and clamping."""
    x_abs = torch.abs(x)
    is_zero = (x_abs == 0.0)
    x_safe = torch.where(is_zero, torch.ones_like(x_abs), x_abs)
    
    e = torch.floor(torch.log2(x_safe))
    e_min_normal = 1.0 - float(bias)
    
    # Subnormal step vs normal step
    step = torch.where(
        e < e_min_normal,
        2.0 ** (e_min_normal - float(m_bits)),
        2.0 ** (e - float(m_bits)),
    )
    
    q = torch.round(x_safe / step) * step
    q = torch.clamp(q, min=0.0, max=float(max_val))
    return torch.where(is_zero, torch.zeros_like(x), torch.sign(x) * q)


def fp8_e4m3fn_quantize(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Quantize to FP8 E4M3FN (1 sign, 4 exp bias 7, 3 mantissa, max 448.0)."""
    scaled = x * (1.0 / float(scale))
    q = quantize_minifloat(scaled, e_bits=4, m_bits=3, bias=7, max_val=448.0)
    return q * float(scale)


def fp8_e5m2_quantize(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Quantize to FP8 E5M2 (1 sign, 5 exp bias 15, 2 mantissa, max 57344.0)."""
    scaled = x * (1.0 / float(scale))
    q = quantize_minifloat(scaled, e_bits=5, m_bits=2, bias=15, max_val=57344.0)
    return q * float(scale)


def fp4_e2m1_quantize(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Quantize to OCP FP4 E2M1 (1 sign, 2 exp bias 1, 1 mantissa, max 6.0)."""
    scaled = x * (1.0 / float(scale))
    q = quantize_minifloat(scaled, e_bits=2, m_bits=1, bias=1, max_val=6.0)
    return q * float(scale)


def delayed_scaled_fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: float,
    b_scale: float,
) -> tuple[torch.Tensor, float, float]:
    """Hopper-style FP8 GEMM with delayed scale calculation for next step."""
    a_q = fp8_e4m3fn_quantize(a, scale=a_scale)
    b_q = fp8_e4m3fn_quantize(b, scale=b_scale)
    out = a_q @ b_q
    
    # Delayed scaling: calculate next scale factor from amax
    a_max = float(torch.max(torch.abs(a)))
    b_max = float(torch.max(torch.abs(b)))
    next_a_scale = max(1e-6, a_max / 448.0)
    next_b_scale = max(1e-6, b_max / 448.0)
    return out, next_a_scale, next_b_scale


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. FP8 E4M3FN Quantization
    fp8_input = torch.randn(4, 6) * 2.5
    e4m3_ref = fp8_e4m3fn_quantize(fp8_input, scale=0.5)

    # 2. FP8 E5M2 Quantization
    e5m2_ref = fp8_e5m2_quantize(fp8_input, scale=0.5)

    # 3. FP4 E2M1 Quantization
    fp4_input = torch.randn(4, 6) * 1.5
    fp4_ref = fp4_e2m1_quantize(fp4_input, scale=0.25)

    # 4. Delayed Scaled FP8 GEMM
    gemm_a = torch.randn(4, 8) * 1.2
    gemm_b = torch.randn(8, 6) * 0.8
    gemm_out_ref, next_scale_a, next_scale_b = delayed_scaled_fp8_gemm(
        gemm_a, gemm_b, a_scale=0.01, b_scale=0.01
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "fp8_input": fp8_input,
        "e4m3_ref": e4m3_ref,
        "e5m2_ref": e5m2_ref,
        "fp4_input": fp4_input,
        "fp4_ref": fp4_ref,
        "gemm_a": gemm_a,
        "gemm_b": gemm_b,
        "gemm_out_ref": gemm_out_ref,
        "next_scale_a": torch.tensor(next_scale_a),
        "next_scale_b": torch.tensor(next_scale_b),
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
