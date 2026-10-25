"""Block Floating Point BFP Microscaling MXFP and NVFP4 — PyTorch 2.x reference implementation.

Topic: 13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4
Golden weights: weights/13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4.safetensors

``python pytorch2/13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4.py`` verifies against the golden checkpoint;
``python pytorch2/13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def quantize_minifloat_core(
    x: torch.Tensor,
    e_bits: int,
    m_bits: int,
    bias: int,
    max_val: float,
) -> torch.Tensor:
    """Core elementwise minifloat quantizer."""
    x_abs = torch.abs(x)
    is_zero = (x_abs == 0.0)
    x_safe = torch.where(is_zero, torch.ones_like(x_abs), x_abs)
    
    e = torch.floor(torch.log2(x_safe))
    e_min_normal = 1.0 - float(bias)
    step = torch.where(
        e < e_min_normal,
        2.0 ** (e_min_normal - float(m_bits)),
        2.0 ** (e - float(m_bits)),
    )
    q = torch.round(x_safe / step) * step
    q = torch.clamp(q, min=0.0, max=float(max_val))
    return torch.where(is_zero, torch.zeros_like(x), torch.sign(x) * q)


def block_floating_point_quantize(
    x: torch.Tensor,
    block_size: int = 16,
    mantissa_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block Floating Point (BFP): common shared exponent per block with integer mantissa.
    
    Each block shares E_shared = floor(log2(max(|x_block|))).
    Mantissas are quantized to signed M-bit integers (-2^(M-1)+1 .. 2^(M-1)-1).
    """
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    
    # Block max magnitude
    amax = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
    is_zero = (amax == 0.0)
    amax_safe = torch.where(is_zero, torch.ones_like(amax), amax)
    
    shared_exp = torch.floor(torch.log2(amax_safe))
    shared_scale = 2.0 ** shared_exp
    
    # Normalized block elements [-1.0, 1.0] relative to shared scale
    norm_x = x_reshaped / shared_scale
    
    # Quantize to signed M-bit mantissa
    max_int = float((1 << (mantissa_bits - 1)) - 1)
    q_mantissa = torch.round(norm_x * max_int)
    q_mantissa = torch.clamp(q_mantissa, -max_int, max_int)
    
    # Dequantized values
    dequant = (q_mantissa / max_int) * shared_scale
    dequant = torch.where(is_zero, torch.zeros_like(dequant), dequant)
    
    return dequant.reshape(orig_shape), shared_scale.reshape(-1)


def ocp_mxfp4_quantize(
    x: torch.Tensor,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OCP Microscaling Specification (MXFP4):
    Block size 32 with E8M0 power-of-2 shared scale factor and FP4 E2M1 elements (max 6.0).
    """
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    
    amax = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
    is_zero = (amax == 0.0)
    amax_safe = torch.where(is_zero, torch.ones_like(amax), amax)
    
    # E8M0 shared scale: power of 2 such that amax / scale <= max_val (6.0)
    exp_scale = torch.ceil(torch.log2(amax_safe / 6.0))
    scale_block = 2.0 ** exp_scale
    scale_block = torch.where(is_zero, torch.ones_like(scale_block), scale_block)
    
    # Scale and quantize elements to FP4 E2M1
    elem_scaled = x_reshaped / scale_block
    elem_q = quantize_minifloat_core(elem_scaled, e_bits=2, m_bits=1, bias=1, max_val=6.0)
    
    dequant = elem_q * scale_block
    dequant = torch.where(is_zero, torch.zeros_like(dequant), dequant)
    
    return dequant.reshape(orig_shape), scale_block.reshape(-1)


def nvfp4_two_level_quantize(
    x: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """NVIDIA Blackwell-style two-level scaling:
    Tensor-level global scale + Block-level local scale with FP4 E2M1 elements.
    """
    global_max = float(torch.max(torch.abs(x)))
    global_scale = max(1e-8, global_max / 6.0)
    
    x_glob = x / global_scale
    dequant_local, local_scales = ocp_mxfp4_quantize(x_glob, block_size=block_size)
    dequant = dequant_local * global_scale
    
    return dequant, global_scale, local_scales


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Block Floating Point (BFP16 with 8-bit mantissa)
    bfp_input = torch.randn(4, 32) * 3.0
    bfp_dequant_ref, bfp_scales_ref = block_floating_point_quantize(bfp_input, block_size=16, mantissa_bits=8)

    # 2. OCP MXFP4 Quantization
    mx_input = torch.randn(2, 64) * 5.0
    mx_dequant_ref, mx_scales_ref = ocp_mxfp4_quantize(mx_input, block_size=32)

    # 3. NVFP4 Two-Level Scaling
    nv_input = torch.randn(4, 32) * 2.0
    nv_dequant_ref, nv_global_scale, nv_local_scales_ref = nvfp4_two_level_quantize(nv_input, block_size=16)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "bfp_input": bfp_input,
        "bfp_dequant_ref": bfp_dequant_ref,
        "bfp_scales_ref": bfp_scales_ref,
        "mx_input": mx_input,
        "mx_dequant_ref": mx_dequant_ref,
        "mx_scales_ref": mx_scales_ref,
        "nv_input": nv_input,
        "nv_dequant_ref": nv_dequant_ref,
        "nv_global_scale": torch.tensor(nv_global_scale),
        "nv_local_scales_ref": nv_local_scales_ref,
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
