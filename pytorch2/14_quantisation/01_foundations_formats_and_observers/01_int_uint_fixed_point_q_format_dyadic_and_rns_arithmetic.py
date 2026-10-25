"""INT UINT Fixed Point Q Format Dyadic and RNS Arithmetic — PyTorch 2.x reference implementation.

Topic: 14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic
Golden weights: weights/14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic.safetensors

``python pytorch2/14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def affine_quantize_dequantize(
    x: torch.Tensor,
    num_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Asymmetric Affine Quantization: q = clamp(round(x / s) + z, 0, 2^b - 1).
    Returns (x_hat, q, scale, zero_point).
    """
    qmax = float((1 << num_bits) - 1)
    xmin = float(torch.min(x))
    xmax = float(torch.max(x))
    
    scale = max(1e-8, (xmax - xmin) / qmax)
    zero_point = float(torch.clamp(torch.round(torch.tensor(-xmin / scale)), 0.0, qmax))
    
    q = torch.clamp(torch.round(x * (1.0 / scale)) + zero_point, 0.0, qmax)
    x_hat = (q - zero_point) * scale
    return x_hat, q, scale, zero_point


def symmetric_quantize_dequantize(
    x: torch.Tensor,
    num_bits: int = 8,
    full_range: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Symmetric Quantization: q = clamp(round(x / s), qmin, qmax).
    Returns (x_hat, q, scale).
    """
    qmax = float(1 << (num_bits - 1)) - 1.0
    qmin = -float(1 << (num_bits - 1)) if full_range else -qmax
    
    amax = float(torch.max(torch.abs(x)))
    scale = max(1e-8, amax / qmax)
    
    q = torch.clamp(torch.round(x * (1.0 / scale)), qmin, qmax)
    x_hat = q * scale
    return x_hat, q, scale


def fixed_point_qmn(
    x: torch.Tensor,
    m: int = 4,
    n: int = 4,
) -> torch.Tensor:
    """Fixed-point Qm.n quantization: m integer bits (including sign) + n fractional bits."""
    scale = float(1 << n)
    qmax = float(1 << (m + n - 1)) - 1.0
    qmin = -float(1 << (m + n - 1))
    
    q = torch.clamp(torch.round(x * scale), qmin, qmax)
    x_hat = q * (1.0 / scale)
    return x_hat


def dyadic_scale_multiplier(real_multiplier: float) -> tuple[int, int]:
    """Decomposes a floating point multiplier into (M0, shift) where M ≈ M0 * 2^(-shift).
    Normalized such that M0 in [2^30, 2^31 - 1].
    """
    if real_multiplier <= 0.0:
        return 0, 0
    significand, exponent = math.frexp(real_multiplier)
    m0 = int(round(significand * (1 << 31)))
    if m0 == (1 << 31):
        m0 //= 2
        exponent += 1
    shift = 31 - exponent
    return m0, shift


def dyadic_rescale_tensor(
    acc_int32: torch.Tensor,
    m0: int,
    shift: int,
) -> torch.Tensor:
    """Simulates integer-only GEMMLOWP dyadic rescaling."""
    # (acc * M0 + 2^30) >> 31, then >> (shift - 31)
    acc_64 = acc_int32.to(torch.int64)
    scaled = (acc_64 * int(m0) + (1 << 30)) >> 31
    final_shift = shift - 31
    if final_shift > 0:
        res = (scaled + (1 << (final_shift - 1))) >> final_shift
    else:
        res = scaled << (-final_shift)
    return res.to(torch.float32)


def rns_multiply_and_crt(
    a: torch.Tensor,
    b: torch.Tensor,
    moduli: tuple[int, ...] = (7, 11, 13),
) -> torch.Tensor:
    """Residue Number System (RNS) carry-free modular multiplication with CRT reconstruction.
    Moduli must be pairwise coprime. Dynamic range M = prod(moduli) = 1001.
    """
    m_total = 1
    for m in moduli:
        m_total *= m
        
    a_int = a.to(torch.int64) % m_total
    b_int = b.to(torch.int64) % m_total
    
    # CRT precomputation
    m_primes = [m_total // m for m in moduli]
    inverses = [pow(m_primes[i], -1, moduli[i]) for i in range(len(moduli))]
    
    # RNS parallel modular multiplication
    c_components = []
    for m in moduli:
        c_i = ((a_int % m) * (b_int % m)) % m
        c_components.append(c_i)
        
    # CRT reconstruction
    reconstructed = torch.zeros_like(a_int)
    for i in range(len(moduli)):
        weight = m_primes[i] * inverses[i]
        reconstructed = reconstructed + c_components[i] * weight
        
    return (reconstructed % m_total).to(torch.float32)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Asymmetric Affine INT8 Quantization
    raw_relu = torch.relu(torch.randn(4, 8) * 3.0 + 1.0)
    affine_hat, affine_q, affine_s, affine_z = affine_quantize_dequantize(raw_relu, num_bits=8)

    # 2. Symmetric INT8 Quantization (restricted vs full)
    raw_sym = torch.randn(4, 8) * 2.5
    sym_hat_restr, sym_q_restr, sym_s_restr = symmetric_quantize_dequantize(raw_sym, num_bits=8, full_range=False)
    sym_hat_full, sym_q_full, sym_s_full = symmetric_quantize_dequantize(raw_sym, num_bits=8, full_range=True)

    # 3. Fixed-Point Q4.4
    raw_fp = torch.randn(4, 6) * 3.0
    qmn_hat = fixed_point_qmn(raw_fp, m=4, n=4)

    # 4. Dyadic Rescaling
    m0, shift = dyadic_scale_multiplier(0.0078125)  # 1/128
    acc_int = torch.randint(low=1000, high=50000, size=(4, 6), dtype=torch.int32)
    dyadic_out = dyadic_rescale_tensor(acc_int, m0, shift)

    # 5. RNS Multiplication & CRT Reconstruction
    rns_a = torch.tensor([12.0, 25.0, 4.0, 9.0])
    rns_b = torch.tensor([5.0, 3.0, 7.0, 8.0])
    rns_out = rns_multiply_and_crt(rns_a, rns_b, moduli=(7, 11, 13))

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "raw_relu": raw_relu,
        "affine_hat": affine_hat,
        "affine_q": affine_q,
        "affine_s": torch.tensor(affine_s),
        "affine_z": torch.tensor(affine_z),
        "raw_sym": raw_sym,
        "sym_hat_restr": sym_hat_restr,
        "sym_q_restr": sym_q_restr,
        "sym_s_restr": torch.tensor(sym_s_restr),
        "sym_hat_full": sym_hat_full,
        "sym_q_full": sym_q_full,
        "sym_s_full": torch.tensor(sym_s_full),
        "raw_fp": raw_fp,
        "qmn_hat": qmn_hat,
        "acc_int": acc_int.to(torch.float32),
        "dyadic_out": dyadic_out,
        "m0": torch.tensor(float(m0)),
        "shift": torch.tensor(float(shift)),
        "rns_a": rns_a,
        "rns_b": rns_b,
        "rns_out": rns_out,
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
