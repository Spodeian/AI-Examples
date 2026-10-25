"""Rotation and Incoherence PTQ: QuaRot, SpinQuant, and QuIP# — PyTorch 2.x reference implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp
Golden weights: weights/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.safetensors

``python pytorch2/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def construct_hadamard_matrix(dim: int) -> torch.Tensor:
    """Recursively constructs a normalized Walsh-Hadamard matrix of size dim = 2^k."""
    if dim == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    h_sub = construct_hadamard_matrix(dim // 2)
    factor = 1.0 / (2.0 ** 0.5)
    top = torch.cat([h_sub, h_sub], dim=1)
    bottom = torch.cat([h_sub, -h_sub], dim=1)
    return factor * torch.cat([top, bottom], dim=0)


def construct_randomized_hadamard(dim: int, signs: torch.Tensor) -> torch.Tensor:
    """Constructs orthogonal randomized Hadamard matrix: Q = H_dim @ diag(signs)."""
    h = construct_hadamard_matrix(dim)
    return h * signs.unsqueeze(0)


def cayley_orthogonal_rotation(skew_matrix: torch.Tensor) -> torch.Tensor:
    """Computes orthogonal matrix via Cayley transform: R = (I - A)(I + A)^{-1} for skew A = -A^T."""
    dim = skew_matrix.shape[0]
    eye = torch.eye(dim, dtype=skew_matrix.dtype, device=skew_matrix.device)
    a = 0.5 * (skew_matrix - skew_matrix.T)  # enforce strict skew-symmetry
    r = torch.inverse(eye + a) @ (eye - a)
    return r


def quantize_per_token_int4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token INT4 quantization [-8, 7]."""
    scale = torch.max(torch.abs(x), dim=-1, keepdim=True).values / 7.0
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(x / scale), -8.0, 7.0)
    return q, scale


def quantize_per_column_int4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-column INT4 quantization [-8, 7]."""
    scale = torch.max(torch.abs(w), dim=0, keepdim=True).values / 7.0
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(w / scale), -8.0, 7.0)
    return q, scale


def w4a4_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Simulated W4A4 GEMM with per-token activation and per-column weight quantization."""
    q_x, s_x = quantize_per_token_int4(x)
    q_w, s_w = quantize_per_column_int4(weight)
    return (q_x * s_x) @ (q_w * s_w)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    dim = 16
    d_out = 8
    b = 8
    x_calib = torch.randn(b, dim)
    # Channel 0 and 5 have massive activation outliers (16x and 10x)
    x_calib[:, 0] *= 16.0
    x_calib[:, 5] *= 10.0

    w_orig = torch.randn(dim, d_out) * 0.2
    b_orig = torch.randn(d_out) * 0.05

    # 1. Hadamard matrix & Randomized Hadamard rotation
    hadamard_16 = construct_hadamard_matrix(dim)
    # Deterministic Rademacher signs
    signs = torch.where(torch.randn(dim) >= 0.0, torch.tensor(1.0), torch.tensor(-1.0))
    q_hadamard = construct_randomized_hadamard(dim, signs)

    # 2. Cayley transform rotation
    skew_raw = torch.randn(dim, dim) * 0.1
    r_cayley = cayley_orthogonal_rotation(skew_raw)

    # 3. QuaRot Hadamard Rotation & W4A4 GEMM
    # Rotated activations and weights
    x_rot = x_calib @ q_hadamard
    w_rot = q_hadamard.T @ w_orig
    # Exact mathematical equivalence before quantization
    out_exact_rot = x_rot @ w_rot + b_orig

    # W4A4 GEMM on rotated space
    out_quarot_w4a4 = w4a4_gemm(x_rot, w_rot) + b_orig

    # W4A4 GEMM on unrotated space (severely degrades under outliers)
    out_unrotated_w4a4 = w4a4_gemm(x_calib, w_orig) + b_orig

    # FP32 ground truth
    out_fp32 = x_calib @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "hadamard_16": hadamard_16,
        "signs": signs,
        "q_hadamard": q_hadamard,
        "r_cayley": r_cayley,
        "x_rot": x_rot,
        "w_rot": w_rot,
        "out_unrotated_w4a4": out_unrotated_w4a4,
        "out_quarot_w4a4": out_quarot_w4a4,
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
