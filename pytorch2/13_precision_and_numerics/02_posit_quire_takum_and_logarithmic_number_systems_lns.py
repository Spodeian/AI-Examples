"""Posit Quire Takum and Logarithmic Number Systems LNS — PyTorch 2.x reference implementation.

Topic: 13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns
Golden weights: weights/13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns.safetensors

``python pytorch2/13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns.py`` verifies against the golden checkpoint;
``python pytorch2/13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def posit_tapered_quantize(x: torch.Tensor, n_bits: int = 8, es: int = 1) -> torch.Tensor:
    """Simulates Posit<N, es> tapered precision arithmetic:
    Regime k = floor(log2(|x|) / 2^es).
    Available fraction bits taper away from 1.0.
    """
    is_zero = (x == 0.0)
    x_safe = torch.where(is_zero, torch.ones_like(x), torch.abs(x))
    log2_x = torch.floor(torch.log2(x_safe))
    useed = 2.0 ** float(es)
    k = torch.floor(log2_x / useed)
    regime_bits = torch.where(k >= 0.0, k + 2.0, -k + 1.0)
    frac_bits = torch.clamp(float(n_bits) - 1.0 - regime_bits - float(es), min=0.0, max=float(n_bits - 2))
    step = 2.0 ** (log2_x - frac_bits)
    q = torch.round(x_safe / step) * step
    out = torch.sign(x) * q
    return torch.where(is_zero, torch.zeros_like(x), out)


def lns_multiplication(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Logarithmic Number System (LNS) multiplication:
    sign_out = sign(a) * sign(b)
    L_out = log2(|a|) + log2(|b|)
    out = sign_out * 2^(L_out)
    """
    sign_out = torch.sign(a) * torch.sign(b)
    log_a = torch.log2(torch.abs(a) + float(eps))
    log_b = torch.log2(torch.abs(b) + float(eps))
    log_out = log_a + log_b
    return sign_out * (2.0 ** log_out)


def lns_gaussian_addition(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Gaussian Log-Addition in LNS for positive quantities:
    L_out = max(L_a, L_b) + log2(1 + 2^(-|L_a - L_b|))
    """
    log_a = torch.log2(torch.clamp(a, min=float(eps)))
    log_b = torch.log2(torch.clamp(b, min=float(eps)))
    diff = torch.abs(log_a - log_b)
    correction = torch.log2(1.0 + (2.0 ** (-diff)))
    log_out = torch.maximum(log_a, log_b) + correction
    return 2.0 ** log_out


def quire_dot_product_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simulates Kulisch Quire exact accumulator by computing full-precision sum of products."""
    return torch.sum(a.double() * b.double(), dim=-1).float()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Posit-8 Tapered Quantization
    posit_in = torch.tensor([-2.5, 0.0, 0.35, 1.0, 1.45, 3.8, -0.75, 0.125])
    posit_ref = posit_tapered_quantize(posit_in, n_bits=8, es=1)

    # 2. LNS Multiplication
    lns_a = torch.tensor([1.5, -2.0, 0.5, -0.25])
    lns_b = torch.tensor([4.0, 3.0, -2.0, -8.0])
    lns_mul_ref = lns_multiplication(lns_a, lns_b)

    # 3. LNS Gaussian Log-Addition
    add_a = torch.tensor([1.0, 2.5, 0.25, 4.0])
    add_b = torch.tensor([2.0, 1.25, 0.75, 3.0])
    lns_add_ref = lns_gaussian_addition(add_a, add_b)

    # 4. Quire Dot Product: Matrix (4, 6) x (6,)
    quire_mat = torch.randn(4, 6)
    quire_vec = torch.randn(6)
    quire_ref = quire_dot_product_sim(quire_mat, quire_vec)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "posit_in": posit_in,
        "posit_ref": posit_ref,
        "lns_a": lns_a,
        "lns_b": lns_b,
        "lns_mul_ref": lns_mul_ref,
        "add_a": add_a,
        "add_b": add_b,
        "lns_add_ref": lns_add_ref,
        "quire_mat": quire_mat,
        "quire_vec": quire_vec,
        "quire_ref": quire_ref,
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
