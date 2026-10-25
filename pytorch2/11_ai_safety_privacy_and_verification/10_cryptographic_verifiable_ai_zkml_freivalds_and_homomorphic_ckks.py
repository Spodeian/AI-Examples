"""Cryptographic Verifiable AI ZKML Freivalds and Homomorphic CKKS — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks
Golden weights: weights/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def freivalds_verify_residual(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Freivalds' randomized matrix multiplication check: v = A(Br) - Cr in O(n^2)."""
    return a @ (b @ r) - c @ r


def ckks_degree2_activation(
    x: torch.Tensor,
    a: float,
    b: float,
    c: float,
) -> torch.Tensor:
    """CKKS / CryptoNets degree-2 polynomial activation: y = a * x^2 + b * x + c."""
    return float(a) * (x ** 2) + float(b) * x + float(c)


def ckks_degree4_activation(
    x: torch.Tensor,
    c4: float,
    c2: float,
    c0: float,
) -> torch.Tensor:
    """CKKS degree-4 polynomial activation: y = c4 * x^4 + c2 * x^2 + c0."""
    x2 = x ** 2
    return float(c4) * (x2 ** 2) + float(c2) * x2 + float(c0)


def zkml_fixed_point_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """ZKML fixed-point quantized linear layer: y = round((x @ W + b) * scale) / scale."""
    y = x @ weight + bias
    return torch.round(y * float(scale)) / float(scale)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Freivalds: M=6, K=5, N=4
    mat_a = torch.randn(6, 5)
    mat_b = torch.randn(5, 4)
    mat_c = mat_a @ mat_b
    rand_r = torch.tensor([1.0, 0.0, 1.0, 1.0])
    freivalds_valid_ref = freivalds_verify_residual(mat_a, mat_b, mat_c, rand_r)

    mat_c_tampered = mat_c.clone()
    mat_c_tampered[0, 0] += 2.0
    freivalds_tampered_ref = freivalds_verify_residual(mat_a, mat_b, mat_c_tampered, rand_r)

    # 2. CKKS degree-2: CryptoNets approximation
    x_poly = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    poly2_out_ref = ckks_degree2_activation(x_poly, a=0.125, b=0.5, c=0.25)

    # 3. CKKS degree-4: Minimax approximation
    poly4_out_ref = ckks_degree4_activation(x_poly, c4=-0.005, c2=0.2, c0=0.15)

    # 4. ZKML fixed-point linear
    zk_x = torch.randn(4, 4)
    zk_w = torch.randn(4, 4) * 0.5
    zk_b = torch.randn(4) * 0.2
    zk_out_ref = zkml_fixed_point_linear(zk_x, zk_w, zk_b, scale=16.0)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "mat_a": mat_a,
        "mat_b": mat_b,
        "mat_c": mat_c,
        "rand_r": rand_r,
        "freivalds_valid_ref": freivalds_valid_ref,
        "mat_c_tampered": mat_c_tampered,
        "freivalds_tampered_ref": freivalds_tampered_ref,
        "x_poly": x_poly,
        "poly2_out_ref": poly2_out_ref,
        "poly4_out_ref": poly4_out_ref,
        "zk_x": zk_x,
        "zk_w": zk_w,
        "zk_b": zk_b,
        "zk_out_ref": zk_out_ref,
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
