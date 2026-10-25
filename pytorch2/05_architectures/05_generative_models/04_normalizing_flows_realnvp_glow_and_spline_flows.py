"""Normalizing Flows Realnvp Glow and Spline Flows — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows
Golden weights: weights/05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows.safetensors

``python pytorch2/05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def realnvp_affine_coupling_forward(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    s_w: torch.Tensor,
    s_b: torch.Tensor,
    t_w: torch.Tensor,
    t_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RealNVP affine coupling forward pass:
    y_a = x_a
    s = tanh(x_a @ s_w + s_b)
    t = x_a @ t_w + t_b
    y_b = x_b * exp(s) + t
    log_det = sum(s, dim=-1)
    """
    s = torch.tanh(x_a @ s_w + s_b)
    t = x_a @ t_w + t_b
    y_a = x_a
    y_b = x_b * torch.exp(s) + t
    log_det = s.sum(dim=-1)
    return y_a, y_b, log_det


def realnvp_affine_coupling_inverse(
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    s_w: torch.Tensor,
    s_b: torch.Tensor,
    t_w: torch.Tensor,
    t_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RealNVP affine coupling inverse pass:
    x_a = y_a
    s = tanh(y_a @ s_w + s_b)
    t = y_a @ t_w + t_b
    x_b = (y_b - t) * exp(-s)
    log_det = -sum(s, dim=-1)
    """
    s = torch.tanh(y_a @ s_w + s_b)
    t = y_a @ t_w + t_b
    x_a = y_a
    x_b = (y_b - t) * torch.exp(-s)
    log_det = -s.sum(dim=-1)
    return x_a, x_b, log_det


def glow_invertible_1x1_conv(
    x: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Glow invertible 1x1 linear transform:
    y = x @ w
    log_det = log|det(w)| per sample
    x_recon = y @ w^{-1}
    """
    y = x @ w
    sign, logabsdet = torch.slogdet(w)
    log_det = logabsdet.unsqueeze(0).expand(x.shape[0])
    w_inv = torch.linalg.inv(w)
    x_recon = y @ w_inv
    return y, log_det, x_recon


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 2. RealNVP Affine Coupling
    B = 4
    D_half = 3
    x_a = torch.randn(B, D_half)
    x_b = torch.randn(B, D_half)
    s_w = torch.randn(D_half, D_half) * 0.3
    s_b = torch.randn(D_half) * 0.1
    t_w = torch.randn(D_half, D_half) * 0.3
    t_b = torch.randn(D_half) * 0.1

    y_a, y_b, log_det = realnvp_affine_coupling_forward(x_a, x_b, s_w, s_b, t_w, t_b)
    x_a_rec, x_b_rec, log_det_inv = realnvp_affine_coupling_inverse(y_a, y_b, s_w, s_b, t_w, t_b)

    # 3. Glow Invertible Transform
    C = 4
    x_glow = torch.randn(B, C)
    # Generate well-conditioned invertible matrix via QR
    q, r = torch.linalg.qr(torch.randn(C, C))
    w_glow = q @ (r + torch.eye(C) * 0.5)

    y_glow, log_det_glow, x_glow_rec = glow_invertible_1x1_conv(x_glow, w_glow)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "x_a": x_a,
        "x_b": x_b,
        "s_w": s_w,
        "s_b": s_b,
        "t_w": t_w,
        "t_b": t_b,
        "y_a": y_a,
        "y_b": y_b,
        "log_det": log_det,
        "x_a_rec": x_a_rec,
        "x_b_rec": x_b_rec,
        "x_glow": x_glow,
        "w_glow": w_glow,
        "y_glow": y_glow,
        "log_det_glow": log_det_glow,
        "x_glow_rec": x_glow_rec,
    }


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous().cpu() for k, v in build_golden_tensors().items()}
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
