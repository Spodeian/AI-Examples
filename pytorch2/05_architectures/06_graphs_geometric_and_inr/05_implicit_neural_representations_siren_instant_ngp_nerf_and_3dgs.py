"""Implicit Neural Representations: SIREN, Instant-NGP, NeRF, and 3DGS — PyTorch 2.x reference implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs
Golden weights: weights/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.safetensors

``python pytorch2/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def siren_layer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, omega_0: float = 30.0) -> torch.Tensor:
    """Sinusoidal Representation Network (SIREN) layer: sin(omega_0 * (x @ W + b))."""
    return torch.sin(omega_0 * (x @ weight + bias))


def volume_rendering_quadrature(
    densities: torch.Tensor,
    colors: torch.Tensor,
    deltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NeRF numerical volume integration along rays using transmittance quadrature.
    
    densities has shape [B, S], colors has shape [B, S, 3], deltas has shape [B, S].
    Returns (rendered_color [B, 3], accumulated_opacity [B, 1]).
    """
    b, s = densities.shape
    tau = F.relu(densities) * deltas  # Optical depth per sample [B, S]
    alpha = 1.0 - torch.exp(-tau)     # Opacity per sample [B, S]

    # Transmittance T_i = exp(-sum_{j < i} tau_j)
    cum_tau = torch.cumsum(tau, dim=-1)
    zeros = torch.zeros(b, 1, device=densities.device, dtype=densities.dtype)
    prev_tau = torch.cat([zeros, cum_tau[:, :-1]], dim=-1)
    transmittance = torch.exp(-prev_tau)  # [B, S]

    weights = (transmittance * alpha).unsqueeze(-1)  # [B, S, 1]
    rendered_color = torch.sum(weights * colors, dim=1)  # [B, 3]
    accum_opacity = torch.sum(weights, dim=1)            # [B, 1]

    return rendered_color, accum_opacity


def gaussian_splat_2d_eval(
    x: torch.Tensor,
    mu: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    """3D Gaussian Splatting (3DGS) 2D projected Gaussian alpha evaluation:
    alpha(x) = exp(-0.5 * (x - mu) @ cov_inv @ (x - mu)^T).
    
    x has shape [B, 2], mu has shape [1, 2], cov_inv has shape [2, 2].
    Returns alpha of shape [B, 1].
    """
    delta = x - mu  # [B, 2]
    # (delta @ cov_inv) * delta
    proj = delta @ cov_inv
    mahalanobis = torch.sum(proj * delta, dim=-1, keepdim=True)
    return torch.exp(-0.5 * mahalanobis)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # SIREN setup: 3D coordinate inputs (x, y, z)
    x_siren = torch.randn(4, 3)
    w_siren = torch.randn(3, 8) * 0.2
    b_siren = torch.randn(8) * 0.1
    h_siren = siren_layer(x_siren, w_siren, b_siren, omega_0=30.0)

    # NeRF Volume rendering: B=2 rays, S=4 step samples, 3 color channels
    densities = torch.abs(torch.randn(2, 4))
    colors = torch.sigmoid(torch.randn(2, 4, 3))
    deltas = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.2, 0.3, 0.2, 0.3]], dtype=torch.float32)
    rendered_color, accum_opacity = volume_rendering_quadrature(densities, colors, deltas)

    # 3DGS: 4 pixel coordinates, 1 Gaussian center, symmetric positive-definite inverse covariance
    x_pixels = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.5],
        [-0.5, 0.2],
        [1.0, -1.0],
    ], dtype=torch.float32)
    mu = torch.tensor([[0.1, -0.1]], dtype=torch.float32)
    cov_inv = torch.tensor([
        [2.0, 0.5],
        [0.5, 2.0],
    ], dtype=torch.float32)
    splat_alpha = gaussian_splat_2d_eval(x_pixels, mu, cov_inv)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_siren": x_siren,
        "w_siren": w_siren,
        "b_siren": b_siren,
        "h_siren": h_siren,
        "densities": densities,
        "colors": colors,
        "deltas": deltas,
        "rendered_color": rendered_color,
        "accum_opacity": accum_opacity,
        "x_pixels": x_pixels,
        "mu": mu,
        "cov_inv": cov_inv,
        "splat_alpha": splat_alpha,
    }
    return tensors


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
