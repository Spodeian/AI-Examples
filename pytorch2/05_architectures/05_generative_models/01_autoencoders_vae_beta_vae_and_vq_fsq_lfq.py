"""Autoencoders: VAE, Beta-VAE, VQ, FSQ, and LFQ — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq
Golden weights: weights/05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq.safetensors

``python pytorch2/05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def vae_reparameterize_and_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    eps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterization trick and Beta-VAE analytical KL divergence.
    
    z = mu + exp(0.5 * logvar) * eps
    kl = -0.5 * beta * sum(1 + logvar - mu^2 - exp(logvar), dim=-1)
    """
    std = torch.exp(0.5 * logvar)
    z = mu + std * eps
    kl_per_sample = -0.5 * beta * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
    return z, kl_per_sample


def finite_scalar_quantization(
    z: torch.Tensor,
    half_width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite Scalar Quantization (FSQ).
    
    Clamps continuous z to [-half_width, half_width] and rounds to nearest integer.
    Uses Straight-Through Estimator (STE) for gradients.
    """
    z_clamped = torch.clamp(z, -half_width, half_width)
    z_round = torch.round(z_clamped)
    # STE: forward uses z_round, backward gradient flows to z
    z_q = z + (z_round - z).detach()
    # Integer code offset: [0, 2 * half_width]
    indices = (z_round + half_width).long()
    return z_q, indices


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    batch = 8
    in_dim = 4
    out_dim = 4
    test_in = torch.randn(batch, in_dim)
    w_linear = torch.randn(in_dim, out_dim) * 0.1
    b_linear = torch.randn(out_dim) * 0.1
    test_out = linear_anchor(test_in, w_linear, b_linear)

    # 2. VAE / Beta-VAE
    B = 2
    D_latent = 4
    beta = 2.0
    beta_tensor = torch.tensor([beta])

    mu = torch.randn(B, D_latent) * 0.5
    logvar = torch.randn(B, D_latent) * 0.2
    eps = torch.randn(B, D_latent)

    z, kl_loss = vae_reparameterize_and_kl(mu, logvar, eps, beta)

    # 3. Finite Scalar Quantization (FSQ)
    half_width = 2.0  # 5 integer levels: -2, -1, 0, 1, 2
    half_width_tensor = torch.tensor([half_width])
    z_continuous = torch.randn(B, D_latent) * 2.5

    z_q, fsq_indices = finite_scalar_quantization(z_continuous, half_width)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "mu": mu,
        "logvar": logvar,
        "eps": eps,
        "beta": beta_tensor,
        "z": z,
        "kl_loss": kl_loss,
        "half_width": half_width_tensor,
        "z_continuous": z_continuous,
        "z_q": z_q,
        "fsq_indices": fsq_indices.float(),
    }


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
