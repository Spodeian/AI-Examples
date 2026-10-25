"""Diffusion DDPM Ddim Edm and DiT — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit
Golden weights: weights/05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit.safetensors

``python pytorch2/05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ddpm_forward_diffuse(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """DDPM forward diffusion: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise."""
    return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * noise


def ddim_step(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    alpha_bar_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic DDIM (eta=0) step:
    x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    x_prev = sqrt(alpha_bar_prev) * x_0_pred + sqrt(1 - alpha_bar_prev) * eps_pred
    """
    sqrt_ab_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)
    x_0_pred = (x_t - sqrt_one_minus_ab_t * eps_pred) / sqrt_ab_t

    sqrt_ab_prev = torch.sqrt(alpha_bar_prev)
    sqrt_one_minus_ab_prev = torch.sqrt(1.0 - alpha_bar_prev)
    x_prev = sqrt_ab_prev * x_0_pred + sqrt_one_minus_ab_prev * eps_pred
    return x_0_pred, x_prev


def classifier_free_guidance(
    eps_uncond: torch.Tensor,
    eps_cond: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """Classifier-Free Guidance (CFG): eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)."""
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


def adaln_zero_modulation(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """DiT adaLN-Zero modulation:
    h = LayerNorm(x) * (1 + gamma) + beta
    y = alpha * h
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    norm = (x - mean) / torch.sqrt(var + eps)
    h = norm * (1.0 + gamma) + beta
    return alpha * h


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 2. DDPM Forward Diffusion
    B = 4
    D = 4
    x_0 = torch.randn(B, D)
    noise = torch.randn(B, D)
    alpha_bar_t = torch.tensor([0.7])
    x_t = ddpm_forward_diffuse(x_0, noise, alpha_bar_t)

    # 3. DDIM Inversion & Step
    alpha_bar_prev = torch.tensor([0.85])
    eps_pred = torch.randn(B, D) * 0.5
    x_0_pred, x_prev = ddim_step(x_t, eps_pred, alpha_bar_t, alpha_bar_prev)

    # 4. Classifier-Free Guidance (CFG)
    eps_uncond = torch.randn(B, D)
    eps_cond = torch.randn(B, D)
    cfg_scale = 3.0
    cfg_out = classifier_free_guidance(eps_uncond, eps_cond, cfg_scale)

    # 5. DiT adaLN-Zero Modulation
    x_dit = torch.randn(B, D)
    gamma = torch.randn(B, D) * 0.1
    beta_dit = torch.randn(B, D) * 0.1
    alpha_dit = torch.randn(B, D) * 0.05
    adaln_out = adaln_zero_modulation(x_dit, gamma, beta_dit, alpha_dit)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "x_0": x_0,
        "noise": noise,
        "alpha_bar_t": alpha_bar_t,
        "x_t": x_t,
        "alpha_bar_prev": alpha_bar_prev,
        "eps_pred": eps_pred,
        "x_0_pred": x_0_pred,
        "x_prev": x_prev,
        "eps_uncond": eps_uncond,
        "eps_cond": eps_cond,
        "cfg_scale": torch.tensor([cfg_scale]),
        "cfg_out": cfg_out,
        "x_dit": x_dit,
        "gamma": gamma,
        "beta_dit": beta_dit,
        "alpha_dit": alpha_dit,
        "adaln_out": adaln_out,
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
