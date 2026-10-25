"""Visual Autoregressive MaskGIT, VAR, and MAR — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar
Golden weights: weights/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.safetensors

``python pytorch2/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def maskgit_cosine_schedule(r: torch.Tensor) -> torch.Tensor:
    """MaskGIT cosine masking schedule: gamma(r) = cos(pi / 2 * r) for r in [0, 1]."""
    return torch.cos(0.5 * math.pi * r)


def maskgit_apply_mask(x: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """Applies binary mask M in {0, 1} to replace masked token positions with mask_token.
    
    x has shape [B, L, D], mask has shape [B, L, 1], mask_token has shape [1, 1, D].
    """
    return (1.0 - mask) * x + mask * mask_token


def var_coarse_to_fine_upsample(
    coarse_map: torch.Tensor,
    repeats: int,
    w_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """VAR next-scale autoregressive expansion.
    
    Repeats coarse map spatially and applies scale modulation residual.
    coarse_map has shape [B, L_c, D], output has shape [B, L_c * repeats, D].
    """
    # Repeat each coarse token across repeats spatial sub-patches
    upsampled = coarse_map.repeat_interleave(repeats, dim=1)
    residual = F.gelu(upsampled @ w_scale + b_scale)
    return upsampled + residual


def mar_masked_diffusion_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked Autoregressive (MAR) diffusion loss over masked positions only.
    
    v_pred and v_target have shape [B, L, D], mask has shape [B, L, 1].
    """
    sq_err = (v_pred - v_target) ** 2
    masked_sq_err = sq_err * mask
    loss = torch.sum(masked_sq_err) / (torch.sum(mask) * v_pred.shape[-1] + 1e-6)
    return loss.reshape(1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Topic setup: B=2, L=4, D=8
    B, L, D = 2, 4, 8
    x_tokens = torch.randn(B, L, D)
    mask_token = torch.randn(1, 1, D) * 0.5
    mask = torch.tensor([
        [[0.0], [1.0], [0.0], [1.0]],
        [[1.0], [0.0], [1.0], [0.0]],
    ], dtype=torch.float32)

    r = torch.tensor([[0.0], [0.33], [0.66], [1.0]], dtype=torch.float32)
    gamma = maskgit_cosine_schedule(r)
    masked_tokens = maskgit_apply_mask(x_tokens, mask, mask_token)

    # VAR coarse-to-fine expansion: coarse L=1 -> fine L=4
    coarse_map = torch.randn(B, 1, D)
    w_scale = torch.randn(D, D) * 0.2
    b_scale = torch.randn(D) * 0.1
    fine_map = var_coarse_to_fine_upsample(coarse_map, repeats=4, w_scale=w_scale, b_scale=b_scale)

    # MAR diffusion head
    v_pred = torch.randn(B, L, D)
    v_target = torch.randn(B, L, D)
    mar_loss = mar_masked_diffusion_loss(v_pred, v_target, mask)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "r": r,
        "gamma": gamma,
        "x_tokens": x_tokens,
        "mask": mask,
        "mask_token": mask_token,
        "masked_tokens": masked_tokens,
        "coarse_map": coarse_map,
        "w_scale": w_scale,
        "b_scale": b_scale,
        "fine_map": fine_map,
        "v_pred": v_pred,
        "v_target": v_target,
        "mar_loss": mar_loss,
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
