"""Vision Segmentation UNet Mask2Former and SAM Promptable — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable
Golden weights: weights/07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable.safetensors

``python pytorch2/07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dice_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    r"""Soft Dice Loss:
    Dice = (2 * sum(p * t) + eps) / (sum(p^2) + sum(t^2) + eps)
    Loss = 1 - Dice
    pred_mask: [B, H, W] (probabilities in [0, 1])
    target_mask: [B, H, W] (binary {0, 1})
    """
    intersection = torch.sum(pred_mask * target_mask, dim=(-2, -1))
    cardinality = torch.sum(pred_mask**2, dim=(-2, -1)) + torch.sum(target_mask**2, dim=(-2, -1))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice


def mask2former_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    r"""Mask2Former Masked Cross-Attention:
    Attention logits = Q @ K^T / sqrt(D) + attn_mask
    where attn_mask = 0 if binarized foreground, -inf (or -1e4) if background.
    q: [B, N_q, D]
    k: [B, S, D]
    v: [B, S, D]
    attn_mask: [B, N_q, S] (values 0.0 or -1e4)
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d) # [B, N_q, S]
    masked_scores = scores + attn_mask
    probs = torch.softmax(masked_scores, dim=-1)
    out = probs @ v # [B, N_q, D]
    return out


def sam_fourier_prompt_embed(
    points: torch.Tensor,
    gaussian_matrix: torch.Tensor,
) -> torch.Tensor:
    r"""SAM (Segment Anything Model) Positional Prompt Encoding via Random Gaussian Fourier Features:
    gamma(p) = [sin(2 * pi * p @ B^T), cos(2 * pi * p @ B^T)]
    points: [B, N_p, 2] in normalized coordinates [0, 1]
    gaussian_matrix: [D_feat, 2]
    Output: [B, N_p, 2 * D_feat]
    """
    # 2 * pi * points @ B^T
    proj = 2.0 * math.pi * (points @ gaussian_matrix.t()) # [B, N_p, D_feat]
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B, H, W = 2, 4, 4
    N_q, S, D = 3, 16, 8
    N_p, D_fourier = 2, 4
    
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. Soft Dice Loss
    logits = torch.randn(B, H, W)
    pred_mask = torch.sigmoid(logits)
    target_mask = (torch.rand(B, H, W) > 0.5).float()
    dice_loss_ref = dice_loss(pred_mask, target_mask)
    
    # 2. Mask2Former Masked Attention
    q = torch.randn(B, N_q, D) * 0.5
    k = torch.randn(B, S, D) * 0.5
    v = torch.randn(B, S, D) * 0.5
    
    # Binary attention mask: 0 for foreground, -1e4 for background
    binary_bg = (torch.rand(B, N_q, S) > 0.6).float() # 1 is bg
    attn_mask = binary_bg * (-1e4)
    m2f_out = mask2former_masked_attention(q, k, v, attn_mask)
    
    # 3. SAM Fourier Prompt Encoder
    points = torch.rand(B, N_p, 2)
    gaussian_matrix = torch.randn(D_fourier, 2) * 1.2
    sam_prompt_embed = sam_fourier_prompt_embed(points, gaussian_matrix)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        
        "pred_mask": pred_mask,
        "target_mask": target_mask,
        "dice_loss_ref": dice_loss_ref,
        
        "q": q,
        "k": k,
        "v": v,
        "attn_mask": attn_mask,
        "m2f_out": m2f_out,
        
        "points": points,
        "gaussian_matrix": gaussian_matrix,
        "sam_prompt_embed": sam_prompt_embed,
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
