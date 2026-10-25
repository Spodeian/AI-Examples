"""Vision Optical Flow Depth and Super Resolution — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution
Golden weights: weights/07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution.safetensors

``python pytorch2/07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def pixel_shuffle(x: torch.Tensor, r: int = 2) -> torch.Tensor:
    r"""Sub-pixel convolution (PixelShuffle):
    Rearranges elements in tensor of shape [B, C * r^2, H, W] to [B, C, H * r, W * r].
    """
    b, c_r2, h, w = x.shape
    c = c_r2 // (r * r)
    x_reshaped = x.view(b, c, r, r, h, w)
    return x_reshaped.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * r, w * r)


def scale_invariant_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    r"""Scale-Invariant Logarithmic Depth Loss (Eigen et al.):
    d = log(pred + eps) - log(target + eps)
    L_si = mean(d^2) - alpha * (mean(d))^2
    pred_depth, target_depth: [B, H, W]
    """
    d = torch.log(pred_depth + eps) - torch.log(target_depth + eps) # [B, H, W]
    mean_d2 = torch.mean(d**2, dim=(-2, -1)) # [B]
    mean_d = torch.mean(d, dim=(-2, -1)) # [B]
    return mean_d2 - alpha * (mean_d**2)


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    r"""Charbonnier Loss (smooth L1 / robust photometric loss):
    L = mean(sqrt((pred - target)^2 + eps^2))
    """
    diff = pred - target
    return torch.mean(torch.sqrt(diff**2 + eps**2), dim=(-2, -1))


def raft_correlation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
) -> torch.Tensor:
    r"""RAFT All-Pairs 4D Correlation Volume (computed as spatial dot product):
    fmap1: [B, C, H, W]
    fmap2: [B, C, H, W]
    Returns correlation matrix: [B, H*W, H*W] normalized by sqrt(C).
    """
    b, c, h, w = fmap1.shape
    v1 = fmap1.view(b, c, h * w)
    v2 = fmap2.view(b, c, h * w)
    scale = 1.0 / (c**0.5)
    # [B, H*W, C] @ [B, C, H*W] -> [B, H*W, H*W]
    return (v1.transpose(1, 2) @ v2) * scale


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B = 2
    
    # 0. Linear anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)
    
    # 1. PixelShuffle (Super-Resolution)
    # B=2, C_in=4 (C_out=1, r=2), H=3, W=3 -> output [2, 1, 6, 6]
    sr_input = torch.randn(B, 4, 3, 3) * 0.5
    sr_output = pixel_shuffle(sr_input, r=2)
    
    # 2. Scale-Invariant Depth Loss
    pred_depth = torch.rand(B, 4, 4) + 0.5 # [0.5, 1.5]
    target_depth = torch.rand(B, 4, 4) + 0.5
    si_depth_ref = scale_invariant_depth_loss(pred_depth, target_depth, alpha=0.5)
    
    # 3. Charbonnier Flow Loss
    pred_flow = torch.randn(B, 4, 4) * 0.8
    target_flow = torch.randn(B, 4, 4) * 0.8
    charb_flow_ref = charbonnier_loss(pred_flow, target_flow, eps=1e-3)
    
    # 4. RAFT Correlation Matrix
    # B=2, C=4, H=2, W=2 -> [2, 4, 4]
    fmap1 = torch.randn(B, 4, 2, 2) * 0.6
    fmap2 = torch.randn(B, 4, 2, 2) * 0.6
    corr_ref = raft_correlation(fmap1, fmap2)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        
        "sr_input": sr_input,
        "sr_output": sr_output,
        
        "pred_depth": pred_depth,
        "target_depth": target_depth,
        "si_depth_ref": si_depth_ref,
        
        "pred_flow": pred_flow,
        "target_flow": target_flow,
        "charb_flow_ref": charb_flow_ref,
        
        "fmap1": fmap1,
        "fmap2": fmap2,
        "corr_ref": corr_ref,
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
