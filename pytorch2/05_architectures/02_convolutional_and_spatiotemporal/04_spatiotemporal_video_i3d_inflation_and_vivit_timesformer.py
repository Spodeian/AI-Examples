"""PyTorch 2.x implementation for Category 05 Subfolder 02 Topic 04:
Spatiotemporal Video: I3D Weight Inflation & TimeSformer Factorized Space-Time Attention
in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def i3d_inflate_weights(conv2d_w: torch.Tensor, k_t: int = 3) -> torch.Tensor:
    """Inflates a 2D convolutional filter into a 3D spatiotemporal filter.
    
    Args:
        conv2d_w: Tensor of shape (C_out, C_in, K_h, K_w)
        k_t: Temporal kernel size
        
    Returns:
        Tensor of shape (C_out, C_in, K_t, K_h, K_w) with 1/k_t energy scaling.
    """
    return conv2d_w.unsqueeze(2).repeat(1, 1, k_t, 1, 1) / float(k_t)


class FactorizedSpaceTimeAttention(nn.Module):
    """TimeSformer / ViViT Factorized Space-Time Attention Block.
    
    Applies spatial self-attention across tokens S, followed by temporal self-attention
    across frames T, reducing full spacetime attention complexity from O((T*S)^2) to O(T*S^2 + S*T^2).
    """

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.scale = dim ** 0.5

        # Spatial projections
        self.spatial_q = nn.Linear(dim, dim, bias=False)
        self.spatial_k = nn.Linear(dim, dim, bias=False)
        self.spatial_v = nn.Linear(dim, dim, bias=False)
        self.spatial_out = nn.Linear(dim, dim, bias=False)

        # Temporal projections
        self.temporal_q = nn.Linear(dim, dim, bias=False)
        self.temporal_k = nn.Linear(dim, dim, bias=False)
        self.temporal_v = nn.Linear(dim, dim, bias=False)
        self.temporal_out = nn.Linear(dim, dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Input tokens of shape (B, T, S, D)
            
        Returns:
            Output tokens of shape (B, T, S, D)
        """
        B, T, S, D = z.shape

        # 1. Spatial Attention across S tokens
        z_s = z.view(B * T, S, D)
        q_s = self.spatial_q(z_s)
        k_s = self.spatial_k(z_s)
        v_s = self.spatial_v(z_s)
        attn_s = F.softmax((q_s @ k_s.transpose(-2, -1)) / self.scale, dim=-1)
        out_s = self.spatial_out(attn_s @ v_s).view(B, T, S, D)
        z_mid = z + out_s

        # 2. Temporal Attention across T frames
        z_t = z_mid.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        q_t = self.temporal_q(z_t)
        k_t = self.temporal_k(z_t)
        v_t = self.temporal_v(z_t)
        attn_t = F.softmax((q_t @ k_t.transpose(-2, -1)) / self.scale, dim=-1)
        out_t = self.temporal_out(attn_t @ v_t).view(B, S, T, D).permute(0, 2, 1, 3).contiguous()

        return z_mid + out_t


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def verify_against_golden() -> None:
    """Verifies PyTorch implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = torch.max(torch.abs(test_out - weights["test_output"])).item()
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    # 2. I3D Weight Inflation verification
    inflated_w = i3d_inflate_weights(weights["conv2d_w"], k_t=3)
    err_inflate = torch.max(torch.abs(inflated_w - weights["i3d_w"])).item()
    assert err_inflate < 1e-6, f"I3D inflation weight error: {err_inflate}"

    # 3. 3D Conv Forward verification
    out_3d = F.conv3d(weights["x_video"], weights["i3d_w"], weights["i3d_b"], padding=(1, 1, 1))
    err_3d = torch.max(torch.abs(out_3d - weights["y_3d"])).item()
    assert err_3d < 1e-6, f"3D conv forward error: {err_3d}"

    # 4. Factorized Space-Time Attention (TimeSformer) verification
    timesformer = FactorizedSpaceTimeAttention(dim=8)
    with torch.no_grad():
        timesformer.spatial_q.weight.copy_(weights["spatial_q_w"].t())
        timesformer.spatial_k.weight.copy_(weights["spatial_k_w"].t())
        timesformer.spatial_v.weight.copy_(weights["spatial_v_w"].t())
        timesformer.spatial_out.weight.copy_(weights["spatial_out_w"].t())
        timesformer.temporal_q.weight.copy_(weights["temporal_q_w"].t())
        timesformer.temporal_k.weight.copy_(weights["temporal_k_w"].t())
        timesformer.temporal_v.weight.copy_(weights["temporal_v_w"].t())
        timesformer.temporal_out.weight.copy_(weights["temporal_out_w"].t())

    out_timesformer = timesformer(weights["tokens"])
    err_tf = torch.max(torch.abs(out_timesformer - weights["timesformer_output"])).item()
    assert err_tf < 1e-6, f"TimeSformer parity error {err_tf:.2e} exceeds 1e-6"

    print("PyTorch 2.x I3D & TimeSformer parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
