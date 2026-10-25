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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file


    def generate_weights() -> None:
        torch.manual_seed(42)

        # 1. Standard Linear Anchor
        test_input = torch.randn(8, 4)
        weight = torch.randn(4, 4)
        bias = torch.randn(4)
        test_output = test_input @ weight + bias

        # 2. 2D Pretrained Conv Weights & Static Image
        # C_in=4, C_out=4, K_h=3, K_w=3
        C = 4
        conv2d_w = torch.randn(C, C, 3, 3) * 0.2
        conv2d_b = torch.randn(C) * 0.1
        x_static_2d = torch.randn(2, C, 4, 4)

        # 2D Conv Output on static image:
        y_2d = F.conv2d(x_static_2d, conv2d_w, conv2d_b, padding=1)

        # 3. I3D Weight Inflation:
        # Inflate 2D kernel (4, 4, 3, 3) into 3D (4, 4, 3, 3, 3) with 1/K_T scaling (K_T=3)
        K_T = 3
        i3d_w = conv2d_w.unsqueeze(2).repeat(1, 1, K_T, 1, 1) / float(K_T)
        i3d_b = conv2d_b.clone()

        # Static video: repeat 2D frame across T=4 frames: [B=2, C=4, T=4, H=4, W=4]
        x_video = x_static_2d.unsqueeze(2).repeat(1, 1, 4, 1, 1)

        # 3D Conv Forward with temporal & spatial padding 1:
        y_3d = F.conv3d(x_video, i3d_w, i3d_b, padding=(1, 1, 1))

        # In center frames (t=1, 2), y_3d must match y_2d within numerical precision!
        center_err = torch.max(torch.abs(y_3d[:, :, 1] - y_2d)).item()
        print(f"I3D Invariance Check on static video (t=1 vs 2D): error = {center_err:.2e}")
        assert center_err < 1e-6, "I3D weight inflation invariance violated!"

        # 4. Factorized Space-Time Attention (TimeSformer)
        # Tokens: B=2, T=4, S=4, D=8
        B, T, S, D = 2, 4, 4, 8
        tokens = torch.randn(B, T, S, D)

        # Spatial Attention Weights
        spatial_q_w = torch.randn(D, D) * 0.2
        spatial_k_w = torch.randn(D, D) * 0.2
        spatial_v_w = torch.randn(D, D) * 0.2
        spatial_out_w = torch.randn(D, D) * 0.2

        # Temporal Attention Weights
        temporal_q_w = torch.randn(D, D) * 0.2
        temporal_k_w = torch.randn(D, D) * 0.2
        temporal_v_w = torch.randn(D, D) * 0.2
        temporal_out_w = torch.randn(D, D) * 0.2

        # Spatial Attention Step:
        # Flatten B and T -> [B*T, S, D]
        z_s = tokens.view(B * T, S, D)
        q_s = z_s @ spatial_q_w
        k_s = z_s @ spatial_k_w
        v_s = z_s @ spatial_v_w
        scale = (D ** 0.5)
        attn_s = F.softmax((q_s @ k_s.transpose(-2, -1)) / scale, dim=-1)
        out_s = (attn_s @ v_s) @ spatial_out_w
        z_after_s = tokens + out_s.view(B, T, S, D)

        # Temporal Attention Step:
        # Transpose to [B, S, T, D] and flatten B and S -> [B*S, T, D]
        z_t = z_after_s.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        q_t = z_t @ temporal_q_w
        k_t = z_t @ temporal_k_w
        v_t = z_t @ temporal_v_w
        attn_t = F.softmax((q_t @ k_t.transpose(-2, -1)) / scale, dim=-1)
        out_t = (attn_t @ v_t) @ temporal_out_w
        out_t_reshaped = out_t.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()
        timesformer_output = z_after_s + out_t_reshaped

        tensors = {
            "test_input": test_input,
            "weight": weight,
            "bias": bias,
            "test_output": test_output,
            "x_static_2d": x_static_2d,
            "conv2d_w": conv2d_w,
            "conv2d_b": conv2d_b,
            "x_video": x_video,
            "i3d_w": i3d_w,
            "i3d_b": i3d_b,
            "y_3d": y_3d,
            "tokens": tokens,
            "spatial_q_w": spatial_q_w,
            "spatial_k_w": spatial_k_w,
            "spatial_v_w": spatial_v_w,
            "spatial_out_w": spatial_out_w,
            "temporal_q_w": temporal_q_w,
            "temporal_k_w": temporal_k_w,
            "temporal_v_w": temporal_v_w,
            "temporal_out_w": temporal_out_w,
            "timesformer_output": timesformer_output,
        }

        out_dir = Path("weights/05_architectures/02_convolutional_and_spatiotemporal")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors"
        save_file(tensors, str(out_path))
        print(f"Generated seed-42 weights: {out_path} ({len(tensors)} tensors)")


    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_golden()
