"""Generate seed-42 golden weights for Category 05 Subfolder 02 Topic 04:
04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.
"""

from __future__ import annotations
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


if __name__ == "__main__":
    generate_weights()
