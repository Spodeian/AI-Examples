"""Generate seed-42 golden weights for Category 05 Subfolder 02 Topic 02:
02_specialized_convolutions_and_channel_attention.
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

    # 2. Main Input Feature Map: [B=2, C=4, H=8, W=8]
    B, C, H, W = 2, 4, 8, 8
    x = torch.randn(B, C, H, W)

    # 3. CoordConv:
    # Coordinate channels in [-1, 1]
    # y coord from -1 to 1 along H; x coord from -1 to 1 along W
    y_coords = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).repeat(B, 1, 1, W)
    x_coords = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).repeat(B, 1, H, 1)
    coord_input = torch.cat([x, x_coords, y_coords], dim=1)  # [B, 6, H, W]

    coord_conv_w = torch.randn(C, C + 2, 1, 1) * 0.2
    coord_conv_b = torch.randn(C) * 0.1
    coord_output = F.conv2d(coord_input, coord_conv_w, coord_conv_b)

    # 4. Squeeze-and-Excitation (SE):
    # GAP over H, W -> [B, C]
    gap = x.mean(dim=(2, 3))
    r_dim = C // 2  # 2
    se_fc1_w = torch.randn(r_dim, C) * 0.3
    se_fc1_b = torch.randn(r_dim) * 0.1
    se_fc2_w = torch.randn(C, r_dim) * 0.3
    se_fc2_b = torch.randn(C) * 0.1

    se_hidden = F.relu(F.linear(gap, se_fc1_w, se_fc1_b))
    se_scale = torch.sigmoid(F.linear(se_hidden, se_fc2_w, se_fc2_b))  # [B, C]
    se_output = x * se_scale.view(B, C, 1, 1)

    # 5. Efficient Channel Attention (ECA):
    # 1D conv over channel dimension: input [B, 1, C], kernel size 3, padding 1
    eca_conv_w = torch.randn(1, 1, 3) * 0.3
    eca_conv_b = torch.randn(1) * 0.1
    eca_gap = gap.unsqueeze(1)  # [B, 1, C]
    eca_conv_out = F.conv1d(eca_gap, eca_conv_w, eca_conv_b, padding=1)  # [B, 1, C]
    eca_scale = torch.sigmoid(eca_conv_out.squeeze(1))  # [B, C]
    eca_output = x * eca_scale.view(B, C, 1, 1)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "input": x,
        "coord_conv_w": coord_conv_w,
        "coord_conv_b": coord_conv_b,
        "coord_output": coord_output,
        "se_fc1_w": se_fc1_w,
        "se_fc1_b": se_fc1_b,
        "se_fc2_w": se_fc2_w,
        "se_fc2_b": se_fc2_b,
        "se_scale": se_scale,
        "se_output": se_output,
        "eca_conv_w": eca_conv_w,
        "eca_conv_b": eca_conv_b,
        "eca_scale": eca_scale,
        "eca_output": eca_output,
    }

    out_dir = Path("weights/05_architectures/02_convolutional_and_spatiotemporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "02_specialized_convolutions_and_channel_attention.safetensors"
    save_file(tensors, str(out_path))
    print(f"Generated seed-42 weights: {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    generate_weights()
