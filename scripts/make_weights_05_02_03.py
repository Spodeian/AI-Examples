"""Generate seed-42 golden weights for Category 05 Subfolder 02 Topic 03:
03_unet_and_multiscale_feature_pyramids.
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

    # 3. U-Net Mini Block:
    # Encoder: Conv1 (4 -> 8, 3x3, pad 1) + ReLU
    enc_w = torch.randn(8, 4, 3, 3) * 0.2
    enc_b = torch.randn(8) * 0.1
    e1 = F.relu(F.conv2d(x, enc_w, enc_b, padding=1))  # [2, 8, 8, 8]

    # MaxPool 2x2 -> [2, 8, 4, 4]
    p1 = F.max_pool2d(e1, kernel_size=2, stride=2)

    # Bottleneck: Conv (8 -> 8, 3x3, pad 1) + ReLU
    bot_w = torch.randn(8, 8, 3, 3) * 0.2
    bot_b = torch.randn(8) * 0.1
    b_feat = F.relu(F.conv2d(p1, bot_w, bot_b, padding=1))  # [2, 8, 4, 4]

    # Decoder: 2x Nearest Upsample -> [2, 8, 8, 8]
    up = F.interpolate(b_feat, scale_factor=2.0, mode="nearest")

    # Skip Concatenation: [up, e1] -> [2, 16, 8, 8]
    cat_feat = torch.cat([up, e1], dim=1)

    # Decoder Conv: (16 -> 4, 3x3, pad 1) + ReLU
    dec_w = torch.randn(4, 16, 3, 3) * 0.2
    dec_b = torch.randn(4) * 0.1
    unet_output = F.relu(F.conv2d(cat_feat, dec_w, dec_b, padding=1))  # [2, 4, 8, 8]

    # 4. BiFPN Fast Normalized Fusion:
    # Weighted fusion of x and unet_output
    bifpn_w = torch.tensor([1.2, 0.8])  # Positive fusion weights
    eps = 1e-4
    w_relu = F.relu(bifpn_w)
    norm_w = w_relu / (w_relu.sum() + eps)
    bifpn_fused = norm_w[0] * x + norm_w[1] * unet_output

    bifpn_conv_w = torch.randn(4, 4, 1, 1) * 0.3
    bifpn_conv_b = torch.randn(4) * 0.1
    bifpn_output = F.conv2d(bifpn_fused, bifpn_conv_w, bifpn_conv_b)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "input": x,
        "enc_w": enc_w,
        "enc_b": enc_b,
        "bot_w": bot_w,
        "bot_b": bot_b,
        "dec_w": dec_w,
        "dec_b": dec_b,
        "unet_output": unet_output,
        "bifpn_w": bifpn_w,
        "bifpn_conv_w": bifpn_conv_w,
        "bifpn_conv_b": bifpn_conv_b,
        "bifpn_output": bifpn_output,
    }

    out_dir = Path("weights/05_architectures/02_convolutional_and_spatiotemporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "03_unet_and_multiscale_feature_pyramids.safetensors"
    save_file(tensors, str(out_path))
    print(f"Generated seed-42 weights: {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    generate_weights()
