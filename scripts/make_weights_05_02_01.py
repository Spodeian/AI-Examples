#!/usr/bin/env python3
"""Generate golden weights and inputs/outputs for Category 05 Subfolder 02 Topic 01:
Classic to Modern CNNs: ResNet v1/v2, DenseNet, MobileNetV2, and ConvNeXt-V2 (GRN).
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


def global_response_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Global Response Normalization (GRN) for ConvNeXt-V2.
    x: (B, C, H, W)
    gamma, beta: (C,)
    """
    # L2 norm across spatial dimensions: (B, C, 1, 1)
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    # Mean across channel dimension: (B, 1, 1, 1)
    nx = gx / (gx.mean(dim=1, keepdim=True) + eps)
    # Feature calibration and residual add:
    return gamma.view(1, -1, 1, 1) * (x * nx) + beta.view(1, -1, 1, 1) + x


def main():
    torch.manual_seed(42)

    batch_size = 2
    in_channels = 4
    height = 8
    width = 8
    hidden_dim = 16

    # Input tensor: (B, C, H, W)
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32)

    # ==============================================================
    # 1. ConvNeXt-V2 Block Parameters (7x7 Depthwise + GRN + Inverted Bottleneck)
    # ==============================================================
    dw_w = torch.randn(in_channels, 1, 7, 7, dtype=torch.float32) * 0.1
    dw_b = torch.randn(in_channels, dtype=torch.float32) * 0.05

    pw1_w = torch.randn(hidden_dim, in_channels, 1, 1, dtype=torch.float32) * 0.2
    pw1_b = torch.randn(hidden_dim, dtype=torch.float32) * 0.05

    grn_gamma = torch.randn(hidden_dim, dtype=torch.float32) * 0.1 + 1.0
    grn_beta = torch.randn(hidden_dim, dtype=torch.float32) * 0.05

    pw2_w = torch.randn(in_channels, hidden_dim, 1, 1, dtype=torch.float32) * 0.2
    pw2_b = torch.randn(in_channels, dtype=torch.float32) * 0.05

    # ConvNeXt-V2 forward pass
    # 1. Depthwise 7x7 conv with padding 3
    h_dw = F.conv2d(x, dw_w, dw_b, padding=3, groups=in_channels)
    # 2. Pointwise expansion 1x1
    h_pw1 = F.conv2d(h_dw, pw1_w, pw1_b)
    # 3. GELU
    h_act = F.gelu(h_pw1)
    # 4. Global Response Normalization (GRN)
    h_grn = global_response_norm(h_act, grn_gamma, grn_beta)
    # 5. Pointwise projection back to in_channels
    h_pw2 = F.conv2d(h_grn, pw2_w, pw2_b)
    # 6. Residual shortcut
    convnext_output = x + h_pw2

    # ==============================================================
    # 2. Pre-Activation ResNet Block Parameters
    # ==============================================================
    res_w1 = torch.randn(in_channels, in_channels, 3, 3, dtype=torch.float32) * 0.15
    res_b1 = torch.randn(in_channels, dtype=torch.float32) * 0.05
    res_w2 = torch.randn(in_channels, in_channels, 3, 3, dtype=torch.float32) * 0.15
    res_b2 = torch.randn(in_channels, dtype=torch.float32) * 0.05

    # Pre-Activation ResNet forward (ReLU -> Conv -> ReLU -> Conv -> (+))
    r1 = F.relu(x)
    r2 = F.conv2d(r1, res_w1, res_b1, padding=1)
    r3 = F.relu(r2)
    r4 = F.conv2d(r3, res_w2, res_b2, padding=1)
    resnet_output = x + r4

    # Standard linear anchor for cross-track parity verification
    D_core = 4
    test_input = torch.randn(8, D_core)
    weight = torch.randn(D_core, D_core)
    bias = torch.randn(D_core)
    test_output = test_input @ weight + bias

    # Output directory
    out_dir = Path("weights/05_architectures/02_convolutional_and_spatiotemporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "01_classic_to_modern_cnns_resnet_densenet_convnext.safetensors"

    tensors = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "input": x,
        "dw_w": dw_w,
        "dw_b": dw_b,
        "pw1_w": pw1_w,
        "pw1_b": pw1_b,
        "grn_gamma": grn_gamma,
        "grn_beta": grn_beta,
        "pw2_w": pw2_w,
        "pw2_b": pw2_b,
        "convnext_output": convnext_output,
        "res_w1": res_w1,
        "res_b1": res_b1,
        "res_w2": res_w2,
        "res_b2": res_b2,
        "resnet_output": resnet_output,
    }

    save_file(tensors, str(out_file))
    print(f"Generated {out_file} successfully.")
    print(f"convnext_output shape: {convnext_output.shape}, norm: {convnext_output.norm():.6f}")
    print(f"resnet_output shape: {resnet_output.shape}, norm: {resnet_output.norm():.6f}")


if __name__ == "__main__":
    main()
