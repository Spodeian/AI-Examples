"""PyTorch 2.x implementation for Category 05 Subfolder 02 Topic 01:
Classic to Modern CNNs: ResNet v1/v2, DenseNet, MobileNetV2, and ConvNeXt-V2 (GRN)
in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization (GRN) Layer for ConvNeXt-V2.
    
    Calibrates channel-wise feature competition to eliminate feature collapse:
    G_c = ||X_c||_2 = sqrt(sum_{h, w} X_{c, h, w}^2 + eps)
    N_c = G_c / (mean_j G_j + eps)
    X' = gamma * (X * N) + beta + X
    
    References:
        - Woo et al. (2023): ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, C, H, W)
            
        Returns:
            Normalized tensor of shape (B, C, H, W)
        """
        # L2 norm along spatial dimensions: (B, C, 1, 1)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        # Relative channel contrast:
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        # Feature calibration and residual skip:
        return self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(1, -1, 1, 1) + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 Inverted Bottleneck Block with 7x7 Depthwise Conv & GRN.
    
    Structure:
    x -> 7x7 Depthwise Conv (padding=3) -> 1x1 Conv (expansion) -> GELU -> GRN -> 1x1 Conv (projection) -> (+) -> Out
    """

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.grn = GlobalResponseNorm(hidden_dim)
        self.pwconv2 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x)
        h = self.pwconv1(h)
        h = F.gelu(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        return x + h


class PreActResNetBlock(nn.Module):
    """Pre-Activation ResNet Block (ResNet-v2).
    
    Structure:
    x -> ReLU -> 3x3 Conv -> ReLU -> 3x3 Conv -> (+) -> Out
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return x + h


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def verify_against_golden() -> None:
    """Verifies PyTorch implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = torch.max(torch.abs(test_out - weights["test_output"])).item()
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]

    # 2. ConvNeXt-V2 Block verification
    convnext = ConvNeXtV2Block(in_channels=4, hidden_dim=16)
    with torch.no_grad():
        convnext.dwconv.weight.copy_(weights["dw_w"])
        convnext.dwconv.bias.copy_(weights["dw_b"])
        convnext.pwconv1.weight.copy_(weights["pw1_w"])
        convnext.pwconv1.bias.copy_(weights["pw1_b"])
        convnext.grn.gamma.copy_(weights["grn_gamma"])
        convnext.grn.beta.copy_(weights["grn_beta"])
        convnext.pwconv2.weight.copy_(weights["pw2_w"])
        convnext.pwconv2.bias.copy_(weights["pw2_b"])

    out_convnext = convnext(x)
    err_convnext = torch.max(torch.abs(out_convnext - weights["convnext_output"])).item()
    assert err_convnext < 1e-6, f"ConvNeXt-V2 parity error {err_convnext:.2e} exceeds 1e-6"

    # 3. Pre-Activation ResNet verification
    resnet = PreActResNetBlock(channels=4)
    with torch.no_grad():
        resnet.conv1.weight.copy_(weights["res_w1"])
        resnet.conv1.bias.copy_(weights["res_b1"])
        resnet.conv2.weight.copy_(weights["res_w2"])
        resnet.conv2.bias.copy_(weights["res_b2"])

    out_resnet = resnet(x)
    err_resnet = torch.max(torch.abs(out_resnet - weights["resnet_output"])).item()
    assert err_resnet < 1e-6, f"ResNet parity error {err_resnet:.2e} exceeds 1e-6"

    print("PyTorch 2.x ConvNeXt & ResNet parity successfully verified against golden SafeTensors!")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
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


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_golden()
