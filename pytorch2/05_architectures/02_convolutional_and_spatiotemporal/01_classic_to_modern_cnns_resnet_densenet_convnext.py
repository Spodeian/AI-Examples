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


if __name__ == "__main__":
    verify_against_golden()
