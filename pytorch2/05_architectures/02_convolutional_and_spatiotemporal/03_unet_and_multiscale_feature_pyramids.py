"""PyTorch 2.x implementation for Category 05 Subfolder 02 Topic 03:
U-Net & Multiscale Feature Pyramids (U-Net skip concatenation, BiFPN fast normalized fusion)
in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class UNetMiniBlock(nn.Module):
    """Miniature U-Net Block with contracting encoder, bottleneck, and expansive decoder.
    
    Structure:
    x -> Conv1 (pad 1) -> ReLU (e1) -> MaxPool 2x2 -> Bottleneck Conv -> ReLU (b)
      -> Nearest 2x Upsample -> Cat([up, e1]) -> Decoder Conv -> ReLU -> Out
    """

    def __init__(self, in_channels: int = 4, mid_channels: int = 8, out_channels: int = 4) -> None:
        super().__init__()
        self.conv_enc = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv_bot = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv_dec = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = F.relu(self.conv_enc(x))
        p1 = F.max_pool2d(e1, kernel_size=2, stride=2)
        b = F.relu(self.conv_bot(p1))
        up = F.interpolate(b, scale_factor=2.0, mode="nearest")
        cat_feat = torch.cat([up, e1], dim=1)
        return F.relu(self.conv_dec(cat_feat))


class BiFPNFusion(nn.Module):
    """BiFPN Fast Normalized Feature Fusion (Tan et al., 2020).
    
    Computes weighted fusion using positive weights with ReLU and normalization:
    O = sum_i (w_i / (sum_j w_j + eps)) * I_i
    followed by a 1x1 projection convolution.
    """

    def __init__(self, channels: int = 4, num_inputs: int = 2, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(num_inputs))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        w_relu = F.relu(self.w)
        norm_w = w_relu / (w_relu.sum() + self.eps)
        fused = norm_w[0] * p1 + norm_w[1] * p2
        return self.conv(fused)


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def verify_against_golden() -> None:
    """Verifies PyTorch implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = torch.max(torch.abs(test_out - weights["test_output"])).item()
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]

    # 2. U-Net Mini Block verification
    unet = UNetMiniBlock(in_channels=4, mid_channels=8, out_channels=4)
    with torch.no_grad():
        unet.conv_enc.weight.copy_(weights["enc_w"])
        unet.conv_enc.bias.copy_(weights["enc_b"])
        unet.conv_bot.weight.copy_(weights["bot_w"])
        unet.conv_bot.bias.copy_(weights["bot_b"])
        unet.conv_dec.weight.copy_(weights["dec_w"])
        unet.conv_dec.bias.copy_(weights["dec_b"])
    out_unet = unet(x)
    err_unet = torch.max(torch.abs(out_unet - weights["unet_output"])).item()
    assert err_unet < 1e-6, f"U-Net parity error {err_unet:.2e} exceeds 1e-6"

    # 3. BiFPN Fusion verification
    bifpn = BiFPNFusion(channels=4, num_inputs=2)
    with torch.no_grad():
        bifpn.w.copy_(weights["bifpn_w"])
        bifpn.conv.weight.copy_(weights["bifpn_conv_w"])
        bifpn.conv.bias.copy_(weights["bifpn_conv_b"])
    out_bifpn = bifpn(x, out_unet)
    err_bifpn = torch.max(torch.abs(out_bifpn - weights["bifpn_output"])).item()
    assert err_bifpn < 1e-6, f"BiFPN parity error {err_bifpn:.2e} exceeds 1e-6"

    print("PyTorch 2.x U-Net & BiFPN parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
