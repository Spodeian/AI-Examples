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


    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_golden()
