"""PyTorch 2.x implementation for Category 05 Subfolder 02 Topic 02:
Specialized Convolutions & Channel Attention (CoordConv, Squeeze-and-Excitation, ECA-Net)
in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class CoordConv2d(nn.Module):
    """CoordConv layer (Liu et al., 2018).
    
    Appends normalized spatial coordinate channels (x in [-1, 1], y in [-1, 1])
    to the input tensor before performing 2D convolution, providing spatial awareness.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        y_coords = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).repeat(B, 1, 1, W)
        x_coords = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).repeat(B, 1, H, 1)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        return self.conv(x_with_coords)


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation (SE) Block (Hu et al., 2018).
    
    Adaptively recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels via global average pooling and a bottleneck MLP.
    """

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        reduced_dim = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_dim)
        self.fc2 = nn.Linear(reduced_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        gap = x.mean(dim=(2, 3))
        hidden = F.relu(self.fc1(gap))
        scale = torch.sigmoid(self.fc2(hidden)).view(B, C, 1, 1)
        return x * scale


class EfficientChannelAttention(nn.Module):
    """Efficient Channel Attention (ECA) Block (Wang et al., 2020).
    
    Captures local cross-channel interaction without dimensionality reduction
    via 1D convolution with adaptive kernel size k.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        gap = x.mean(dim=(2, 3)).unsqueeze(1)  # [B, 1, C]
        scale = torch.sigmoid(self.conv(gap)).view(B, C, 1, 1)
        return x * scale


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def verify_against_golden() -> None:
    """Verifies PyTorch implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = torch.max(torch.abs(test_out - weights["test_output"])).item()
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]
    B, C, H, W = x.shape

    # 2. CoordConv verification
    coord_conv = CoordConv2d(in_channels=C, out_channels=C, kernel_size=1)
    with torch.no_grad():
        coord_conv.conv.weight.copy_(weights["coord_conv_w"])
        coord_conv.conv.bias.copy_(weights["coord_conv_b"])
    out_coord = coord_conv(x)
    err_coord = torch.max(torch.abs(out_coord - weights["coord_output"])).item()
    assert err_coord < 1e-6, f"CoordConv parity error {err_coord:.2e} exceeds 1e-6"

    # 3. Squeeze-and-Excitation verification
    se = SqueezeAndExcitation(channels=C, reduction=2)
    with torch.no_grad():
        se.fc1.weight.copy_(weights["se_fc1_w"])
        se.fc1.bias.copy_(weights["se_fc1_b"])
        se.fc2.weight.copy_(weights["se_fc2_w"])
        se.fc2.bias.copy_(weights["se_fc2_b"])
    out_se = se(x)
    err_se = torch.max(torch.abs(out_se - weights["se_output"])).item()
    assert err_se < 1e-6, f"SE parity error {err_se:.2e} exceeds 1e-6"

    # 4. Efficient Channel Attention verification
    eca = EfficientChannelAttention(kernel_size=3)
    with torch.no_grad():
        eca.conv.weight.copy_(weights["eca_conv_w"])
        eca.conv.bias.copy_(weights["eca_conv_b"])
    out_eca = eca(x)
    err_eca = torch.max(torch.abs(out_eca - weights["eca_output"])).item()
    assert err_eca < 1e-6, f"ECA parity error {err_eca:.2e} exceeds 1e-6"

    print("PyTorch 2.x CoordConv, SE & ECA parity successfully verified against golden SafeTensors!")


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


    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_golden()
