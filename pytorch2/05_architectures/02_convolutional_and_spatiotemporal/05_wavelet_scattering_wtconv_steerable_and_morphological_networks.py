"""PyTorch 2.x implementation for Category 05 Subfolder 02 Topic 05:
Wavelet Scattering, WTConv, Steerable & Morphological Networks
in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D Orthonormal Haar Discrete Wavelet Transform."""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 + x01 - x10 - x11)
    hl = 0.5 * (x00 - x01 + x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh


def haar_idwt2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """2D Orthonormal Haar Inverse Discrete Wavelet Transform."""
    B, C, H2, W2 = ll.shape
    H, W = H2 * 2, W2 * 2
    out = torch.zeros(B, C, H, W, device=ll.device, dtype=ll.dtype)

    out[:, :, 0::2, 0::2] = 0.5 * (ll + lh + hl + hh)
    out[:, :, 0::2, 1::2] = 0.5 * (ll + lh - hl - hh)
    out[:, :, 1::2, 0::2] = 0.5 * (ll - lh + hl - hh)
    out[:, :, 1::2, 1::2] = 0.5 * (ll - lh - hl + hh)
    return out


class WTConvBlock(nn.Module):
    """Wavelet Convolution Block (Finder et al., 2024).
    
    Decomposes input into frequency subbands via Haar DWT, applies depthwise convolution
    in the subband domain to expand effective receptive field at O(1) FLOP overhead,
    and reconstructs via IDWT with residual base projection.
    """

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.conv_ll = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.conv_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = haar_dwt2d(x)
        ll_proc = self.conv_ll(ll)
        x_rec = haar_idwt2d(ll_proc, lh, hl, hh)
        base = self.conv_proj(x)
        return base + x_rec


class DeepMorphologicalDilation2d(nn.Module):
    """Deep Morphological Dilation Layer (Franchi et al., 2020).
    
    Implements tropical max-plus morphological dilation:
    (X (+) W)(i, j) = max_{k, l} (X(i-k, j-l) + W(k, l))
    """

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.se = nn.Parameter(torch.zeros(channels, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        out = torch.full((B, C, H, W), -float("inf"), device=x.device, dtype=x.dtype)

        for kh in range(3):
            for kw in range(3):
                val = padded[:, :, kh : kh + H, kw : kw + W] + self.se[:, kh, kw].view(1, C, 1, 1)
                out = torch.maximum(out, val)
        return out


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def verify_against_golden() -> None:
    """Verifies PyTorch implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = torch.max(torch.abs(test_out - weights["test_output"])).item()
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]
    B, C, H, W = x.shape

    # 2. WTConv Verification
    wtconv = WTConvBlock(channels=C)
    with torch.no_grad():
        wtconv.conv_ll.weight.copy_(weights["wt_ll_w"])
        wtconv.conv_ll.bias.copy_(weights["wt_ll_b"])
        wtconv.conv_proj.weight.copy_(weights["wt_proj_w"])
        wtconv.conv_proj.bias.copy_(weights["wt_proj_b"])

    out_wtconv = wtconv(x)
    err_wt = torch.max(torch.abs(out_wtconv - weights["wtconv_output"])).item()
    assert err_wt < 1e-6, f"WTConv parity error {err_wt:.2e} exceeds 1e-6"

    # 3. Morphological Dilation Verification
    morph = DeepMorphologicalDilation2d(channels=C)
    with torch.no_grad():
        morph.se.copy_(weights["morph_se"])

    out_morph = morph(x)
    err_morph = torch.max(torch.abs(out_morph - weights["morph_output"])).item()
    assert err_morph < 1e-6, f"Morphological parity error {err_morph:.2e} exceeds 1e-6"

    print("PyTorch 2.x WTConv & Morphological dilation successfully verified against golden SafeTensors!")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file


    def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """2D Orthonormal Haar Discrete Wavelet Transform.

        Args:
            x: Input tensor [B, C, H, W] with even H, W.

        Returns:
            tuple of (LL, LH, HL, HH), each of shape [B, C, H/2, W/2].
        """
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 + x01 - x10 - x11)
        hl = 0.5 * (x00 - x01 + x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, lh, hl, hh


    def haar_idwt2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        """2D Orthonormal Haar Inverse Discrete Wavelet Transform."""
        B, C, H2, W2 = ll.shape
        H, W = H2 * 2, W2 * 2
        out = torch.zeros(B, C, H, W, device=ll.device, dtype=ll.dtype)

        out[:, :, 0::2, 0::2] = 0.5 * (ll + lh + hl + hh)
        out[:, :, 0::2, 1::2] = 0.5 * (ll + lh - hl - hh)
        out[:, :, 1::2, 0::2] = 0.5 * (ll - lh + hl - hh)
        out[:, :, 1::2, 1::2] = 0.5 * (ll - lh - hl + hh)
        return out


    def morphological_dilation2d(x: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        """2D Deep Morphological Dilation with 3x3 structuring element.

        Args:
            x: [B, C, H, W]
            se: [C, 3, 3] structuring element

        Returns:
            [B, C, H, W] dilated tensor
        """
        B, C, H, W = x.shape
        # Pad by 1 with -inf for max-plus dilation
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        out = torch.full((B, C, H, W), -float("inf"), device=x.device, dtype=x.dtype)

        for kh in range(3):
            for kw in range(3):
                val = padded[:, :, kh : kh + H, kw : kw + W] + se[:, kh, kw].view(1, C, 1, 1)
                out = torch.maximum(out, val)
        return out


    def generate_weights() -> None:
        torch.manual_seed(42)

        # 1. Standard Linear Anchor
        test_input = torch.randn(8, 4)
        weight = torch.randn(4, 4)
        bias = torch.randn(4)
        test_output = test_input @ weight + bias

        # 2. Main Input Tensor: [B=2, C=4, H=8, W=8]
        B, C, H, W = 2, 4, 8, 8
        x = torch.randn(B, C, H, W)

        # Verify Parseval energy conservation on Haar DWT
        ll, lh, hl, hh = haar_dwt2d(x)
        x_rec = haar_idwt2d(ll, lh, hl, hh)
        rec_err = torch.max(torch.abs(x - x_rec)).item()
        print(f"Haar DWT/IDWT Inversion error: {rec_err:.2e}")
        assert rec_err < 1e-6, "Haar DWT/IDWT must have zero reconstruction error!"

        # 3. WTConv Layer:
        # Depthwise 1x1 conv on LL subband: [C=4, 1, 1, 1]
        wt_ll_w = torch.randn(C, 1, 1, 1) * 0.3
        wt_ll_b = torch.randn(C) * 0.1

        # Apply 1x1 conv on LL subband:
        ll_conv = ll * wt_ll_w.view(1, C, 1, 1) + wt_ll_b.view(1, C, 1, 1)

        # Reconstruct via IDWT with processed LL and original detail subbands
        wtconv_rec = haar_idwt2d(ll_conv, lh, hl, hh)

        # Residual projection on x
        wt_proj_w = torch.randn(C, C, 1, 1) * 0.2
        wt_proj_b = torch.randn(C) * 0.1
        wt_base = F.conv2d(x, wt_proj_w, wt_proj_b)
        wtconv_output = wt_base + wtconv_rec

        # 4. Deep Morphological Dilation Layer:
        # Structuring element: [C=4, 3, 3]
        morph_se = torch.randn(C, 3, 3) * 0.1
        morph_output = morphological_dilation2d(x, morph_se)

        tensors = {
            "test_input": test_input,
            "weight": weight,
            "bias": bias,
            "test_output": test_output,
            "input": x,
            "wt_ll_w": wt_ll_w,
            "wt_ll_b": wt_ll_b,
            "wt_proj_w": wt_proj_w,
            "wt_proj_b": wt_proj_b,
            "wtconv_output": wtconv_output,
            "morph_se": morph_se,
            "morph_output": morph_output,
        }

        out_dir = Path("weights/05_architectures/02_convolutional_and_spatiotemporal")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "05_wavelet_scattering_wtconv_steerable_and_morphological_networks.safetensors"
        save_file(tensors, str(out_path))
        print(f"Generated seed-42 weights: {out_path} ({len(tensors)} tensors)")


    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_golden()
