"""Pooling and Downsampling Operators: GeM, BlurPool, PixelUnshuffle, and PMA (C03 Topic 05).

Implements fundamental spatial and token downsampling architectures:
1. Generalized Mean Pooling (GeM): Learnable parameter p smoothly interpolating between Average (p=1) and Max (p->inf) pooling (Radenovic et al. 2018).
2. Anti-Aliased Downsampling (BlurPool): Low-pass filtered sub-sampling preventing Nyquist-Shannon aliasing and restoring shift equivariance (Zhang 2019).
3. PixelUnshuffle: Lossless spatial-to-channel reorganization (Shi et al. 2016).
4. Pooling by Multihead Attention (PMA): Learned seed query tokens attending over variable-length inputs (Lee et al. 2019).
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file


class GeMPool2d(nn.Module):
    """Radenovic et al. (2018) Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(
        self, x: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch channels"]:
        # (1/|R| * sum x^p)^(1/p) over spatial dimensions (-2, -1)
        x_clamped = x.clamp(min=self.eps)
        return (x_clamped.pow(self.p).mean(dim=(-2, -1))).pow(1.0 / self.p)


class BlurPool2d(nn.Module):
    """Zhang (2019) Anti-Aliased Downsampling (BlurPool)."""

    def __init__(self, channels: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.stride = stride
        kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel_2d)

    def forward(
        self, x: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch channels out_h out_w"]:
        # Reflect pad by 1 on all sides for same-padding downsampling
        x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x_padded, self.kernel, stride=self.stride, groups=self.channels)


class PixelUnshuffle2d(nn.Module):
    """Shi et al. (2016) Lossless Spatial-to-Channel Downsampler."""

    def __init__(self, downscale_factor: int = 2) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(
        self, x: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch out_channels out_h out_w"]:
        return F.pixel_unshuffle(x, downscale_factor=self.downscale_factor)


class AttentionPoolingPMA(nn.Module):
    """Lee et al. (2019) Pooling by Multihead Attention (PMA)."""

    def __init__(self, dim: int = 8, num_seeds: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.num_seeds = num_seeds
        self.s_seed = nn.Parameter(torch.randn(1, num_seeds, dim, dtype=torch.float32))

    def forward(
        self, x: Float[torch.Tensor, "batch seq_len dim"]
    ) -> Float[torch.Tensor, "batch dim"]:
        batch_size = x.shape[0]
        q = self.s_seed.expand(batch_size, -1, -1)
        # Scaled dot-product attention
        scores = torch.matmul(q, x.transpose(-2, -1)) / math.sqrt(float(self.dim))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, x) # [batch, num_seeds, dim]
        return out.squeeze(1)


class PoolingUnifiedModule(nn.Module):
    """Unified module combining pooling operators with a linear anchor."""

    def __init__(
        self,
        in_dim: int = 8,
        out_dim: int = 4,
        channels: int = 2,
    ) -> None:
        super().__init__()
        self.gem = GeMPool2d(p=3.0)
        self.blur = BlurPool2d(channels=channels, stride=2)
        self.pixel_unshuffle = PixelUnshuffle2d(downscale_factor=2)
        self.pma = AttentionPoolingPMA(dim=in_dim, num_seeds=1)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[PoolingUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = PoolingUnifiedModule(in_dim=8, out_dim=4, channels=2)

    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])
        module.gem.p.copy_(weights["p_gem"])
        module.blur.kernel.copy_(weights["blur_kernel"])
        module.pma.s_seed.copy_(weights["s_seed"])

    return module, weights


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/05_pooling_and_downsampling_operators.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    module, weights = load_checkpoint(str(weights_path))
    module.eval()

    test_in = weights["test_input"]
    x_gem = weights["x_gem"]
    x_blur = weights["x_blur"]
    x_pixel = weights["x_pixel"]
    x_pma = weights["x_pma"]

    with torch.no_grad():
        # 1. Linear Anchor Parity
        y_anchor = module(test_in)
        diff_anchor = (y_anchor - weights["test_output"]).abs().max().item()
        assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

        # 2. GeM Pooling Parity
        y_gem = module.gem(x_gem)
        diff_gem = (y_gem - weights["y_gem"]).abs().max().item()
        assert diff_gem < 1e-5, f"GeM mismatch: {diff_gem:.2e}"

        # 3. BlurPool Parity
        y_blur = module.blur(x_blur)
        diff_blur = (y_blur - weights["y_blur"]).abs().max().item()
        assert diff_blur < 1e-5, f"BlurPool mismatch: {diff_blur:.2e}"

        # 4. PixelUnshuffle Parity
        y_pixel = module.pixel_unshuffle(x_pixel)
        diff_pixel = (y_pixel - weights["y_pixel"]).abs().max().item()
        assert diff_pixel < 1e-5, f"PixelUnshuffle mismatch: {diff_pixel:.2e}"

        # 5. PMA Attention Pooling Parity
        y_pma = module.pma(x_pma)
        diff_pma = (y_pma - weights["y_pma"]).abs().max().item()
        assert diff_pma < 1e-5, f"PMA mismatch: {diff_pma:.2e}"

    print(
        f"All PyTorch assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"GeM: {diff_gem:.2e}, Blur: {diff_blur:.2e}, Pixel: {diff_pixel:.2e}, PMA: {diff_pma:.2e})"
    )


if __name__ == "__main__":
    main()
