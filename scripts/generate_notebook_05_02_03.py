"""Generate 8-cell educational notebook for Category 05 Subfolder 02 Topic 03:
03_unet_and_multiscale_feature_pyramids.ipynb in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 03. U-Net & Multiscale Feature Pyramids: U-Net, FPN, BiFPN & HRNet

- **Category:** 05. Architectures / 02. Convolutional & Spatiotemporal
- **Lineage & Citations:** Olaf Ronneberger et al. (MICCAI 2015) *U-Net: Convolutional Networks for Biomedical Image Segmentation*; Tsung-Yi Lin et al. (CVPR 2017) *Feature Pyramid Networks for Object Detection*; Mingxing Tan et al. (CVPR 2020) *EfficientDet: Scalable and Efficient Object Detection (BiFPN)*; Jingdong Wang et al. (IEEE TPAMI 2020) *Deep High-Resolution Representation Learning for Visual Recognition (HRNet)*.
- **Purpose:** Bridges deep semantic abstractions with high-resolution pixel localization through symmetrical contracting-expanding pathways with long skip connections (U-Net) and bidirectional multiscale pyramids with Fast Normalized Feature Fusion (BiFPN).
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext](01_classic_to_modern_cnns_resnet_densenet_convnext.ipynb), [05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention](02_specialized_convolutions_and_channel_attention.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer](04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.ipynb), [05_architectures/05_generative_models/01_denoising_diffusion_probabilistic_models](../05_generative_models/01_denoising_diffusion_probabilistic_models.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never pass arbitrary odd spatial resolutions into a multi-stage U-Net without spatial reflection padding; integer floor division during downsampling ($15 \\to 7$) causes a 1-pixel dimensional mismatch upon $2\\times$ upsampling ($14 \\neq 15$), causing runtime shape exceptions or silent misaligned boundary crops during skip concatenation."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = r"""## Mathematical Foundations & Governing Equations

### 1. U-Net Skip Connection Concatenation
At each symmetrical resolution level $l$, high-resolution spatial details bypass the contracting bottleneck:
$$\\hat{\\mathbf{x}}_{\\text{dec}}^{(l)} = \\text{Upsample}_{2\\times}\\left(\\mathbf{x}_{\\text{dec}}^{(l+1)}\\right), \\quad \\mathbf{y}^{(l)} = \\text{ReLU}\\left(\\mathbf{W}_{\\text{dec}}^{(l)} * [\\hat{\\mathbf{x}}_{\\text{dec}}^{(l)}, \\mathbf{x}_{\\text{enc}}^{(l)}] + \\mathbf{b}^{(l)}\\right)$$
where $[\cdot, \\cdot]$ concatenates feature maps along the channel axis ($C_{\\text{dec}} + C_{\\text{enc}}$), preserving fine spatial gradients during backward propagation.

### 2. BiFPN Fast Normalized Feature Fusion
Conventional multi-scale pyramid fusion uses Softmax, which introduces expensive exponential operations. BiFPN uses bounded Fast Normalized Fusion:
$$\\mathbf{O} = \\text{Conv}\\left(\\sum_{i=1}^n \\frac{\\text{ReLU}(w_i)}{\\epsilon + \\sum_{j=1}^n \\text{ReLU}(w_j)} \\mathbf{I}_i\\right)$$
where learnable scalar weights $w_i$ dynamically balance semantic vs high-resolution multi-scale pyramid contributions with pure arithmetic operations.

### 3. Asymptotic Time & Memory Complexity
- **U-Net FLOPs:** $\\mathcal{O}\\left(\\sum_{l=1}^L 2 \\cdot K^2 \\cdot H_l W_l C_l C_{l+1}\\right)$, dominated by high-resolution early stages where $H_1 W_1$ is largest.
- **BiFPN Fast Fusion:** $\\mathcal{O}(n \\cdot H W C)$ arithmetic operations with zero transcendental operations, followed by a $1\\times 1$ conv of $\\mathcal{O}(H W C^2)$ FLOPs.
- **Activation Memory:** $\\mathcal{O}\\left(B \\cdot \\sum_{l=1}^L H_l W_l C_l\\right)$; all encoder activations must be retained in memory until the corresponding decoder level executes."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from pathlib import Path
import core

env = core.setup_env(seed=42)

# Load Universal Golden SafeTensors Checkpoint
ckpt_rel = "weights/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, C, H, W = x.shape
print(f"Loaded U-Net & BiFPN checkpoint on {env.device}: X={x.shape}, C={C}, Spatial=({H},{W})")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class UNetMiniBlock(nn.Module):
    def __init__(self, in_channels: int = 4, mid_channels: int = 8, out_channels: int = 4):
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
    def __init__(self, channels: int = 4, num_inputs: int = 2, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(num_inputs))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        w_relu = F.relu(self.w)
        norm_w = w_relu / (w_relu.sum() + self.eps)
        fused = norm_w[0] * p1 + norm_w[1] * p2
        return self.conv(fused)"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """unet = UNetMiniBlock(in_channels=4, mid_channels=8, out_channels=4)
bifpn = BiFPNFusion(channels=4, num_inputs=2)

with torch.no_grad():
    unet.conv_enc.weight.copy_(weights["enc_w"]); unet.conv_enc.bias.copy_(weights["enc_b"])
    unet.conv_bot.weight.copy_(weights["bot_w"]); unet.conv_bot.bias.copy_(weights["bot_b"])
    unet.conv_dec.weight.copy_(weights["dec_w"]); unet.conv_dec.bias.copy_(weights["dec_b"])
    bifpn.w.copy_(weights["bifpn_w"])
    bifpn.conv.weight.copy_(weights["bifpn_conv_w"]); bifpn.conv.bias.copy_(weights["bifpn_conv_b"])

# 1. U-Net Verification
out_unet = unet(x)
err_unet = float(torch.max(torch.abs(out_unet.detach() - weights["unet_output"])))
assert err_unet < 1e-5, f"U-Net parity error: {err_unet:.2e}"

# 2. BiFPN Fusion Verification
out_bifpn = bifpn(x, out_unet)
err_bifpn = float(torch.max(torch.abs(out_bifpn.detach() - weights["bifpn_output"])))
assert err_bifpn < 1e-5, f"BiFPN parity error: {err_bifpn:.2e}"

print(f"Parity Verified: U-Net={err_unet:.2e}, BiFPN={err_bifpn:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = r"""import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_unet = benchmark_fn(lambda: unet(x), iters=50, name="U-Net Mini (Enc + Bot + Dec)")
b_bifpn = benchmark_fn(lambda: bifpn(x, out_unet), iters=50, name="BiFPN Fast Normalized Fusion")
print(format_benchmark_table([b_unet, b_bifpn]))

# Visualize Multi-Scale Feature Map & Normalized Fusion Weights
w_relu = F.relu(bifpn.w)
norm_w = (w_relu / (w_relu.sum() + 1e-4)).detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=100)
im0 = ax[0].imshow(out_unet[0, 0].detach().numpy(), cmap="viridis")
ax[0].set_title("U-Net Decoder Output Feature (Channel 0)")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

ax[1].bar(["Input Feature P1", "U-Net Feature P2"], norm_w, color=["#1f77b4", "#2ca02c"], width=0.5)
ax[1].set_title("BiFPN Normalized Fusion Weights $w_i / (\\sum w_j + \\epsilon)$")
ax[1].set_ylabel("Normalized Weight")
ax[1].set_ylim(0, 1.0)
ax[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Odd Spatial Dimensions in U-Net Contracting Hierarchy
# When spatial dimensions are odd (e.g. 15x15), 2x2 max-pooling floors coordinates: 15 // 2 = 7.
# Nearest-neighbor 2x upsampling then expands: 7 * 2 = 14 != 15!
# Concatenating the decoder tensor (14x14) with the encoder skip (15x15) causes a shape mismatch!

x_odd = torch.randn(1, 4, 15, 15)

try:
    # Attempting forward pass on odd dimension without padding
    _ = unet(x_odd)
    failed = False
except RuntimeError as e:
    failed = True
    print(f"Caught Expected Dimension Mismatch Error: {e}")

assert failed, "Odd input dimensions without reflection padding must fail shape alignment!"

# Clean Solution: Reflection-pad spatial dimensions to the nearest multiple of 2^L (here 16x16)
pad_h = (16 - 15)
x_padded = F.pad(x_odd, (0, pad_h, 0, pad_h), mode="reflect")
out_padded = unet(x_padded)
# Crop back to the original resolution (15x15)
out_clean = out_padded[:, :, :15, :15]

print(f"Original Input Shape: {x_odd.shape}")
print(f"Padded Input Shape:   {x_padded.shape}")
print(f"Clean Output Shape:   {out_clean.shape}")
assert out_clean.shape == x_odd.shape, "Padding and cropping restores perfect spatial alignment!"
print("Demonstrated U-Net odd-dimension alignment pitfall and its reflection-padding fix.")"""
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Long Skip Feature Preservation:** U-Net bypasses information bottlenecks by concatenating high-resolution encoder features directly with decoder features, preserving sharp localization.
2. **Fast Normalized Fusion:** BiFPN eliminates expensive Softmax transcendental calculations in multiscale pyramids with zero degradation in accuracy.
3. **Power-of-Two Spatial Constraints:** Multi-scale architectures strictly require spatial dimensions divisible by $2^L$ to avoid 1-pixel skip connection misalignment.

---
### Cross-Domain Generalisation Matrix
- **Denoising Diffusion Probabilistic Models (DDPM):** U-Net forms the core denoising backbone of Latent Diffusion and Stable Diffusion models.
- **Biomedical Organ Segmentation (CT / MRI):** 2D and 3D U-Nets segment organ contours and tumors at single-voxel precision.
- **Aerial & Satellite Earth Observation:** BiFPN feature pyramids detect multi-scale vehicles, buildings, and agricultural fields across gigapixel orthomosaics.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [03_unet_and_multiscale_feature_pyramids.md](../../markdown/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.md)
- **PyTorch 2.x Module:** [03_unet_and_multiscale_feature_pyramids.py](../../pytorch2/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.py)
- **Functional JAX Module:** [03_unet_and_multiscale_feature_pyramids.py](../../jax/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.py)
- **Native Rust Candle Track:** [03_unet_and_multiscale_feature_pyramids.rs](../../candle/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.rs)
- **Hardware WGSL Kernel:** [03_unet_and_multiscale_feature_pyramids.rs](../../kernels/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.rs)
- **Universal Golden SafeTensors:** [03_unet_and_multiscale_feature_pyramids.safetensors](../../weights/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 02. Specialized Convolutions & Channel Attention](02_specialized_convolutions_and_channel_attention.ipynb) | [Category Index](../README.md) | [Next Topic: 04. Spatiotemporal Video & ViViT/TimeSformer](04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/02_convolutional_and_spatiotemporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "03_unet_and_multiscale_feature_pyramids.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
