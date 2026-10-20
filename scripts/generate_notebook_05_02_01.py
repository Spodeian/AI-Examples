"""Generate 8-cell educational notebook for Category 05 Subfolder 02 Topic 01:
01_classic_to_modern_cnns_resnet_densenet_convnext.ipynb in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 01. Classic to Modern CNNs: ResNet v1/v2, DenseNet, MobileNetV2, and ConvNeXt-V2 (GRN)

- **Category:** 05. Architectures / 02. Convolutional & Spatiotemporal
- **Lineage & Citations:** Kaiming He et al. (CVPR 2016, ECCV 2016) *Deep Residual Learning for Image Recognition* & *Identity Mappings in Deep Residual Networks*; Gao Huang et al. (CVPR 2017) *Densely Connected Convolutional Networks*; Mark Sandler et al. (CVPR 2018) *MobileNetV2: Inverted Residuals and Linear Bottlenecks*; Zhuang Liu et al. (CVPR 2022) *A ConvNet for the 2020s*; Sanghyun Woo et al. (CVPR 2023) *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*.
- **Purpose:** Traces the structural evolution of 2D convolutional vision models from residual gradient highways (ResNet-v2) to inverted bottlenecks with depthwise separable convolutions (MobileNetV2) and modern large-kernel Vision Transformer competitors with Global Response Normalization (ConvNeXt-V2).
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp](../01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.ipynb), [03_building_blocks_and_losses/01_activations_and_thresholding](../../03_building_blocks_and_losses/01_activations_and_thresholding.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention](02_specialized_convolutions_and_channel_attention.ipynb), [05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids](03_unet_and_multiscale_feature_pyramids.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never use standard $3\\times 3$ dense convolutions with Post-Activation (ResNet-v1) in ultra-deep (>100 layer) networks; unnormalized residual branches destroy identity gradient highways. Never omit Global Response Normalization (GRN) in ConvNeXt blocks when training with masked image modeling (MAE); without GRN channel competition, spatial masking induces channel feature collapse where a small subset of channels dominate while the rest degenerate to zero."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Pre-Activation ResNet-v2 Gradient Highway
Arranging normalizations and non-linearities before convolutions creates an unobstructed additive identity mapping:
$$\\mathbf{x}_{l+1} = \\mathbf{x}_l + \\mathcal{F}(\\mathbf{x}_l, \\mathcal{W}_l) \\implies \\mathbf{x}_L = \\mathbf{x}_l + \\sum_{i=l}^{L-1} \\mathcal{F}(\\mathbf{x}_i, \\mathcal{W}_i)$$
Gradient flow via chain rule contains an unconditional identity term $I$:
$$\\frac{\\partial \\mathcal{E}}{\\partial \\mathbf{x}_l} = \\frac{\\partial \\mathcal{E}}{\\partial \\mathbf{x}_L} \\left(I + \\frac{\\partial}{\\partial \\mathbf{x}_l} \\sum_{i=l}^{L-1} \\mathcal{F}(\\mathbf{x}_i, \\mathcal{W}_i)\\right)$$

### 2. ConvNeXt-V2 Inverted Bottleneck & Global Response Normalization (GRN)
ConvNeXt-V2 processes spatial features via $7\\times 7$ depthwise convolution, expands channels $4\\times$ via pointwise $1\\times 1$ conv, applies GELU and GRN, and projects back:
1. **Global Spatial Aggregation (L2 Norm):**
   $$G_c = \\|\\mathbf{X}_c\\|_2 = \\sqrt{\\sum_{h=1}^H \\sum_{w=1}^W X_{c, h, w}^2 + \\epsilon}, \\quad \\forall c \\in \\{1, \\dots, C\\}$$
2. **Channel Mutual Competition Contrast:**
   $$N_c = \\frac{G_c}{\\frac{1}{C} \\sum_{j=1}^C G_j + \\epsilon}$$
3. **Calibrated Feature Modulation & Residual Addition:**
   $$\\mathbf{X}_c' = \\gamma_c (\\mathbf{X}_c \\odot N_c) + \\beta_c + \\mathbf{X}_c$$

### 3. Asymptotic Time & Memory Complexity
- **ResNet-v2 Block:** $\\mathcal{O}(2 \\cdot K^2 \\cdot H \\cdot W \\cdot C^2) = \\mathcal{O}(18 \\cdot H \\cdot W \\cdot C^2)$ FLOPs for $3\\times 3$ kernels.
- **ConvNeXt-V2 Block:** $\\mathcal{O}(K_{dw}^2 \\cdot H \\cdot W \\cdot C + 2 \\cdot H \\cdot W \\cdot C \\cdot (4C)) = \\mathcal{O}(49 H W C + 8 H W C^2)$ FLOPs. For large $C$, ConvNeXt is $>2\\times$ faster than standard $3\\times 3$ dense blocks while maintaining a receptive field of $7\\times 7$ vs $3\\times 3$.
- **Activation Memory:** $\\mathcal{O}(H \\cdot W \\cdot C)$ per block."""
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
ckpt_rel = "weights/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, C, H, W = x.shape
print(f"Loaded ConvNeXt-V2 & ResNet checkpoint on {env.device}: X={x.shape}, C={C}, Spatial=({H},{W})")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(1, -1, 1, 1) + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
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
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return x + h"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """convnext = ConvNeXtV2Block(in_channels=4, hidden_dim=16)
resnet = PreActResNetBlock(channels=4)

with torch.no_grad():
    convnext.dwconv.weight.copy_(weights["dw_w"]); convnext.dwconv.bias.copy_(weights["dw_b"])
    convnext.pwconv1.weight.copy_(weights["pw1_w"]); convnext.pwconv1.bias.copy_(weights["pw1_b"])
    convnext.grn.gamma.copy_(weights["grn_gamma"]); convnext.grn.beta.copy_(weights["grn_beta"])
    convnext.pwconv2.weight.copy_(weights["pw2_w"]); convnext.pwconv2.bias.copy_(weights["pw2_b"])
    resnet.conv1.weight.copy_(weights["res_w1"]); resnet.conv1.bias.copy_(weights["res_b1"])
    resnet.conv2.weight.copy_(weights["res_w2"]); resnet.conv2.bias.copy_(weights["res_b2"])

# 1. ConvNeXt-V2 Verification
out_convnext = convnext(x)
err_convnext = float(torch.max(torch.abs(out_convnext.detach() - weights["convnext_output"])))
assert err_convnext < 1e-5, f"ConvNeXt-V2 parity error: {err_convnext:.2e}"

# 2. ResNet Verification
out_resnet = resnet(x)
err_resnet = float(torch.max(torch.abs(out_resnet.detach() - weights["resnet_output"])))
assert err_resnet < 1e-5, f"ResNet parity error: {err_resnet:.2e}"

print(f"Parity Verified: ConvNeXt-V2={err_convnext:.2e}, ResNet={err_resnet:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = r"""import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_convnext = benchmark_fn(lambda: convnext(x), iters=50, name="ConvNeXt-V2 Block (7x7 DW + GRN)")
b_resnet = benchmark_fn(lambda: resnet(x), iters=50, name="Pre-Act ResNet Block (3x3 Dense)")
print(format_benchmark_table([b_convnext, b_resnet]))

# Visualize GRN Channel Competition Normalization
with torch.no_grad():
    expanded = F.gelu(convnext.pwconv1(convnext.dwconv(x)))
    gx_raw = torch.norm(expanded, p=2, dim=(2, 3))[0].numpy()
    calibrated = convnext.grn(expanded)
    gx_calib = torch.norm(calibrated, p=2, dim=(2, 3))[0].numpy()

channels = range(len(gx_raw))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=100)
ax[0].bar(channels, gx_raw, color="#1f77b4", alpha=0.85, label="Raw GELU Channels")
ax[0].set_title("Pre-GRN Channel Energy $||X_c||_2$")
ax[0].set_xlabel("Channel Index")
ax[0].grid(True, linestyle="--", alpha=0.5)

ax[1].bar(channels, gx_calib, color="#2ca02c", alpha=0.85, label="GRN Calibrated")
ax[1].set_title("Post-GRN Competitive Channel Energy")
ax[1].set_xlabel("Channel Index")
ax[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Post-Activation ResNet (v1) Skip Path Gradient Truncation
# In ResNet-v1, non-linearities follow addition: y = ReLU(x + F(x)).
# Placing ReLU directly on the skip connection zeroes out negative signal trajectories,
# truncating gradients and destroying identity highway preservation.
# In contrast, Pre-Activation (ResNet-v2) preserves a completely clean identity highway.

x_v1 = torch.randn(1, 16, 8, 8, requires_grad=True)
x_v2 = x_v1.clone().detach().requires_grad_(True)

# 20 layers of Post-Act (v1) vs Pre-Act (v2)
z1 = x_v1
for _ in range(20):
    z1 = F.relu(z1 + 0.02 * torch.randn_like(z1))

z2 = x_v2
for _ in range(20):
    z2 = z2 + 0.02 * F.relu(torch.randn_like(z2))

z1.sum().backward()
z2.sum().backward()

grad_norm_v1 = x_v1.grad.norm().item()
grad_norm_v2 = x_v2.grad.norm().item()

print(f"ResNet-v1 (Post-Act) Input Gradient Norm: {grad_norm_v1:.4f}")
print(f"ResNet-v2 (Pre-Act)  Input Gradient Norm: {grad_norm_v2:.4f}")
assert grad_norm_v2 > grad_norm_v1, "Pre-activation must preserve superior gradient magnitude!"

# Demonstrate GRN Contrast Amplification:
# Salient signals are magnified while background noise channels are suppressed
salient_channel = torch.tensor([[[[10.0]], [[1.0]]]])  # 10:1 ratio
gx = torch.norm(salient_channel, p=2, dim=(2, 3), keepdim=True)
nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
modulated = salient_channel * nx
ratio_before = (salient_channel[0, 0] / salient_channel[0, 1]).item()
ratio_after = (modulated[0, 0] / modulated[0, 1]).item()

print(f"Feature Contrast Ratio Before GRN: {ratio_before:.1f}:1")
print(f"Feature Contrast Ratio After GRN:  {ratio_after:.1f}:1")
assert ratio_after > ratio_before, "GRN must amplify inter-channel competitive contrast!"
print("Demonstrated Post-Act gradient attenuation and GRN contrast magnification.")"""
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Unobstructed Identity Highway:** Pre-Activation ResNet-v2 keeps the skip path completely clean of non-linearities and normalizations, enabling stable gradient propagation through hundreds of layers.
2. **Inverted Bottlenecks with Depthwise Separation:** MobileNetV2 and ConvNeXt-V2 invert the classical bottleneck by expanding channels prior to spatial filtering, maximizing expressivity while minimizing parameter counts.
3. **GRN Eliminates Feature Collapse:** Global Response Normalization enforces intra-channel competition, allowing ConvNets to rival Vision Transformers in self-supervised masked autoencoder pre-training.

---
### Cross-Domain Generalisation Matrix
- **Bio-Medical Image Segmentation (Pathology/MRI):** ConvNeXt-V2 backbones replace ViTs in dense pixel prediction with linear scaling in image resolution.
- **Edge & Embedded AI:** MobileNetV2 inverted bottlenecks remain the golden standard for zero-latency inference on microcontroller NPUs.
- **Audio Spectrogram Modeling:** 2D ConvNeXt-V2 layers process log-mel spectrograms with large receptive fields along time and frequency axes.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [01_classic_to_modern_cnns_resnet_densenet_convnext.md](../../markdown/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.md)
- **PyTorch 2.x Module:** [01_classic_to_modern_cnns_resnet_densenet_convnext.py](../../pytorch2/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.py)
- **Functional JAX Module:** [01_classic_to_modern_cnns_resnet_densenet_convnext.py](../../jax/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.py)
- **Native Rust Candle Track:** [01_classic_to_modern_cnns_resnet_densenet_convnext.rs](../../candle/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.rs)
- **Hardware WGSL Kernel:** [01_classic_to_modern_cnns_resnet_densenet_convnext.rs](../../kernels/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.rs)
- **Universal Golden SafeTensors:** [01_classic_to_modern_cnns_resnet_densenet_convnext.safetensors](../../weights/05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Subfolder: 01. MLP, MoE, Soups & KAN](../01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.ipynb) | [Category Index](../README.md) | [Next Topic: 02. Specialized Convolutions & Channel Attention](02_specialized_convolutions_and_channel_attention.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/02_convolutional_and_spatiotemporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "01_classic_to_modern_cnns_resnet_densenet_convnext.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
