"""Generate 8-cell educational notebook for Category 05 Subfolder 02 Topic 02:
02_specialized_convolutions_and_channel_attention.ipynb in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 02. Specialized Convolutions & Channel Attention: CoordConv, SE-Net, ECA-Net & ASPP

- **Category:** 05. Architectures / 02. Convolutional & Spatiotemporal
- **Lineage & Citations:** Fisher Yu & Vladlen Koltun (ICLR 2016) *Multi-Scale Context Aggregation by Dilated Convolutions*; Liang-Chieh Chen et al. (IEEE TPAMI 2017) *Rethinking Atrous Convolution for Semantic Image Segmentation*; Jie Hu et al. (CVPR 2018) *Squeeze-and-Excitation Networks*; Rosanne Liu et al. (NeurIPS 2018) *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*; Qilong Wang et al. (CVPR 2020) *ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks*.
- **Purpose:** Augments standard convolutions by breaking translational blindness via explicit coordinate injection (CoordConv), dynamically recalibrating channel importance (SE-Net, ECA-Net), and capturing multiscale receptive fields without spatial decimation via atrous dilated convolutions (ASPP).
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext](01_classic_to_modern_cnns_resnet_densenet_convnext.ipynb), [03_building_blocks_and_losses/05_pooling_and_downsampling_operators](../../03_building_blocks_and_losses/05_pooling_and_downsampling_operators.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids](03_unet_and_multiscale_feature_pyramids.ipynb), [05_architectures/04_transformers_and_heads/01_multi_head_multi_query_grouped_query_attention](../04_transformers_and_heads/01_multi_head_multi_query_grouped_query_attention.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never use CoordConv for pure texture classification or translation-invariant tasks where absolute position in the frame is irrelevant or misleading (e.g., classifying a random cropped patch of fabric). Never stack consecutive dilated convolutions with powers-of-two dilation rates ($[2, 4, 8]$) without intermediate smoothing convolutions; the resulting periodic gridding artifact leaves interspersed pixels completely unobserved."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. CoordConv Coordinate Augmentation
Given feature map $\\mathbf{X} \\in \\mathbb{R}^{B \\times C \\times H \\times W}$:
$$x_{i, j} = \\frac{2j}{W - 1} - 1, \\quad y_{i, j} = \\frac{2i}{H - 1} - 1, \\quad \\mathbf{X}_{\\text{coord}} = [\\mathbf{X}, \\mathbf{x}_{\\text{grid}}, \\mathbf{y}_{\\text{grid}}] \\in \\mathbb{R}^{B \\times (C+2) \\times H \\times W}$$
$$\\mathbf{Y} = \\mathbf{W} * \\mathbf{X}_{\\text{coord}} + \\mathbf{b}$$

### 2. Squeeze-and-Excitation (SE Block)
1. **Global Spatial Average Pooling (Squeeze):**
   $$z_c = \\frac{1}{HW} \\sum_{h=1}^H \\sum_{w=1}^W X_{c, h, w}, \\quad \\mathbf{z} \\in \\mathbb{R}^C$$
2. **Channel Excitation (Two-Layer Bottleneck MLP):**
   $$\\mathbf{s} = \\sigma\\left(\\mathbf{W}_2 \\cdot \\text{ReLU}(\\mathbf{W}_1 \\mathbf{z} + \\mathbf{b}_1) + \\mathbf{b}_2\\right), \\quad \\mathbf{W}_1 \\in \\mathbb{R}^{\\frac{C}{r} \\times C}, \\ \\mathbf{W}_2 \\in \\mathbb{R}^{C \\times \\frac{C}{r}}$$
3. **Channel Recalibration:**
   $$\\tilde{\\mathbf{X}}_c = s_c \\cdot \\mathbf{X}_c, \\quad \\forall c \\in \\{1, \\dots, C\\}$$

### 3. Efficient Channel Attention (ECA Block)
Replaces the dimensionality-reduction MLP with a direct $1\\text{D}$ depth-wise convolution along the channel dimension with kernel size $k$:
$$\\mathbf{s} = \\sigma\\left(\\text{Conv1D}_k(\\mathbf{z})\\right) \\implies s_c = \\sigma\\left(\\sum_{j \\in \\Omega_c^k} w_j z_j + b\\right)$$

### 4. Asymptotic Time & Memory Complexity
- **CoordConv:** $\\mathcal{O}(2 \\cdot K^2 \\cdot H \\cdot W)$ extra FLOPs; adds only $2 \\cdot K^2$ parameters.
- **SE Block:** $\\mathcal{O}(H W C + \\frac{2}{r} C^2)$ FLOPs, $\\frac{2}{r} C^2$ parameters.
- **ECA Block:** $\\mathcal{O}(H W C + k C)$ FLOPs, $k$ parameters ($k=3$), parameter overhead is virtually zero ($<0.01\\%$)."""
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
ckpt_rel = "weights/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, C, H, W = x.shape
print(f"Loaded Specialized Convolutions checkpoint on {env.device}: X={x.shape}, C={C}, Spatial=({H},{W})")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class CoordConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        y_coords = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).repeat(B, 1, 1, W)
        x_coords = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).repeat(B, 1, H, 1)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        return self.conv(x_with_coords)

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 2):
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
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        gap = x.mean(dim=(2, 3)).unsqueeze(1)
        scale = torch.sigmoid(self.conv(gap)).view(B, C, 1, 1)
        return x * scale"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """coord_conv = CoordConv2d(in_channels=C, out_channels=C, kernel_size=1)
se = SqueezeAndExcitation(channels=C, reduction=2)
eca = EfficientChannelAttention(kernel_size=3)

with torch.no_grad():
    coord_conv.conv.weight.copy_(weights["coord_conv_w"]); coord_conv.conv.bias.copy_(weights["coord_conv_b"])
    se.fc1.weight.copy_(weights["se_fc1_w"]); se.fc1.bias.copy_(weights["se_fc1_b"])
    se.fc2.weight.copy_(weights["se_fc2_w"]); se.fc2.bias.copy_(weights["se_fc2_b"])
    eca.conv.weight.copy_(weights["eca_conv_w"]); eca.conv.bias.copy_(weights["eca_conv_b"])

# 1. CoordConv Verification
out_coord = coord_conv(x)
err_coord = float(torch.max(torch.abs(out_coord.detach() - weights["coord_output"])))
assert err_coord < 1e-5, f"CoordConv parity error: {err_coord:.2e}"

# 2. SE Verification
out_se = se(x)
err_se = float(torch.max(torch.abs(out_se.detach() - weights["se_output"])))
assert err_se < 1e-5, f"SE parity error: {err_se:.2e}"

# 3. ECA Verification
out_eca = eca(x)
err_eca = float(torch.max(torch.abs(out_eca.detach() - weights["eca_output"])))
assert err_eca < 1e-5, f"ECA parity error: {err_eca:.2e}"

print(f"Parity Verified: CoordConv={err_coord:.2e}, SE={err_se:.2e}, ECA={err_eca:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = r"""import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_coord = benchmark_fn(lambda: coord_conv(x), iters=50, name="CoordConv2d (1x1)")
b_se = benchmark_fn(lambda: se(x), iters=50, name="Squeeze-and-Excitation (SE)")
b_eca = benchmark_fn(lambda: eca(x), iters=50, name="Efficient Channel Attention (ECA)")
print(format_benchmark_table([b_coord, b_se, b_eca]))

# Visualize Channel Attention Weights and CoordConv Grids
with torch.no_grad():
    gap = x.mean(dim=(2, 3))
    se_weights = torch.sigmoid(se.fc2(F.relu(se.fc1(gap))))[0].numpy()
    eca_weights = torch.sigmoid(eca.conv(gap.unsqueeze(1)))[0, 0].numpy()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=100)
channels = range(len(se_weights))
ax[0].bar([c - 0.15 for c in channels], se_weights, width=0.3, label="SE (MLP Bottleneck)", color="#1f77b4")
ax[0].bar([c + 0.15 for c in channels], eca_weights, width=0.3, label="ECA (1D Conv)", color="#2ca02c")
ax[0].set_title("Learned Channel Recalibration Scales $\\mathbf{s} \\in (0, 1)$")
ax[0].set_xlabel("Channel Index")
ax[0].set_ylabel("Attention Weight")
ax[0].legend()
ax[0].grid(True, linestyle="--", alpha=0.5)

# Coordinate Grid overlay
y_coords = torch.linspace(-1.0, 1.0, H).view(H, 1).repeat(1, W).numpy()
im = ax[1].imshow(y_coords, cmap="coolwarm")
ax[1].set_title("CoordConv Explicit Spatial Coordinate Grid $Y_{coord}$")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Checkerboard Gridding Artifact from Non-Coprime Dilations
# Stacking dilated convolutions with rates [2, 4, 8] creates a periodic sampling lattice
# where intermediate pixels inside the receptive field span are never reached by any kernel!
# In contrast, proper dilation rates [1, 2, 5] provide dense, continuous spatial coverage.

def compute_effective_receptive_field(dilation_rates):
    field = torch.zeros(1, 1, 65)
    field[0, 0, 32] = 1.0  # impulse at center
    for r in dilation_rates:
        kernel = torch.ones(1, 1, 3)
        field = F.conv1d(field, kernel, padding=r, dilation=r)
    return field[0, 0]

def count_holes_in_span(rf):
    active = torch.nonzero(rf > 0).squeeze()
    min_i, max_i = active[0].item(), active[-1].item()
    return int((rf[min_i : max_i + 1] == 0).sum()), (min_i, max_i)

rf_gridded = compute_effective_receptive_field([2, 4, 8])
rf_dense = compute_effective_receptive_field([1, 2, 5])

holes_gridded, span_gridded = count_holes_in_span(rf_gridded)
holes_dense, span_dense = count_holes_in_span(rf_dense)

print(f"Non-Coprime Dilations [2, 4, 8]: Span={span_gridded}, Holes inside span={holes_gridded}")
print(f"Coprime Dilations     [1, 2, 5]: Span={span_dense}, Holes inside span={holes_dense}")

assert holes_gridded > 0, "Non-coprime dilations must produce gridding holes!"
assert holes_dense == 0, "Coprime dilations must produce a solid dense receptive field!"
print("Demonstrated catastrophic receptive field gridding and its mitigation via coprime dilation rates.")"""
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Coordinate Awareness:** CoordConv solves translational blindness in coordinate regression tasks with negligible compute overhead by appending spatial coordinate channels.
2. **Channel Squeeze & Recalibration:** Squeeze-and-Excitation dynamically allocates capacity across feature channels via global spatial pooling and an MLP bottleneck.
3. **ECA-Net Parameter Efficiency:** Efficient Channel Attention replaces the MLP bottleneck with a parameter-free 1D convolution, capturing cross-channel interactions at zero computational cost.
4. **Coprime Dilation Design:** Atrous / Dilated convolutions expand receptive fields without downsampling, but rates must be coprime ($[1, 2, 5]$) to eliminate checkerboard gridding artifacts.

---
### Cross-Domain Generalisation Matrix
- **Robotic Manipulation & Grasping:** CoordConv allows vision backbones to map visual pixels directly to end-effector 3D Cartesian coordinates.
- **Biomedical Semantic Segmentation:** ASPP with coprime dilation rates captures multi-scale tumor boundaries without losing cellular resolution.
- **Audio Spectrogram Enhancement:** 1D ECA blocks recalibrate frequency channels dynamically for acoustic event detection.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [02_specialized_convolutions_and_channel_attention.md](../../markdown/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.md)
- **PyTorch 2.x Module:** [02_specialized_convolutions_and_channel_attention.py](../../pytorch2/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.py)
- **Functional JAX Module:** [02_specialized_convolutions_and_channel_attention.py](../../jax/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.py)
- **Native Rust Candle Track:** [02_specialized_convolutions_and_channel_attention.rs](../../candle/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.rs)
- **Hardware WGSL Kernel:** [02_specialized_convolutions_and_channel_attention.rs](../../kernels/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.rs)
- **Universal Golden SafeTensors:** [02_specialized_convolutions_and_channel_attention.safetensors](../../weights/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 01. Classic to Modern CNNs](01_classic_to_modern_cnns_resnet_densenet_convnext.ipynb) | [Category Index](../README.md) | [Next Topic: 03. U-Net & Multiscale Feature Pyramids](03_unet_and_multiscale_feature_pyramids.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/02_convolutional_and_spatiotemporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "02_specialized_convolutions_and_channel_attention.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
