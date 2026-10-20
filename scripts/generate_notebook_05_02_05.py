"""Generate 8-cell educational notebook for Category 05 Subfolder 02 Topic 05:
05_wavelet_scattering_wtconv_steerable_and_morphological_networks.ipynb in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 05. Wavelet Scattering, WTConv, Steerable & Morphological Networks

- **Category:** 05. Architectures / 02. Convolutional & Spatiotemporal
- **Lineage & Citations:** Stéphane Mallat (CPAM 2012) *Group Invariant Scattering*; Joan Bruna & Stéphane Mallat (IEEE TPAMI 2013) *Invariant Scattering Convolution Networks*; Shahaf E. Finder et al. (arXiv:2407.05848, 2024) *Wavelet Convolutions for Large Receptive Fields (WTConv)*; Taco S. Cohen & Max Welling (ICML 2016) *Group Equivariant Convolutional Networks (G-CNNs)*; Gianni Franchi et al. (IEEE TNNLS 2020) *Deep Morphological Networks: A Max-Plus Algebra Framework*.
- **Purpose:** Solves the inductive bias limitations of standard CNNs by decomposing feature maps into multi-scale frequency subbands with $\\mathcal{O}(1)$ parameter overhead (WTConv), providing provable deformation stability via wavelet scattering transforms, and learning non-linear structural shape primitives via tropical max-plus algebra (Deep Morphological Networks).
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext](01_classic_to_modern_cnns_resnet_densenet_convnext.ipynb), [05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer](04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/03_recurrent_ssm_and_memory/01_lstm_gru_and_xlstm](../03_recurrent_ssm_and_memory/01_lstm_gru_and_xlstm.ipynb), [05_architectures/06_graphs_geometric_and_inr/01_message_passing_mpnn_gcn_gat_gin](../06_graphs_geometric_and_inr/01_message_passing_mpnn_gcn_gat_gin.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never apply Haar DWT to odd spatial dimensions without explicit symmetric padding; downsampling non-even grids creates off-by-one phase shift artifacts. Never construct deep morphological networks purely out of max-plus layers without linear projection mixing; unconstrained max activations saturate quickly, leading to vanishing gradient flow across dead structuring element coordinates."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = r"""## Mathematical Foundations & Governing Equations

### 1. 2D Orthonormal Haar Discrete Wavelet Transform (DWT)
For spatial input $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$, downsample into 4 interleaved sub-grids:
$$\mathbf{X}_{00} = \mathbf{X}[:, :, 0::2, 0::2], \quad \mathbf{X}_{01} = \mathbf{X}[:, :, 0::2, 1::2], \quad \mathbf{X}_{10} = \mathbf{X}[:, :, 1::2, 0::2], \quad \mathbf{X}_{11} = \mathbf{X}[:, :, 1::2, 1::2]$$
The orthonormal 2D Haar subband decomposition is:
$$\mathbf{LL} = \frac{1}{2}(\mathbf{X}_{00} + \mathbf{X}_{01} + \mathbf{X}_{10} + \mathbf{X}_{11}) \quad \text{(Low-frequency approximation)}$$
$$\mathbf{LH} = \frac{1}{2}(\mathbf{X}_{00} + \mathbf{X}_{01} - \mathbf{X}_{10} - \mathbf{X}_{11}) \quad \text{(Horizontal detail)}$$
$$\mathbf{HL} = \frac{1}{2}(\mathbf{X}_{00} - \mathbf{X}_{01} + \mathbf{X}_{10} - \mathbf{X}_{11}) \quad \text{(Vertical detail)}$$
$$\mathbf{HH} = \frac{1}{2}(\mathbf{X}_{00} - \mathbf{X}_{01} - \mathbf{X}_{10} + \mathbf{X}_{11}) \quad \text{(Diagonal high-frequency detail)}$$

**Parseval Energy Conservation Invariant:**
$$\|\mathbf{X}\|_F^2 = \|\mathbf{LL}\|_F^2 + \|\mathbf{LH}\|_F^2 + \|\mathbf{HL}\|_F^2 + \|\mathbf{HH}\|_F^2$$

### 2. 2D Haar Inverse Discrete Wavelet Transform (IDWT)
Exact spatial reconstruction ($\|\mathbf{X} - \hat{\mathbf{X}}\|_\infty < 10^{-7}$) without numerical dissipation:
$$\hat{\mathbf{X}}[:, :, 0::2, 0::2] = \frac{1}{2}(\mathbf{LL} + \mathbf{LH} + \mathbf{HL} + \mathbf{HH})$$
$$\hat{\mathbf{X}}[:, :, 0::2, 1::2] = \frac{1}{2}(\mathbf{LL} + \mathbf{LH} - \mathbf{HL} - \mathbf{HH})$$
$$\hat{\mathbf{X}}[:, :, 1::2, 0::2] = \frac{1}{2}(\mathbf{LL} - \mathbf{LH} + \mathbf{HL} - \mathbf{HH})$$
$$\hat{\mathbf{X}}[:, :, 1::2, 1::2] = \frac{1}{2}(\mathbf{LL} - \mathbf{LH} - \mathbf{HL} + \mathbf{HH})$$

### 3. Tropical Max-Plus Morphological Dilation
In tropical algebra $(\mathbb{R} \cup \{-\infty\}, \oplus, \otimes)$ with $\oplus = \max$ and $\otimes = +$:
$$(\mathbf{X} \oplus \mathbf{W})(c, i, j) = \max_{k, l \in \{0, 1, 2\}} \left( \mathbf{X}_{\text{pad}}(c, i+k-1, j+l-1) + \mathbf{W}(c, k, l) \right)$$

### 4. Asymptotic Time & Memory Complexity
- **Haar DWT & IDWT:** $\mathcal{O}(B \cdot C \cdot H \cdot W)$ FLOPs, linear time, memory-bandwidth bound.
- **WTConv Layer:** $\mathcal{O}(B \cdot C \cdot H \cdot W)$ with only $0.25\times$ FLOPs on $\mathbf{LL}$ vs standard full-resolution conv.
- **Morphological Dilation:** $\mathcal{O}(B \cdot C \cdot H \cdot W \cdot K^2)$ comparisons with zero multiplications."""
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
ckpt_rel = "weights/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, C, H, W = x.shape
print(f"Loaded WTConv & Morphology checkpoint on {env.device}: Input shape={x.shape}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """def haar_dwt2d(x: torch.Tensor):
    x00, x01 = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
    x10, x11 = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 + x01 - x10 - x11)
    hl = 0.5 * (x00 - x01 + x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh

def haar_idwt2d(ll, lh, hl, hh):
    B, C, H2, W2 = ll.shape
    out = torch.zeros(B, C, H2 * 2, W2 * 2, device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, 0::2] = 0.5 * (ll + lh + hl + hh)
    out[:, :, 0::2, 1::2] = 0.5 * (ll + lh - hl - hh)
    out[:, :, 1::2, 0::2] = 0.5 * (ll - lh + hl - hh)
    out[:, :, 1::2, 1::2] = 0.5 * (ll - lh - hl + hh)
    return out

class WTConvBlock(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        self.conv_ll = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.conv_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = haar_dwt2d(x)
        ll_proc = self.conv_ll(ll)
        return self.conv_proj(x) + haar_idwt2d(ll_proc, lh, hl, hh)

class DeepMorphologicalDilation2d(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        self.se = nn.Parameter(torch.zeros(channels, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        out = torch.full((B, C, H, W), -float("inf"), device=x.device, dtype=x.dtype)
        for kh in range(3):
            for kw in range(3):
                val = padded[:, :, kh:kh + H, kw:kw + W] + self.se[:, kh, kw].view(1, C, 1, 1)
                out = torch.maximum(out, val)
        return out"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# 1. Parseval Energy Conservation Invariant Verification
ll, lh, hl, hh = haar_dwt2d(x)
energy_in = torch.sum(x.double() ** 2).item()
energy_subbands = (torch.sum(ll.double() ** 2) + torch.sum(lh.double() ** 2) + torch.sum(hl.double() ** 2) + torch.sum(hh.double() ** 2)).item()
parseval_err = abs(energy_in - energy_subbands)
assert parseval_err < 1e-5, f"Parseval invariant violated: err={parseval_err:.2e}"

# 2. Perfect Invertibility Verification (DWT + IDWT)
x_rec = haar_idwt2d(ll, lh, hl, hh)
idwt_err = float(torch.max(torch.abs(x - x_rec)))
assert idwt_err < 1e-6, f"IDWT reconstruction error: {idwt_err:.2e}"

# 3. WTConv Golden Parity Verification
wtconv = WTConvBlock(channels=C)
with torch.no_grad():
    wtconv.conv_ll.weight.copy_(weights["wt_ll_w"])
    wtconv.conv_ll.bias.copy_(weights["wt_ll_b"])
    wtconv.conv_proj.weight.copy_(weights["wt_proj_w"])
    wtconv.conv_proj.bias.copy_(weights["wt_proj_b"])
out_wtconv = wtconv(x)
err_wt = float(torch.max(torch.abs(out_wtconv.detach() - weights["wtconv_output"])))
assert err_wt < 1e-6, f"WTConv parity error: {err_wt:.2e}"

# 4. Deep Morphological Dilation Golden Parity Verification
morph = DeepMorphologicalDilation2d(channels=C)
with torch.no_grad():
    morph.se.copy_(weights["morph_se"])
out_morph = morph(x)
err_morph = float(torch.max(torch.abs(out_morph.detach() - weights["morph_output"])))
assert err_morph < 1e-6, f"Morphological dilation error: {err_morph:.2e}"

print(f"Parity Verified: Parseval Err={parseval_err:.2e}, IDWT Err={idwt_err:.2e}, WTConv={err_wt:.2e}, Morph={err_morph:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = r"""import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_dwt = benchmark_fn(lambda: haar_dwt2d(x), iters=50, name="2D Haar DWT")
b_wt = benchmark_fn(lambda: wtconv(x), iters=50, name="WTConv Forward")
b_morph = benchmark_fn(lambda: morph(x), iters=50, name="Morphological Dilation")
print(format_benchmark_table([b_dwt, b_wt, b_morph]))

# Visualize Haar Frequency Subbands & Morphological Boundary Dilation
fig, ax = plt.subplots(1, 5, figsize=(14, 2.8), dpi=100)
ax[0].imshow(x[0, 0].numpy(), cmap="viridis")
ax[0].set_title("Input (8x8)")
ax[1].imshow(ll[0, 0].numpy(), cmap="plasma")
ax[1].set_title("LL Approx (4x4)")
ax[2].imshow(lh[0, 0].numpy(), cmap="magma")
ax[2].set_title("LH Horiz (4x4)")
ax[3].imshow(hl[0, 0].numpy(), cmap="cividis")
ax[3].set_title("HL Vert (4x4)")
ax[4].imshow(out_morph[0, 0].detach().numpy(), cmap="inferno")
ax[4].set_title("Dilated Output")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Unscaled Discrete Wavelet Transform (Omitting 0.5 Factor)
# Omitting the orthonormal 1/2 factor (e.g. using raw addition without division)
# quadruples energy (2x magnitude per axis) at each wavelet level, leading to exponential explosion!

x_test = x.clone()
x_norm = x.clone()

with torch.no_grad():
    for level in range(4):
        # Orthonormal DWT
        ll_norm, _, _, _ = haar_dwt2d(x_norm)
        x_norm = F.interpolate(ll_norm, scale_factor=2.0)  # Upsample to maintain spatial size
        
        # Unscaled DWT (missing 0.5 multiplier)
        x00, x01 = x_test[:, :, 0::2, 0::2], x_test[:, :, 0::2, 1::2]
        x10, x11 = x_test[:, :, 1::2, 0::2], x_test[:, :, 1::2, 1::2]
        ll_unscaled = x00 + x01 + x10 + x11
        x_test = F.interpolate(ll_unscaled, scale_factor=2.0)

norm_ortho = float(torch.norm(x_norm))
norm_unscaled = float(torch.norm(x_test))

print(f"Signal Norm after 4 Levels (Orthonormal): {norm_ortho:.2f}")
print(f"Signal Norm after 4 Levels (Unscaled):    {norm_unscaled:.2f}")
print(f"Explosion Factor: {norm_unscaled / norm_ortho:.1f}x")

assert norm_unscaled > norm_ortho * 10.0, "Unscaled DWT must cause catastrophic energy explosion!"
print("Demonstrated catastrophic signal explosion from unscaled wavelet transforms.")"""
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Multi-Scale Frequency Decomposition at $\\mathcal{O}(1)$ Overhead:** WTConv decomposes feature maps into orthogonal frequency subbands with zero learned downsampling parameters, allowing deep models to operate at broad receptive fields.
2. **Parseval Energy Conservation:** Orthonormal wavelets strictly conserve $L_2$ signal energy ($\\|X\\|_F^2 = \\sum \\|subbands\\|_F^2$), preventing the gradient explosion/decay typical of unconstrained multi-scale pyramids.
3. **Tropical Algebra for Boundary Geometry:** Deep morphological operations replace linear multiplication with max-plus addition, providing exact translation-equivariant geometric boundary extraction robust to non-additive noise.

---
### Cross-Domain Generalisation Matrix
- **Medical Imaging & Ultrasound:** Wavelet scattering extracts high-frequency microcalcifications and tissue speckle patterns without training large CNNs on scarce datasets.
- **Seismic & Geomechanical Subsurface Modeling:** Multi-scale DWT captures layered acoustic velocity discontinuities across heterogeneous geological strata.
- **Edge Devices & Neuromorphic Computing:** Tropical morphological dilation operates entirely with comparison and addition operations, eliminating floating-point multiplication units.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [05_wavelet_scattering_wtconv_steerable_and_morphological_networks.md](../../markdown/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.md)
- **PyTorch 2.x Module:** [05_wavelet_scattering_wtconv_steerable_and_morphological_networks.py](../../pytorch2/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.py)
- **Functional JAX Module:** [05_wavelet_scattering_wtconv_steerable_and_morphological_networks.py](../../jax/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.py)
- **Native Rust Candle Track:** [05_wavelet_scattering_wtconv_steerable_and_morphological_networks.rs](../../candle/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.rs)
- **Hardware WGSL Kernel:** [05_wavelet_scattering_wtconv_steerable_and_morphological_networks.rs](../../kernels/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.rs)
- **Universal Golden SafeTensors:** [05_wavelet_scattering_wtconv_steerable_and_morphological_networks.safetensors](../../weights/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 04. Spatiotemporal Video (I3D & TimeSformer)](04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.ipynb) | [Category Index](../README.md) | [Next Subfolder: 03. Recurrent, SSM & Memory](../03_recurrent_ssm_and_memory/01_lstm_gru_and_xlstm.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/02_convolutional_and_spatiotemporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "05_wavelet_scattering_wtconv_steerable_and_morphological_networks.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
