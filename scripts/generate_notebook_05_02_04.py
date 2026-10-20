"""Generate 8-cell educational notebook for Category 05 Subfolder 02 Topic 04:
04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.ipynb in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 04. Spatiotemporal Video: I3D Weight Inflation & TimeSformer Factorized Space-Time Attention

- **Category:** 05. Architectures / 02. Convolutional & Spatiotemporal
- **Lineage & Citations:** Joao Carreira & Andrew Zisserman (CVPR 2017) *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (I3D)*; Gedas Bertasius et al. (ICML 2021) *Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)*; Anurag Arnab et al. (ICCV 2021) *ViViT: A Video Vision Transformer*.
- **Purpose:** Solves video representation scaling by bootstrapping pretrained 2D image models into 3D spatiotemporal networks via $1/K_T$ weight inflation (I3D) and reducing full spacetime attention complexity from $\\mathcal{O}((T \\cdot S)^2)$ down to $\\mathcal{O}(T S^2 + S T^2)$ via divided space-time attention (TimeSformer).
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext](01_classic_to_modern_cnns_resnet_densenet_convnext.ipynb), [05_architectures/04_transformers_and_heads/01_multi_head_multi_query_grouped_query_attention](../04_transformers_and_heads/01_multi_head_multi_query_grouped_query_attention.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks](05_wavelet_scattering_wtconv_steerable_and_morphological_networks.ipynb), [07_domain_tasks/13_robotics_and_motor_control_diffusion_policy_act_and_mppi](../../07_domain_tasks/13_robotics_and_motor_control_diffusion_policy_act_and_mppi.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never inflate 2D weights into 3D by simple repetition without dividing by $K_T$; unscaled inflation multiplies activation variance by $K_T\\times$ at every single layer, causing catastrophic numerical explosion (`NaN`) at step 0! Never compute unconstrained joint space-time attention for videos longer than 8 frames ($T \\ge 16$); full attention memory scales quadratically with both temporal and spatial resolution."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = r"""## Mathematical Foundations & Governing Equations

### 1. I3D Weight Inflation Invariance
Inflate a 2D pretrained convolution filter $\mathbf{W}_{2D} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W}$ into a 3D spatiotemporal filter:
$$\mathbf{W}_{3D}(:, :, t, :, :) = \frac{1}{K_T} \mathbf{W}_{2D}(:, :, :, :), \quad \forall t \in \{1, \dots, K_T\}$$
**Static Video Mathematical Invariant:**
Given a static video $\mathbf{X}_{\text{video}} \in \mathbb{R}^{B \times C_{\text{in}} \times T \times H \times W}$ where all temporal frames are identical copies of image $\mathbf{X}_{2D}$:
$$\mathbf{Y}_{3D}(:, :, t, :, :) = \sum_{k=1}^{K_T} \frac{1}{K_T} \mathbf{W}_{2D} * \mathbf{X}_{2D} + \mathbf{b} = \mathbf{W}_{2D} * \mathbf{X}_{2D} + \mathbf{b}$$
Because $\sum_{k=1}^{K_T} \frac{1}{K_T} \equiv 1$, the 3D convolution on static video preserves the exact numerical activations of the 2D model.

### 2. TimeSformer Divided Space-Time Attention
Given patch tokens $\mathbf{Z} \in \mathbb{R}^{B \times T \times S \times D}$:
1. **Spatial Self-Attention (within frames):**
   $$\mathbf{Z}_S = \text{Reshape}(\mathbf{Z}, (B \cdot T, S, D)), \quad \mathbf{Z}' = \mathbf{Z} + \text{Attention}(\mathbf{Z}_S)$$
2. **Temporal Self-Attention (across frames):**
   $$\mathbf{Z}_T = \text{Reshape}(\text{Permute}(\mathbf{Z}', (B, S, T, D)), (B \cdot S, T, D)), \quad \mathbf{Z}'' = \mathbf{Z}' + \text{Permute}(\text{Attention}(\mathbf{Z}_T), (B, T, S, D))$$

### 3. Asymptotic Time & Memory Complexity
- **Full Joint Spacetime Attention:** $\mathcal{O}((T \cdot S)^2 \cdot D) = \mathcal{O}(T^2 S^2 D)$ FLOPs.
- **Divided Space-Time Attention:** $\mathcal{O}(T \cdot S^2 \cdot D + S \cdot T^2 \cdot D)$ FLOPs ($>10\times$ faster for typical $T=16, S=196$)."""
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
ckpt_rel = "weights/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x_video = weights["x_video"]
tokens = weights["tokens"]
B, C, T, H, W = x_video.shape
print(f"Loaded I3D & TimeSformer checkpoint on {env.device}: Video={x_video.shape}, Tokens={tokens.shape}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """def i3d_inflate_weights(conv2d_w: torch.Tensor, k_t: int = 3) -> torch.Tensor:
    return conv2d_w.unsqueeze(2).repeat(1, 1, k_t, 1, 1) / float(k_t)

class FactorizedSpaceTimeAttention(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim
        self.scale = dim ** 0.5

        self.spatial_q = nn.Linear(dim, dim, bias=False)
        self.spatial_k = nn.Linear(dim, dim, bias=False)
        self.spatial_v = nn.Linear(dim, dim, bias=False)
        self.spatial_out = nn.Linear(dim, dim, bias=False)

        self.temporal_q = nn.Linear(dim, dim, bias=False)
        self.temporal_k = nn.Linear(dim, dim, bias=False)
        self.temporal_v = nn.Linear(dim, dim, bias=False)
        self.temporal_out = nn.Linear(dim, dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, S, D = z.shape

        # 1. Spatial Attention across S tokens
        z_s = z.view(B * T, S, D)
        q_s, k_s, v_s = self.spatial_q(z_s), self.spatial_k(z_s), self.spatial_v(z_s)
        attn_s = F.softmax((q_s @ k_s.transpose(-2, -1)) / self.scale, dim=-1)
        out_s = self.spatial_out(attn_s @ v_s).view(B, T, S, D)
        z_mid = z + out_s

        # 2. Temporal Attention across T frames
        z_t = z_mid.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        q_t, k_t, v_t = self.temporal_q(z_t), self.temporal_k(z_t), self.temporal_v(z_t)
        attn_t = F.softmax((q_t @ k_t.transpose(-2, -1)) / self.scale, dim=-1)
        out_t = self.temporal_out(attn_t @ v_t).view(B, S, T, D).permute(0, 2, 1, 3).contiguous()

        return z_mid + out_t"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# 1. I3D Inflation & Static Video Invariance Verification
inflated_w = i3d_inflate_weights(weights["conv2d_w"], k_t=3)
err_inflate = float(torch.max(torch.abs(inflated_w - weights["i3d_w"])))
assert err_inflate < 1e-5, f"I3D inflation error: {err_inflate:.2e}"

out_3d = F.conv3d(x_video, weights["i3d_w"], weights["i3d_b"], padding=(1, 1, 1))
err_3d = float(torch.max(torch.abs(out_3d.detach() - weights["y_3d"])))
assert err_3d < 1e-5, f"3D conv parity error: {err_3d:.2e}"

# 2. Factorized Space-Time Attention Verification
timesformer = FactorizedSpaceTimeAttention(dim=8)
with torch.no_grad():
    timesformer.spatial_q.weight.copy_(weights["spatial_q_w"].t())
    timesformer.spatial_k.weight.copy_(weights["spatial_k_w"].t())
    timesformer.spatial_v.weight.copy_(weights["spatial_v_w"].t())
    timesformer.spatial_out.weight.copy_(weights["spatial_out_w"].t())
    timesformer.temporal_q.weight.copy_(weights["temporal_q_w"].t())
    timesformer.temporal_k.weight.copy_(weights["temporal_k_w"].t())
    timesformer.temporal_v.weight.copy_(weights["temporal_v_w"].t())
    timesformer.temporal_out.weight.copy_(weights["temporal_out_w"].t())

out_tf = timesformer(tokens)
err_tf = float(torch.max(torch.abs(out_tf.detach() - weights["timesformer_output"])))
assert err_tf < 1e-5, f"TimeSformer parity error: {err_tf:.2e}"

print(f"Parity Verified: I3D Inflation={err_inflate:.2e}, 3D Conv={err_3d:.2e}, TimeSformer={err_tf:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = r"""import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_3d = benchmark_fn(lambda: F.conv3d(x_video, weights["i3d_w"], weights["i3d_b"], padding=(1, 1, 1)), iters=50, name="I3D Conv3D Forward")
b_tf = benchmark_fn(lambda: timesformer(tokens), iters=50, name="TimeSformer (Space + Time Attn)")
print(format_benchmark_table([b_3d, b_tf]))

# Visualize I3D Static Video Response Invariance
with torch.no_grad():
    y_2d = F.conv2d(weights["x_static_2d"], weights["conv2d_w"], weights["conv2d_b"], padding=1)
    y_3d_center = out_3d[:, :, 1]
    diff_slice = torch.abs(y_3d_center - y_2d)[0, 0].numpy()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=100)
im0 = ax[0].imshow(y_2d[0, 0].numpy(), cmap="plasma")
ax[0].set_title("2D Conv Activation on Static Image")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].imshow(diff_slice, cmap="magma")
ax[1].set_title("I3D Invariance Error $|Y_{3D}(t=1) - Y_{2D}|$")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Unscaled 3D Weight Inflation (Omitting 1/K_T Scaling)
# Inflating 2D weights into 3D by repeating K_T times without the 1/K_T scaling factor
# multiplies activation variance by K_T at every layer, causing exponential signal explosion!

w_2d = weights["conv2d_w"]
K_T = 3

# Proper Scaled Inflation vs Naive Unscaled Inflation
w_scaled = w_2d.unsqueeze(2).repeat(1, 1, K_T, 1, 1) / float(K_T)
w_unscaled = w_2d.unsqueeze(2).repeat(1, 1, K_T, 1, 1)  # Missing 1/K_T!

# Simulate a 10-layer deep network forward pass
h_scaled = x_video.clone()
h_unscaled = x_video.clone()

with torch.no_grad():
    for layer in range(5):
        h_scaled = F.conv3d(h_scaled, w_scaled, padding=(1, 1, 1))
        h_unscaled = F.conv3d(h_unscaled, w_unscaled, padding=(1, 1, 1))

norm_scaled = float(torch.norm(h_scaled))
norm_unscaled = float(torch.norm(h_unscaled))

print(f"Activation Norm after 5 Layers (Scaled 1/K_T):   {norm_scaled:.2f}")
print(f"Activation Norm after 5 Layers (Naive Unscaled): {norm_unscaled:.2f}")
print(f"Explosion Ratio: {norm_unscaled / norm_scaled:.1f}x")

assert norm_unscaled > norm_scaled * 10.0, "Unscaled inflation must cause severe activation magnitude explosion!"
print("Demonstrated catastrophic signal explosion from unscaled 3D weight inflation.")"""
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Mathematical Invariance via Scaling:** I3D weight inflation strictly requires dividing by $K_T$ to preserve the response of 2D pretrained backbones on static video sequences.
2. **Divided Space-Time Efficiency:** TimeSformer and ViViT reduce video attention complexity from $\\mathcal{O}((T \\cdot S)^2)$ to $\\mathcal{O}(T S^2 + S T^2)$, enabling scalable training on high-framerate clips.
3. **Bootstrapping 2D Representations:** Pretrained 2D spatial features provide immediate transferability to 3D video, eliminating the sample inefficiency of training 3D networks from scratch.

---
### Cross-Domain Generalisation Matrix
- **Robotics & Policy Learning (Diffusion Policy):** Divided space-time attention tracks multi-camera robot workspace state history over multi-second horizons.
- **Biomedical 3D/4D Volumetric Imaging (fMRI, 4D CT):** I3D inflation transfers 2D CT segmentation models to dynamic cardiopulmonary motion scans.
- **Meteorological & Climate Forecasting:** Factorized spatio-temporal layers process atmospheric wind and pressure fields over multi-day trajectories.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.md](../../markdown/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.md)
- **PyTorch 2.x Module:** [04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.py](../../pytorch2/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.py)
- **Functional JAX Module:** [04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.py](../../jax/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.py)
- **Native Rust Candle Track:** [04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.rs](../../candle/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.rs)
- **Hardware WGSL Kernel:** [04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.rs](../../kernels/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.rs)
- **Universal Golden SafeTensors:** [04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors](../../weights/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 03. U-Net & Multiscale Feature Pyramids](03_unet_and_multiscale_feature_pyramids.ipynb) | [Category Index](../README.md) | [Next Topic: 05. Wavelet Scattering, WTConv & Morphological Networks](05_wavelet_scattering_wtconv_steerable_and_morphological_networks.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/02_convolutional_and_spatiotemporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
