# Vision Optical Flow, Depth & Super-Resolution

## 1. Evolution of Dense Visual Fields: Motion, Geometry & Restoration

Dense visual perception extends beyond discrete bounding boxes and semantic masks to continuous, dense 2D and 3D field estimation:

1. **Optical Flow Estimation (Horn-Schunck $\to$ FlowNet $\to$ RAFT)**:
   - Estimates per-pixel 2D motion displacement vectors $\mathbf{u} = (u, v)$ across two consecutive frames $I_1, I_2 \in \mathbb{R}^{H \times W \times 3}$.
   - Classical variational formulations solved the optical flow constraint equation $I_x u + I_y v + I_t = 0$ with spatial smoothness regularizers.
   - Modern deep methods (**RAFT**) construct multi-scale **4D all-pairs correlation volumes** and iteratively update flow fields using recurrent gated units (GRU).

2. **Monocular Depth Estimation (Eigen et al. $\to$ DPT $\to$ Depth Anything)**:
   - Infers a dense metric or relative distance map $d \in \mathbb{R}^{H \times W}$ from a single monocular 2D image.
   - Solves scale ambiguity using **Scale-Invariant Logarithmic Loss** formulations, invariant to global scene scaling $\alpha \cdot d$.

3. **Sub-Pixel Super-Resolution (SRCNN $\to$ ESPCN $\to$ ESRGAN)**:
   - Reconstructs high-resolution images $I_{\text{HR}} \in \mathbb{R}^{rH \times rW \times C}$ from low-resolution observations $I_{\text{LR}}$.
   - Replaced lossy bicubic upsampling with **Sub-Pixel Convolutions (PixelShuffle)**, shifting feature expansion to the final layer.

---

## 2. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

Teed and Deng (2020) revolutionized optical flow with **RAFT (Recurrent All-Pairs Field Transforms)**:

```
Frame 1 ──> Feature Encoder ──> F_1 [B, C, H, W] ──┐
                                                    ├──> All-Pairs 4D Correlation Volume [B, H, W, H, W]
Frame 2 ──> Feature Encoder ──> F_2 [B, C, H, W] ──┘         │
                                                             ▼
                                                    Correlation Pyramid Lookup
                                                             │
Context Net ──> Context Features [B, D, H, W] ──────> Iterative GRU Cell ──> Residual Flow Update Δf
                                                             ▲                      │
                                                             └──────────────────────┘
```

### 2.1 4D Correlation Volume Computation
RAFT extracts feature maps $F_1, F_2 \in \mathbb{R}^{H \times W \times C}$ from both frames. The full 4D correlation volume $C \in \mathbb{R}^{H \times W \times H \times W}$ computes the pairwise cosine/dot-product similarity between every pixel $(i, j)$ in $I_1$ and every pixel $(k, l)$ in $I_2$:
$$C_{ijkl} = \frac{1}{\sqrt{C}} \sum_{c=1}^C F_1(c, i, j) \cdot F_2(c, k, l).$$

Unlike coarse-to-fine pyramids that discard small, fast-moving objects at low resolutions, the all-pairs correlation volume preserves all candidate matches simultaneously.

### 2.2 Recurrent GRU Flow Update
The current flow estimate $\mathbf{f}_k$ indexes into the correlation volume via bilinear sampling. A lightweight convolutional GRU predicts a residual flow increment:
$$\mathbf{f}_{k+1} = \mathbf{f}_k + \Delta \mathbf{f}_k.$$
Supervising intermediate flow fields with exponentially decayed weights $\sum_{k=1}^K \gamma^{K-k} \|\mathbf{f}_k - \mathbf{f}^\star\|_1$ guarantees monotonic refinement.

---

## 3. Charbonnier Loss & Robust Photometric Objectives

Optical flow and video restoration suffer from severe outliers caused by motion boundaries, specular reflections, and disocclusions. While standard $L_2$ loss $(\Delta I)^2$ over-penalizes large outlier residuals, $L_1$ loss $|\Delta I|$ has a discontinuous derivative at zero.

The **Charbonnier loss** is a smooth, differentiable approximation to $L_1$:
$$\mathcal{L}_{\text{charb}}(x, y) = \frac{1}{N} \sum_{i=1}^N \sqrt{(x_i - y_i)^2 + \epsilon^2},$$
where $\epsilon > 0$ (typically $10^{-3}$ or $10^{-6}$) acts as a curvature parameter:
- For $|x - y| \ll \epsilon$: Quadratic behavior $\approx \epsilon + \frac{(x - y)^2}{2\epsilon}$ (smooth, well-conditioned gradient at origin).
- For $|x - y| \gg \epsilon$: Linear behavior $\approx |x - y|$ (robust against photometric outliers).

---

## 4. Monocular Depth Estimation & Scale-Invariant Loss

In monocular depth estimation, global camera focal length or scene scale is often ambiguous. Eigen et al. (2014) derived the **Scale-Invariant Logarithmic Depth Loss**:

Let $d_i = \log y_i - \log y_i^\star$ be the difference between predicted log-depth and ground-truth log-depth at pixel $i$.

$$\mathcal{L}_{\text{SI}}(y, y^\star) = \frac{1}{N} \sum_{i=1}^N d_i^2 - \frac{\alpha}{N^2} \left( \sum_{i=1}^N d_i \right)^2,$$
where $\alpha \in [0, 1]$ (standard value $\alpha = 0.5$ or $\alpha = 0.85$):
- When $\alpha = 0$, the loss reduces to standard mean squared error in log-space $\text{MSE}_{\log}$.
- When $\alpha = 1$, the loss is strictly scale-invariant: multiplying predicted depth by an arbitrary scalar constant $c > 0$ ($y \to c \cdot y$) leaves $\mathcal{L}_{\text{SI}}$ unchanged since $\log(c y) - \log(y^\star) = d_i + \log c$, and:
  $$\frac{1}{N} \sum (d_i + \log c)^2 - \frac{1}{N^2} \left( \sum (d_i + \log c) \right)^2 = \frac{1}{N} \sum d_i^2 - \frac{1}{N^2} \left( \sum d_i \right)^2.$$

---

## 5. Sub-Pixel Convolution (PixelShuffle) for Super-Resolution

Shi et al. (2016) proposed the Efficient Sub-Pixel Convolutional Neural Network (**ESPCN**).

Traditional super-resolution upscaled low-resolution input images before convolution, wasting computation in high-resolution feature spaces. ESPCN extracts features entirely in the low-resolution space and rearranges multi-channel feature maps into spatial pixels in a single step:

$$\operatorname{PixelShuffle}(T)_{b, c, y, x} = T_{b, \ c \cdot r^2 + r \cdot (y \bmod r) + (x \bmod r), \ \lfloor y / r \rfloor, \ \lfloor x / r \rfloor},$$
transforming a tensor of shape $[B, C \cdot r^2, H, W]$ into $[B, C, H \cdot r, W \cdot r]$.

### 5.1 Benefits over Transposed Convolutions
- **Eliminates Checkerboard Artifacts**: Transposed convolutions (deconvolutions) suffer from uneven kernel overlap when kernel size is not divisible by stride.
- **Maximum Computational Efficiency**: All heavy convolutional matrix multiplications run at low resolution $H \times W$, saving $(r^2)\times$ FLOPs and memory.

---

## 6. Multi-Task Synergies: Geometry, Flow & NeRF

Dense visual tasks share deep geometric dualities:
1. **Optical Flow $\leftrightarrow$ Rigid Scene Flow**: Given camera intrinsics $K$ and depth $D$, optical flow is the 2D projection of 3D camera egomotion $T \in \text{SE}(3)$.
2. **Super-Resolution $\leftrightarrow$ Novel View Synthesis**: NeRF and 3D Gaussian Splatting leverage multi-view optical flow and sub-pixel loss objectives to supervise continuous radiance fields.
3. **Temporal Coherence in Video Generation**: Diffusion video models (Sora, CogVideoX) enforce cross-frame optical flow consistency using Charbonnier losses.

---

## 7. Mathematical Invariants & Roofline Arithmetic Intensity

| Operation | Input Shape | Output Shape | Arithmetic Intensity (FLOPs/Byte) | Bound |
| :--- | :--- | :--- | :--- | :--- |
| **PixelShuffle** | $[B, C \cdot r^2, H, W]$ | $[B, C, H \cdot r, W \cdot r]$ | $0$ (pure memory stride rearrange) | Memory-Bound |
| **Charbonnier Loss** | $2 \times [B, H, W]$ | $[B]$ | $\approx 4$ FLOPs / $8$ Bytes = $0.5$ | Memory-Bound |
| **Scale-Invariant Depth** | $2 \times [B, H, W]$ | $[B]$ | $\approx 8$ FLOPs / $8$ Bytes = $1.0$ | Memory-Bound |
| **All-Pairs Correlation** | $2 \times [B, C, H, W]$ | $[B, H \cdot W, H \cdot W]$ | $2 \cdot C \cdot (HW)^2 / (\text{Bytes})$ | **Compute-Bound** |

---

## 8. Cross-Track Verification & Systems-Level Contracts

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every operator adheres to the following parity guarantees:

| Operator | Verification Metric | Tolerance | Numerical Invariant |
| :--- | :--- | :--- | :--- |
| **PixelShuffle** | Reshaped spatial elements match analytical stride mapping | $< 10^{-5}$ | Total element count preserved: $B \cdot C \cdot r^2 \cdot H \cdot W$ |
| **Scale-Invariant Depth Loss** | Logarithmic scale-invariant formula with $\alpha = 0.5$ | $< 10^{-5}$ | Non-negative: $\mathcal{L}_{\text{SI}} \ge 0$ |
| **Charbonnier Loss** | Element-wise $\sqrt{(p - t)^2 + \epsilon^2}$ spatial mean | $< 10^{-5}$ | Strictly positive: $\mathcal{L}_{\text{charb}} \ge \epsilon$ |
| **RAFT Correlation Matrix** | Normalized dot product $F_1^T F_2 / \sqrt{C}$ | $< 10^{-5}$ | Symmetric for $F_1 = F_2$: $C_{ij} = C_{ji}$ |
| **Linear Anchor** | Shared cross-track regression anchor | $< 10^{-5}$ | Verified against golden checkpoint |


