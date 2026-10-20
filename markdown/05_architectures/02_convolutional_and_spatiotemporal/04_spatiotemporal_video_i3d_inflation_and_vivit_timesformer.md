# Spatiotemporal Video: I3D Weight Inflation & TimeSformer Factorized Space-Time Attention

## Executive Summary & ELI5
Extending deep vision architectures from 2D images to 3D spatiotemporal video sequences poses two major challenges: **training from scratch requires massive video datasets**, and **joint space-time self-attention suffers from quartic complexity $\mathcal{O}((T \cdot S)^2)$**.
1. **I3D (Inflated 3D ConvNet)** (*Carreira & Zisserman, 2017*): Bootstraps successful 2D ImageNet models into 3D video models by **inflating 2D convolutional filters into 3D** along the temporal dimension ($K_T$), scaling weights by $\frac{1}{K_T}$. This guarantees that when given a static video (identical frames repeated across time), the 3D network produces the **exact same activations** as the original 2D network on a single image.
2. **TimeSformer & ViViT (Factorized Space-Time Attention)** (*Bertasius et al., 2021; Arnab et al., 2021*): Replaces dense space-time attention across all $T \cdot S$ tokens with **factorized attention**:
   - **Divided Spatial Attention:** Tokens attend to all other spatial tokens within the same temporal frame ($T \times S^2$).
   - **Divided Temporal Attention:** Tokens attend to the same spatial token across all temporal frames ($S \times T^2$).
   This reduces compute from $\mathcal{O}(T^2 S^2)$ down to $\mathcal{O}(T S^2 + S T^2)$—enabling $>10\times$ faster training and inference on long video sequences.

```mermaid
flowchart TD
    subgraph I3DInflation ["I3D Weight Inflation"]
        W2D["2D Pretrained Filter W_2D: [C_out, C_in, K_h, K_w]"] --> Repeat["Repeat along Time: K_t copies"]
        Repeat --> Scale["Scale by 1 / K_t"]
        Scale --> W3D["Inflated 3D Filter W_3D: [C_out, C_in, K_t, K_h, K_w]"]
    end

    subgraph TimeSformerBlock ["Factorized Space-Time Attention (TimeSformer)"]
        Tokens["Input Tokens Z: [B, T, S, D]"] --> FlatS["Reshape to [(B*T), S, D]"]
        FlatS --> AttnS["Spatial Self-Attention"]
        AttnS --> ResS["Residual Add: Z + Out_S"]
        ResS --> TransT["Transpose & Reshape to [(B*S), T, D]"]
        TransT --> AttnT["Temporal Self-Attention"]
        AttnT --> ResT["Residual Add: Z_mid + Out_T"]
        ResT --> OutTokens["Spatiotemporal Tokens: [B, T, S, D]"]
    end
```

---

## Historical Citations & Seminal Literature
1. **Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (I3D)** (*Joao Carreira, Andrew Zisserman, CVPR 2017, arXiv:1705.07750*). Formulated 2D-to-3D weight inflation with $1/K_T$ scaling to transfer ImageNet features directly to video.
2. **Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)** (*Gedas Bertasius, Heng Wang, Lorenzo Torresani, ICML 2021, arXiv:2102.05095*). Introduced divided space-time attention for video Transformers.
3. **ViViT: A Video Vision Transformer** (*Anurag Arnab et al., ICCV 2021, arXiv:2103.15691*). Explored factorized encoder architectures and spatio-temporal tubelet embeddings for video classification.

---

## Mathematical Foundations & Governing Equations

### 1. I3D Weight Inflation Invariance
Let $\mathbf{W}_{2D} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W}$ be a pretrained 2D convolution kernel. The inflated 3D kernel $\mathbf{W}_{3D} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_T \times K_H \times K_W}$ is constructed by:
$$\mathbf{W}_{3D}(:, :, t, :, :) = \frac{1}{K_T} \mathbf{W}_{2D}(:, :, :, :), \quad \forall t \in \{1, \dots, K_T\}$$
**Static Video Mathematical Invariant:**
Given a static video $\mathbf{X}_{\text{video}} \in \mathbb{R}^{B \times C_{\text{in}} \times T \times H \times W}$ where all temporal frames are identical copies of image $\mathbf{X}_{2D}$ ($\mathbf{X}(:, :, t, :, :) \equiv \mathbf{X}_{2D}$):
$$\mathbf{Y}_{3D}(:, :, t, :, :) = \sum_{k=1}^{K_T} \mathbf{W}_{3D}(:, :, k, :, :) * \mathbf{X}_{2D} + \mathbf{b} = \left(\sum_{k=1}^{K_T} \frac{1}{K_T}\right) \mathbf{W}_{2D} * \mathbf{X}_{2D} + \mathbf{b} = \mathbf{W}_{2D} * \mathbf{X}_{2D} + \mathbf{b}$$
Because $\sum_{k=1}^{K_T} \frac{1}{K_T} \equiv 1$, the 3D convolution on static video is mathematically identical to 2D convolution.

### 2. TimeSformer Divided Space-Time Attention
Given patch tokens $\mathbf{Z} \in \mathbb{R}^{B \times T \times S \times D}$ ($T$ frames, $S$ spatial patches per frame, embedding dimension $D$):
1. **Divided Spatial Attention:**
   $$\mathbf{Z}_S = \text{Reshape}(\mathbf{Z}, (B \cdot T, S, D))$$
   $$\mathbf{A}_S = \text{Softmax}\left(\frac{\mathbf{Q}_S \mathbf{K}_S^\top}{\sqrt{D}}\right) \mathbf{V}_S, \quad \mathbf{Z}' = \mathbf{Z} + \mathbf{W}_O^{(S)} \mathbf{A}_S$$
2. **Divided Temporal Attention:**
   $$\mathbf{Z}_T = \text{Reshape}\left(\text{Permute}(\mathbf{Z}', (B, S, T, D)), (B \cdot S, T, D)\right)$$
   $$\mathbf{A}_T = \text{Softmax}\left(\frac{\mathbf{Q}_T \mathbf{K}_T^\top}{\sqrt{D}}\right) \mathbf{V}_T, \quad \mathbf{Z}'' = \mathbf{Z}' + \text{Permute}\left(\text{Reshape}(\mathbf{W}_O^{(T)} \mathbf{A}_T, (B, S, T, D)), (B, T, S, D)\right)$$

### 3. Asymptotic Time & Memory Complexity
- **Full Joint Spacetime Attention:** $\mathcal{O}((T \cdot S)^2 \cdot D) = \mathcal{O}(T^2 S^2 D)$ FLOPs and $\mathcal{O}(T^2 S^2)$ attention memory (intractable for $T \ge 16, S \ge 196$).
- **Divided Space-Time Attention:** $\mathcal{O}(T \cdot S^2 \cdot D + S \cdot T^2 \cdot D)$ FLOPs—reducing compute and memory by $>10\times$ while matching or exceeding classification accuracy.

---

## Decision Boundaries & Misuse Anti-Patterns
> [!WARNING]
> **Decision Boundary: When NOT to Use It**
> - **Unscaled 3D Inflation:** Never inflate 2D weights by simple repetition without dividing by $K_T$ ($\mathbf{W}_{3D} = \mathbf{W}_{2D}$). Omitting the $\frac{1}{K_T}$ factor multiplies activation variance by $K_T\times$ at every single layer, causing catastrophic gradient explosion and `NaN` at step 0!
> - **Joint Spacetime Attention on Long Videos:** Never use unconstrained joint space-time attention for videos longer than 8 frames; memory consumption scales quadratically with both temporal and spatial resolution.

---

## Synchronized Track Artifacts
- **Universal SafeTensors Checkpoint:** [`weights/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors`](../../../weights/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors)
- **PyTorch 2.x Module:** [`pytorch2/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.py`](../../../pytorch2/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.py)
- **Functional JAX Module:** [`jax/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.py`](../../../jax/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.py)
- **Candle Rust Module:** [`candle/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.rs`](../../../candle/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.rs)
- **Hardware Kernel:** [`kernels/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.rs`](../../../kernels/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.rs)
- **Interactive Concept Notebook:** [`notebooks/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.ipynb`](../../../notebooks/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.ipynb)
