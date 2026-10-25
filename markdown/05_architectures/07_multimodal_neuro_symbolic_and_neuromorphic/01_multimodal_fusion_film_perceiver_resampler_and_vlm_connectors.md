# Multimodal Fusion: FiLM, Perceiver Resampler, and VLM Connectors

## 1. Executive Summary & Architectural Overview

Modern vision-language models (VLMs) and multimodal perception systems bridge heterogeneous modalities—such as high-resolution vision grids, continuous audio spectrograms, and discrete token sequences. The central design challenge is **representation impedance matching**: raw visual features (e.g. 576 ViT patches per image) present excessive sequence lengths and mismatched geometric topologies compared to autoregressive language model embeddings.

This topic analyzes and implements three foundational multimodal fusion paradigms:
1. **Feature-wise Linear Modulation (FiLM)**: An affine transformation where conditioning vectors dynamically scale and shift intermediate feature maps ($\gamma \odot x + \beta$).
2. **Perceiver Resampler**: A fixed-length cross-attention bottleneck (pioneered in Flamingo) where $K$ learned query latents attend to variable-length visual tokens, compressing token counts by an order of magnitude.
3. **VLM Spatial Pooling Connectors**: Pixel-unshuffle and MLP projection mechanisms (utilized in LLaVA, LLaVA-NeXT, and Qwen-VL) that bundle local $2 \times 2$ patch clusters into unified language tokens while preserving spatial adjacency.

Together, these mechanisms span the design spectrum from tight parametric conditioning (FiLM) to flexible set-to-set attention routing (Perceiver) and compute-efficient spatial compression (VLM MLPs).

---

## 2. Mathematical Foundations & Governing Equations

### 2.1 Feature-wise Linear Modulation (FiLM)
Given an input feature representation $x \in \mathbb{R}^{B \times L \times D}$ and conditioning embedding $c \in \mathbb{R}^{B \times D_c}$, conditioning projections produce modulation vectors $\gamma(c), \beta(c) \in \mathbb{R}^{B \times 1 \times D}$:

$$\text{FiLM}(x; \gamma, \beta) = \gamma(c) \odot x + \beta(c)$$

FiLM acts as an input-dependent dynamic normalization layer, generalizing conditional batch normalization to arbitrary tensor shapes and sequences.

### 2.2 Perceiver Resampler Cross-Attention
Given visual tokens $X_v \in \mathbb{R}^{B \times L \times D_v}$ and learned latent queries $Z \in \mathbb{R}^{B \times K \times D}$ with $K \ll L$:

$$Q = Z W_Q, \quad K = X_v W_K, \quad V = X_v W_V$$

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{D}}\right) V$$

The computational complexity scales as $\mathcal{O}(B \cdot K \cdot L \cdot D)$, decoupled from quadratic sequence length $\mathcal{O}(L^2)$, yielding exactly $K$ prefix tokens for the LLM.

### 2.3 Spatial Pixel Unshuffle & Connector MLP
For a $2 \times 2$ patch grid reshuffled to compress sequence length by factor 4:

$$x_{\text{unshuffle}} = \text{Reshape}(X_{\text{patches}}, [B, 1, 4D])$$

$$h_{\text{vlm}} = \text{GELU}(x_{\text{unshuffle}} W_1 + b_1) W_2 + b_2$$

---

## 3. Tensor Memory Layout & Implementation Details

| Component | Tensor Name | Shape | Layout / Memory Semantics |
| :--- | :--- | :--- | :--- |
| **FiLM Input** | `x_feat` | `[B, L, D]` | Contiguous float32 feature tensor |
| **FiLM Params** | `gamma`, `beta` | `[B, 1, D]` | Broadcastable scale and shift vectors |
| **Resampler Queries** | `latent_queries` | `[B, K, D]` | Learned query tokens, $K \ll L$ |
| **Resampler Weights** | `w_q`, `w_k`, `w_v` | `[D, D]` | Standard cross-attention projection matrices |
| **VLM Patches** | `patches` | `[B, 4, D]` | Local spatial patch cluster before unshuffle |
| **VLM Connector** | `w1`, `b1`, `w2`, `b2` | `[4D, 16]`, `[16]`, `[16, D_llm]`, `[D_llm]` | 2-layer MLP projection into language dimension |

---

## 4. Multi-Ecosystem Implementation Matrix

| Framework | Implementation Module | Execution Model | Memory Strategy |
| :--- | :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/.../01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.py` | Eager / `torch.compile` | Broadcasted elementwise ops and F.softmax |
| **JAX** | `jax/.../01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.py` | Functional `jax.jit` | Pure functions over `jax.Array` |
| **Candle** | `candle/.../01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.rs` | Rust zero-cost safety | 2D/3D reshaped matmuls, `broadcast_mul` |
| **WGSL / CPU** | `kernels/.../01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.rs` | WebGPU compute / SIMD | Fused parallel compute workgroups |

---

## 5. Pathological Misuse & Failure Modes

1. **FiLM Degeneracy from Zero-Scale Collapsing**: If $\gamma \to 0$, visual/feature information is completely erased, leaving only the additive bias $\beta$. Initialization must center $\gamma$ around $1.0$.
2. **Perceiver Attention Entropy Collapse**: Unconstrained cross-attention across long visual contexts can over-allocate attention mass to background tokens. Temperature scaling and learned positional encodings are mandatory.
3. **Spatial Unshuffle Dimension Mismatch**: Unshuffling must preserve 2D grid coordinates $(2y, 2x), (2y, 2x+1), \dots$; arbitrary token flattening scrambles spatial semantics before the LLM can interpret positional cues.

---

## 6. Cross-Domain Generalisation Matrix

| Domain | Conditioning / Query | Target Feature Stream | Operational Role |
| :--- | :--- | :--- | :--- |
| **Robotics (RT-2, Octo)** | Language goal instruction | Gripper & wrist camera video tokens | FiLM conditioning on robot control policy |
| **Vision-Language (LLaVA)** | 2x2 spatial patch unshuffle | ViT patch tokens | Token reduction for autoregressive LLM |
| **Audio-Visual Video** | Text query latents | Video + audio multichannel streams | Multimodal Perceiver bottleneck |
| **Diffusion (ControlNet)** | Conditioning image features | Diffusion U-Net / DiT hidden states | Zero-conv & FiLM spatial modulation |

---

## 7. Verification, Parity & Empirical Benchmarks

All operators are verified against the deterministic golden checkpoint:
- `weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.safetensors` (Seed 42)
- Max numerical discrepancy across PyTorch, JAX, and Candle: $\Delta < 1.0 \times 10^{-6}$.
- Candle test `test_39_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors_parity` executed cleanly with 0 failures.

---

## 8. Synchronized Track Artifacts & Navigation

- PyTorch: `pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.py`
- JAX: `jax/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.py`
- Candle: `candle/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.rs`
- Kernels: `kernels/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.rs`
- Weights: `weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.safetensors`
- Notebook: `notebooks/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.ipynb`
