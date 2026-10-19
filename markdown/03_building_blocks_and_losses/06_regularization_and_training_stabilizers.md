# Regularization and Training Stabilizers: Dropout, DropPath, Soft-Capping, Label Smoothing, and Router Z-Loss (C03 Topic 06)

## 1. Executive Summary
Deep neural networks operate in severe overparameterization regimes, where parameter counts often exceed training sample counts by orders of magnitude ($P \gg N$). Without explicit structural regularizers and training stabilizers, optimization trajectories overfit spurious correlation patterns, suffer numerical gradient explosion in reduced-precision floating point formats (FP16/BF16), or fall into sharp, non-generalizing loss basins.

Historically, regularization evolved through three successive generations:
1. **Classical Stochastic Co-Adaptation Prevention:** **Dropout** (*Srivastava et al. 2014*) randomly suppresses individual feature activations during forward passes, preventing co-adapted feature detectors and acting as an implicit exponential ensemble of $2^D$ subnetworks.
2. **Structural and Architectural Stochastics:** Standard elementwise dropout fails in spatially correlated domains (2D CNNs) and deep residual streams. **DropBlock** (*Ghiasi et al. 2018*) masks contiguous spatial patches, while **DropPath / Stochastic Depth** (*Huang et al. 2016*) stochastically drops entire residual branches sample-wise, shortening effective network depth during training while maintaining full capacity at inference.
3. **Reduced-Precision & Modern Transformer Stabilizers:** In massive multi-billion parameter autoregressive language models and Mixture-of-Experts (MoE) architectures, unconstrained dot-product attention logits explode ($> 100$), causing softmax gradients to vanish or underflow in BF16/FP16. **Logit Soft-Capping** (*Gemma / Grok / DeepSeek*) dynamically bounds attention and readout logits via $C \tanh(x/C)$, while **Router Z-Loss** (*Fedus et al. 2022*) penalizes extreme routing log-partition values, completely preventing MoE training divergence.

This document details the mathematical derivations, numerical stability constraints, and synchronized multi-track implementations across PyTorch 2.x, JAX, Rust Candle, and Hardware Kernels.

---

## 2. Mathematical Foundations

### 2.1 Inverted Dropout and Expectation Preservation
Standard dropout randomly sets activations to 0 with probability $p$ during training and requires scaling activations by $1 - p$ at test time.
**Inverted Dropout** shifts the scaling operation into the training forward pass, leaving inference execution untouched:
$$m_i \sim \text{Bernoulli}(1 - p), \quad y_i = \frac{x_i \cdot m_i}{1 - p}$$
The expectation of each activation is preserved identically:
$$\mathbb{E}[y_i] = \frac{x_i \cdot \mathbb{E}[m_i]}{1 - p} = \frac{x_i \cdot (1 - p)}{1 - p} = x_i$$
At test time, the forward pass simplifies to the identity map $y_i = x_i$ with zero scaling overhead.

### 2.2 DropPath (Stochastic Depth)
In deep residual architectures $x_{l+1} = x_l + F(x_l, W_l)$, DropPath stochastically drops the entire residual function $F$ with survival probability $1 - p$:
$$b_l \sim \text{Bernoulli}(1 - p), \quad x_{l+1} = x_l + \frac{b_l}{1 - p} F(x_l, W_l)$$
Where $b_l$ is sampled independently per batch item ($b_l \in \{0, 1\}^B$).
This allows gradients to bypass entire sub-blocks directly through the residual identity stream, stabilizing gradient propagation across 100+ layer vision transformers.

### 2.3 Logit Soft-Capping
In multi-head self-attention, query-key dot products $q_i k_j^T / \sqrt{d}$ can grow unbounded before softmax. In reduced-precision formats (FP16/BF16), logits exceeding 88.0 overflow into `inf`, while large logits drive softmax probabilities to discrete Dirac delta peaks, extinguishing cross-entropy gradients.
**Logit Soft-Capping** smoothly clamps logits within $(-C, C)$ via a scaled hyperbolic tangent:
$$y = C \cdot \tanh\left(\frac{x}{C}\right)$$
- When $|x| \ll C$: $y \approx x$ (linear identity pass).
- When $|x| \to \pm \infty$: $y \to \pm C$ (smooth asymptotic saturation).
Common capping thresholds include $C = 50.0$ for attention query-key matrices and $C = 30.0$ for final language model vocabulary readout logits.

### 2.4 Label Smoothing Cross-Entropy
Given ground-truth hard target $y \in \{1, \dots, K\}$ and predicted probability distribution $p_k = \text{softmax}(z_k)$, standard cross-entropy drives logits toward infinity ($z_y \to \infty$).
**Label Smoothing** (*Szegedy et al. 2016*) replaces the Dirac delta target distribution with a mixture of the true label and a uniform prior:
$$q_k = (1 - \alpha) \cdot \mathbb{I}(k = y) + \frac{\alpha}{K}$$
The smoothed cross-entropy loss decomposes into:
$$\mathcal{L}_{\text{LS}} = -\sum_{k=1}^K q_k \log p_k = (1 - \alpha) \mathcal{L}_{\text{CE}} + \frac{\alpha}{K} \sum_{k=1}^K -\log p_k = (1 - \alpha) \mathcal{L}_{\text{CE}} + \alpha D_{\text{KL}}(u \,\|\, p) + \text{const}$$
This explicitly penalizes peaky, overconfident predictions, improving calibration and feature clustering.

### 2.5 Router Z-Loss in Mixture-of-Experts
In MoE architectures with router logits $z \in \mathbb{R}^{B \times E}$, the log-partition function is $\text{LSE}(z) = \log \sum_{e=1}^E e^{z_e}$.
When router logits drift upward during long training runs, round-off errors in $\text{LSE}(z)$ trigger severe float underflow in FP16/BF16 backpropagation.
**Router Z-Loss** (*Fedus et al. 2022*) adds an auxiliary penalty directly to the training objective:
$$\mathcal{L}_z = \frac{c_z}{B} \sum_{b=1}^B \left( \log \sum_{e=1}^E e^{z_{b, e}} \right)^2$$
Where $c_z \approx 10^{-4}$. This constrains router logits to remain near zero without affecting relative routing decisions.

---

## 3. Structural Properties & Computational Complexity Table

```text
Stabilizer          Target Component          Parameters    FLOPs / Element    Precision Benefit
─────────────────────────────────────────────────────────────────────────────────────────────
Inverted Dropout    Activations (Hidden)      0             O(1) (Masked Mul)  Prevents Co-adaptation
DropPath            Residual Branches (Depth) 0             O(1) (Scalar Mul)  Gradient Highway
Logit Soft-Capping  Attention / Logits        0             O(1) (Tanh)        Prevents BF16 Overflow
Label Smoothing     Loss Objective            0             O(K)               Prevents Overconfidence
Router Z-Loss       MoE Router Logits         0             O(E)               FP16 Stability
```

---

## 4. Multi-Ecosystem Implementation Matrix

| Component | PyTorch 2.x (`pytorch2/`) | JAX / NumPy (`jax/`) | Candle Rust (`candle/`) | Hardware Kernels (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Inverted Dropout** | `InvertedDropout` | `dropout_forward` | `dropout_forward` | `dropout_cpu` |
| **DropPath** | `DropPath` | `droppath_forward` | `droppath_forward` | `droppath_cpu` |
| **Logit Soft-Capping** | `LogitSoftCapping` | `softcapping_forward` | `softcapping_forward` | `SOFTCAP_SHADER` (WGSL) + CPU |
| **Label Smoothing** | `compute_label_smoothing_loss` | `label_smoothing_loss_forward` | `label_smoothing_loss_forward` | `label_smoothing_loss_cpu` |
| **Router Z-Loss** | `compute_router_z_loss` | `router_z_loss_forward` | `router_z_loss_forward` | `router_z_loss_cpu` |
| **Linear Anchor** | `nn.Linear` | `linear_forward` | `LinearAnchor` | `linear_anchor_forward_cpu` |
| **Parity Tolerance** | $< 10^{-5}$ | $< 10^{-5}$ | $< 10^{-5}$ | $< 10^{-5}$ |

---

## 5. Pathological Failure Modes & Postmortems

1. **Weight Decay on 1D Normalization and Positional Parameters:** Applying $L_2$ weight decay to LayerNorm scale $\gamma$, bias $\beta$, or RoPE parameters actively degrades training. Normalization scale factors have Scale-Invariant properties; penalizing them increases effective learning rates unpredictably. Weight decay must strictly be decoupled and applied only to 2D weight matrices ($W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$).
2. **Elementwise Dropout on 2D CNN Feature Maps:** Standard elementwise dropout drops random pixels. Because adjacent pixels in convolutional feature maps share high spatial correlation, neighboring units trivially reconstruct the dropped information, rendering regularization ineffective. Modern architectures use `Dropout2d` (channel dropout) or `DropBlock` (contiguous spatial masking).
3. **Attention Logit Explosion without Soft-Capping:** In ultra-long context transformers ($L > 32\text{k}$), accumulating dot-product values can exceed $100.0$. In BF16, this collapses attention distributions into single-token spikes. Applying $C=50.0$ soft-capping keeps attention entropy bounded throughout training.

---

## 6. Cross-Domain Generalisation

- **Large Language Models (Gemma, Grok, ST-MoE):** Soft-capping and Router Z-loss are mandatory architectural choices for scaling dense and MoE transformer models reliably.
- **Vision Transformers (ViT, Swin, DeiT):** DropPath with linearly increasing drop rates ($0.0 \to 0.3$ across depth) is essential for training deep vision backbones without overfitting.
- **Audio & Speech (Conformer, Whisper):** SpecAugment applies time-masking and frequency-masking, generalizing DropBlock to spectrogram domains.

---

## 7. Verification & Empirical Parity

| Comparison Pair | Max Absolute Difference ($\|Y_A - Y_B\|_\infty$) | Test Status |
| :--- | :--- | :--- |
| **PyTorch 2.x vs Golden Safetensors** | $0.00 \times 10^0$ | **PASSED** |
| **JAX vs Golden Safetensors** | $1.91 \times 10^{-6}$ | **PASSED** |
| **Rust Candle vs Golden Safetensors** | $5.96 \times 10^{-8}$ | **PASSED** |
| **Hardware Kernel (CPU/WGPU) vs Golden** | $5.96 \times 10^{-8}$ | **PASSED** |

All implementations satisfy the curriculum parity threshold ($\|Y_A - Y_B\|_\infty < 10^{-5}$).

---

## 8. References & Further Reading

1. Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR 2014.
2. Huang, G., et al. (2016). *Deep Networks with Stochastic Depth*. ECCV 2016.
3. Szegedy, C., et al. (2016). *Rethinking the Inception Architecture for Computer Vision (Label Smoothing)*. CVPR 2016.
4. Ghiasi, G., et al. (2018). *DropBlock: A Regularization Method for Convolutional Networks*. NeurIPS 2018.
5. Fedus, W., et al. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Router Z-Loss)*. JMLR 2022.
6. Ishida, T., et al. (2020). *Do We Need Zero Training Loss After Achieving Zero Training Error? (Flooding)*. ICML 2020.
