# Glossary and Mathematical Notation Reference

This document establishes the standard mathematical notation, precision formats, bit-level numeric layouts, and asymptotic complexity bounds used uniformly across all 263 topics.

---

## 1. Unified Mathematical Notation

### Linear Algebra & Calculus
- $x, y, z \in \mathbb{R}$: Scalars (lowercase roman).
- $\mathbf{x}, \mathbf{y}, \mathbf{z} \in \mathbb{R}^D$: Vectors (bold lowercase). Column vectors by default.
- $\mathbf{X}, \mathbf{Y}, \mathbf{W} \in \mathbb{R}^{N \times D}$: Matrices (bold uppercase).
- $\mathbf{\mathcal{T}} \in \mathbb{R}^{B \times T \times D}$: Higher-order tensors (calligraphic bold).
- $\mathbf{X}^\top$: Matrix transpose; $\mathbf{X}^\dagger$: Moore-Penrose pseudoinverse.
- $\operatorname{Tr}(\mathbf{X})$: Matrix trace $\sum_{i} X_{ii}$.
- $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$: Euclidean ($L_2$) vector norm.
- $\|\mathbf{x}\|_1 = \sum_i |x_i|$: Taxicab ($L_1$) vector norm.
- $\|\mathbf{X}\|_F = \sqrt{\operatorname{Tr}(\mathbf{X}^\top \mathbf{X})}$: Frobenius matrix norm.
- $\|\mathbf{X}\|_2 = \sigma_{\max}(\mathbf{X})$: Spectral norm (largest singular value).
- $\kappa(\mathbf{X}) = \frac{\sigma_{\max}(\mathbf{X})}{\sigma_{\min}(\mathbf{X})}$: Condition number under matrix inversion.
- $\nabla_\theta \mathcal{L} \in \mathbb{R}^{|\theta|}$: Gradient vector of scalar loss $\mathcal{L}$ with respect to parameters $\theta$.
- $\mathbf{J}_f(\mathbf{x}) \in \mathbb{R}^{M \times N}$: Jacobian matrix of vector-valued function $f: \mathbb{R}^N \to \mathbb{R}^M$, where $J_{ij} = \frac{\partial f_i}{\partial x_j}$.
- $\mathbf{H}_\mathcal{L}(\theta) \in \mathbb{R}^{P \times P}$: Hessian matrix of second partial derivatives, where $H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}$.

### Probability, Information Theory & Geometry
- $p(x), q(x)$: Probability density functions.
- $\mathbb{E}_{x \sim p}[f(x)]$: Expected value of $f(x)$ under distribution $p$.
- $\operatorname{Var}(x) = \mathbb{E}[(x - \mathbb{E}[x])^2]$: Variance.
- $D_{KL}(p \parallel q) = \int p(x) \ln \frac{p(x)}{q(x)} \, dx$: Kullback-Leibler divergence (relative entropy).
- $D_{JS}(p \parallel q) = \frac{1}{2} D_{KL}(p \parallel m) + \frac{1}{2} D_{KL}(q \parallel m)$ where $m = \frac{1}{2}(p + q)$: Jensen-Shannon divergence.
- $\mathcal{W}_2(p, q) = \left( \inf_{\gamma \in \Pi(p, q)} \int \|x - y\|_2^2 \, d\gamma(x, y) \right)^{1/2}$: 2-Wasserstein optimal transport distance.
- $\mathbf{F}_\theta = \mathbb{E}_{x \sim p_\theta}[\nabla_\theta \ln p_\theta(x) \nabla_\theta \ln p_\theta(x)^\top]$: Fisher Information Matrix.
- $G$: Lie group; $\mathfrak{g}$: Lie algebra.
- $SE(3)$: Special Euclidean group (3D translations and rotations).
- $SO(3)$: Special Orthogonal group (3D rotations, $\mathbf{R}^\top \mathbf{R} = \mathbf{I}, \det(\mathbf{R}) = 1$).
- $S_n$: Symmetric permutation group of $n$ elements.

---

## 2. The 27-Format Precision & Numeric Bit Layout Lattice

| Format Name | Total Bits | Sign Bits | Exponent Bits | Mantissa Bits | Exponent Bias | Numerical Dynamic Range | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP64 (IEEE 754)** | 64 | 1 | 11 | 52 | 1023 | $\approx [10^{-308}, 10^{308}]$ | Scientific simulation, ground-truth reference |
| **FP32 (IEEE 754)** | 32 | 1 | 8 | 23 | 127 | $\approx [1.18 \times 10^{-38}, 3.4 \times 10^{38}]$ | Standard master weights & numerical baseline |
| **FP16 (IEEE 754)** | 16 | 1 | 5 | 10 | 15 | $\approx [6.1 \times 10^{-5}, 65504]$ | Legacy half-precision training & inference |
| **BF16 (BFloat16)** | 16 | 1 | 8 | 7 | 127 | Same range as FP32, lower precision | Standard modern deep learning training |
| **FP8 (E4M3FN)** | 8 | 1 | 4 | 3 | 7 | $[-448, 448]$ (No infinities, 1 NaN) | Forward activation & weight storage |
| **FP8 (E4M3FNUZ)**| 8 | 1 | 4 | 3 | 8 | $[-240, 240]$ (Unsigned zero, NaN at -128) | AMD / PyTorch-specific quantization |
| **FP8 (E5M2)** | 8 | 1 | 5 | 2 | 15 | $[-57344, 57344]$ (IEEE style with Inf) | Backward gradient scaling |
| **FP8 (E5M2FNUZ)**| 8 | 1 | 5 | 2 | 16 | $[-57344, 57344]$ (No Inf, unsigned zero) | Micro-scaling alternate standard |
| **MXFP8** | 8 | 1 | 4 | 3 | 7 | Shared 8-bit scale per 32 elements | OCP Microscaling standard |
| **MXFP6 (E3M2)** | 6 | 1 | 3 | 2 | 3 | $[-28, 28]$ with shared scale | Microscaling compression |
| **MXFP6 (E2M3)** | 6 | 1 | 2 | 3 | 1 | $[-7.5, 7.5]$ with shared scale | Microscaling high-precision mantissa |
| **MXFP4 (E2M1)** | 4 | 1 | 2 | 1 | 1 | $[-6, 6]$ with shared scale | OCP 4-bit standard |
| **FP4 (E2M1)** | 4 | 1 | 2 | 1 | 1 | $[-6, 6]$ | Extreme floating point weight quantization |
| **INT32** | 32 | 1 (two's) | 0 | 31 | N/A | $[-2^{31}, 2^{31}-1]$ | Indexing, integer counters |
| **INT16** | 16 | 1 (two's) | 0 | 15 | N/A | $[-32768, 32767]$ | Audio signals, DSP hardware |
| **INT8 (Symmetric)**| 8 | 1 (two's) | 0 | 7 | N/A | $[-127, 127]$ (zero-point = 0) | Enterprise CPU/GPU linear layer inference |
| **INT8 (Asymmetric)**| 8 | 1 (two's) | 0 | 7 | N/A | $[-128, 127]$ with non-zero offset | Asymmetric activations with ReLU |
| **UINT8** | 8 | 0 | 0 | 8 | N/A | $[0, 255]$ | Raw image pixels & unsigned embeddings |
| **INT4 (Symmetric)**| 4 | 1 (two's) | 0 | 3 | N/A | $[-7, 7]$ | GPTQ / AWQ edge weight serving |
| **INT4 (Asymmetric)**| 4 | 1 (two's) | 0 | 3 | N/A | $[-8, 7]$ with zero point | Packed nibble inference |
| **INT2** | 2 | 1 (two's) | 0 | 1 | N/A | $\{-2, -1, 0, 1\}$ | Sub-ternary extreme compression |
| **Ternary (BitNet 1.58)** | $\log_2(3) \approx 1.58$ | Signed | 0 | Trinary | N/A | $\{-1, 0, +1\}$ with per-tensor scale | Multiplication-free additions on CPU |
| **Binary (1-bit / XNOR)** | 1 | Sign only | 0 | 0 | N/A | $\{-1, +1\}$ | Hardware XNOR and popcount |
| **Posit(8, 0)** | 8 | 1 | 0 | Dynamic | Golden | Self-adaptive dynamic range | Non-IEEE edge arithmetic |
| **Posit(8, 1)** | 8 | 1 | 1 | Dynamic | Golden | Broad dynamic range edge computing | Non-IEEE edge arithmetic |
| **Posit(16, 1)** | 16 | 1 | 1 | Dynamic | Golden | High-accuracy scientific arithmetic | Non-IEEE scientific acceleration |
| **Logarithmic (LNS8)** | 8 | 1 | Log2 val | 0 | N/A | Multiplication becomes addition | DSP and neuromorphic processors |

---

## 3. Asymptotic Time and Memory Complexity Summary

| Architecture / Technique | Training Time | Inference Time (Prefill) | Inference Time (Decode 1 Token) | Memory Complexity (State / KV Cache) |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Multi-Head Attention** | $\mathcal{O}(B \cdot T^2 \cdot D)$ | $\mathcal{O}(B \cdot T^2 \cdot D)$ | $\mathcal{O}(B \cdot T \cdot D)$ | $\mathcal{O}(B \cdot T \cdot D)$ (Uncompressed KV Cache) |
| **FlashAttention-2** | $\mathcal{O}(B \cdot T^2 \cdot D)$ | $\mathcal{O}(B \cdot T^2 \cdot D)$ | $\mathcal{O}(B \cdot T \cdot D)$ | $\mathcal{O}(B \cdot \frac{M_{\text{SRAM}}}{D})$ (Zero $T^2$ memory in HBM) |
| **Sliding Window Attention ($W$)** | $\mathcal{O}(B \cdot T \cdot W \cdot D)$ | $\mathcal{O}(B \cdot T \cdot W \cdot D)$ | $\mathcal{O}(B \cdot W \cdot D)$ | $\mathcal{O}(B \cdot W \cdot D)$ (Fixed buffer) |
| **Linear Attention** | $\mathcal{O}(B \cdot T \cdot D^2)$ | $\mathcal{O}(B \cdot T \cdot D^2)$ | $\mathcal{O}(B \cdot D^2)$ | $\mathcal{O}(B \cdot D^2)$ (Constant state) |
| **Mamba / S6 Selective SSM** | $\mathcal{O}(B \cdot T \cdot D \cdot N)$ | $\mathcal{O}(B \cdot T \cdot D \cdot N)$ | $\mathcal{O}(B \cdot D \cdot N)$ | $\mathcal{O}(B \cdot D \cdot N)$ (Constant state) |
| **RWKV / RetNet** | $\mathcal{O}(B \cdot T \cdot D)$ | $\mathcal{O}(B \cdot T \cdot D)$ | $\mathcal{O}(B \cdot D)$ | $\mathcal{O}(B \cdot D)$ (Constant state) |
| **Graph Convolutional Network (GCN)**| $\mathcal{O}(\|E\| \cdot D + \|V\| \cdot D^2)$ | $\mathcal{O}(\|E\| \cdot D + \|V\| \cdot D^2)$ | $\mathcal{O}(\|E\| \cdot D)$ | $\mathcal{O}(\|V\| \cdot D + \|E\|)$ |
| **Continuous Diffusion SDE** | $\mathcal{O}(K \cdot T_{\text{steps}} \cdot \text{Cost}(f_\theta))$ | N/A | $\mathcal{O}(N_{\text{steps}} \cdot \text{Cost}(f_\theta))$ | $\mathcal{O}(\text{Activations of } f_\theta)$ |
| **Flow Matching ODE (PCEC)** | $\mathcal{O}(K \cdot \text{Cost}(f_\theta))$ | N/A | $\mathcal{O}(N_{\text{eval}} \cdot \text{Cost}(f_\theta))$ ($N_{\text{eval}} \le 10$) | $\mathcal{O}(\text{Activations of } f_\theta)$ |
| **Physics-Informed Neural Network (PINN)**| $\mathcal{O}(N_{\text{colloc}} \cdot \text{Cost}(\text{autograd}^2))$ | N/A | $\mathcal{O}(N_{\text{eval}} \cdot \text{Cost}(f_\theta))$ | $\mathcal{O}(N_{\text{colloc}} \cdot D)$ |
