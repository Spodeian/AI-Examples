# Positional and Structural Encodings: Sinusoidal, RoPE, ALiBi, and Graph Random Walk Encodings (C03 Topic 04)

## 1. Executive Summary
Transformers and permutation-invariant neural architectures operate intrinsically over sets: self-attention possesses permutation equivariance, $\text{Attn}(\Pi X) = \Pi \text{Attn}(X)$ for any permutation matrix $\Pi$. To process ordered sequences, multidimensional grids, and topological graphs, structural inductive biases must be injected into the representation space.

Historically, coordinate injection evolved through four generations:
1. **Absolute Additive Encodings (Vaswani Sinusoidal & Learned 1D):** Introduced in the original Transformer (*Vaswani et al. 2017*), summing harmonic trigonometric basis functions directly into input token embeddings.
2. **Relative Positional Representations:** Recognizing that token interaction depends on relative distance $i - j$ rather than absolute timestamps $i$ and $j$ (*Shaw et al. 2018*), shifting bias directly into the attention logit matrix.
3. **Rotary Positional Embedding (RoPE):** Multiplicatively rotating query and key vector pairs in 2D orthogonal subspaces (*Su et al. 2021*), unifying absolute coordinate multiplication with pure relative inner product decay $\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = g(q, k, m - n)$.
4. **Attention with Linear Biases (ALiBi):** Eliminating embedding tables entirely by penalizing query-key attention logits proportionally to distance $|i - j|$ with static geometric head slopes (*Press et al. 2021*), achieving train-short, test-long length extrapolation.
5. **Non-Euclidean & Graph Structural Encodings (RWPE):** In topological domains where 1D order is absent, **Random Walk Positional Encodings (RWPE)** (*Dwivedi et al. 2022*) capture $k$-step landing probabilities $P^k_{ii} = [D^{-1}A]^k_{ii}$, overcoming the $\pm v$ sign ambiguity of Laplacian eigenvectors.

This document details the first-principles mathematics, WGSL/CPU kernel implementations, and transitive numerical parity across PyTorch 2.x, JAX, Rust Candle, and Hardware Kernels.

---

## 2. Mathematical Foundations

### 2.1 Vaswani Sinusoidal Absolute Positional Encodings
For sequence position $pos \in [0, S-1]$ and channel dimension $2i, 2i+1 \in [0, D-1]$:
$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/D}}\right), \quad \text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/D}}\right)$$
The inner product between positions $pos$ and $pos + k$ depends strictly on offset $k$ via trigonometric angle-addition identities:
$$\sin(\omega(pos + k)) = \sin(\omega pos)\cos(\omega k) + \cos(\omega pos)\sin(\omega k)$$
However, because $\text{PE}$ is additively combined with $x$ ($x_{\text{input}} = x_{\text{token}} + \text{PE}$), deep layers attenuate positional signals through non-linear residual accumulation.

### 2.2 Rotary Positional Embedding (RoPE)
RoPE decomposes $D$-dimensional query and key vectors into $D/2$ orthogonal 2D subspaces. For coordinate $m$ and 2D pair $(x_{2i}, x_{2i+1})$:
$$R_{\theta_i, m} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$
where $\theta_i = b^{-2i/D}$ with base $b = 10000$.
In complex form, letting $z = x_{2i} + j x_{2i+1}$:
$$R_{\theta_i, m} z = z \cdot e^{j m \theta_i}$$
The inner product between query at $m$ and key at $n$ satisfies:
$$\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = \text{Re}\left[ \sum_{i=0}^{D/2-1} (q_{2i} + j q_{2i+1})(k_{2i} - j k_{2i+1}) e^{j(m-n)\theta_i} \right]$$
which depends strictly on the relative displacement $m - n$, preserving self-attention norm and decaying naturally with distance.

### 2.3 Attention with Linear Biases (ALiBi)
ALiBi replaces positional embeddings with a static linear penalty subtracted directly from pre-softmax attention logits:
$$A_{i, j} = \frac{q_i k_j^T}{\sqrt{d}} - m_h \cdot |i - j|$$
For a multi-head architecture with $H$ heads, slopes $m_h$ form a fixed geometric sequence:
$$m_h = 2^{-\frac{8 h}{H}}, \quad h \in \{1, 2, \dots, H\}$$
Because no position indices are embedded into learned vectors, ALiBi generalizes zero-shot to sequences $4\times$ to $10\times$ longer than the training context window.

### 2.4 Random Walk Positional Encodings (RWPE) on Graphs
On an irregular graph $G = (V, E)$ with adjacency matrix $A$ and degree matrix $D$, the random walk transition matrix is:
$$P = D^{-1} A$$
The $k$-step landing probability for node $i$ returning to itself is:
$$p_{ii}^{(k)} = [P^k]_{ii}$$
The $K$-dimensional structural descriptor for node $i$ is:
$$\text{RWPE}(i) = \left[ p_{ii}^{(1)}, p_{ii}^{(2)}, \dots, p_{ii}^{(K)} \right] \in [0, 1]^K$$
Unlike Laplacian Eigenvector Positional Encodings (LapPE), which suffer from arbitrary sign flips ($\pm v_k$) and basis rotation ambiguities for repeated eigenvalues, RWPE is uniquely defined and permutation-invariant.

---

## 3. Structural Properties & Extrapolation Spectrum

```text
Encoding Scheme     Location        Parameter Cost    Relative-Aware    Extrapolation
────────────────────────────────────────────────────────────────────────────────────────
Sinusoidal (1D)     Input (Add)     0                 Implicit          Poor
RoPE (Rotary)       Q, K (Mul)      0                 Exact Hermite     Moderate (NTK / YaRN)
ALiBi (Linear)      Logits (Add)    0                 Exact Linear      Superior (Zero-shot)
RWPE (Graph)        Node (Concat)   0                 Topological       Topology-Invariant
```

---

## 4. Multi-Ecosystem Implementation Matrix

| Component | PyTorch 2.x (`pytorch2/`) | JAX / Equinox (`jax/`) | Candle Rust (`candle/`) | Hardware Kernels (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Sinusoidal PE** | `SinusoidalPositionalEncoding` | `sinusoidal_pe_forward` | `sinusoidal_pe_forward` | `sinusoidal_pe_cpu` |
| **RoPE (Rotary)** | `RotaryEmbedding` | `rope_forward` | `rope_forward` | `ROPE_SHADER` (WGSL) + `rope_cpu` |
| **ALiBi** | `ALiBi` | `alibi_forward` | `alibi_forward` | `alibi_cpu` |
| **Graph RWPE** | `RandomWalkPositionalEncoding`| `rwpe_forward` | `rwpe_forward` | `rwpe_cpu` |
| **Linear Anchor** | `nn.Linear` | `linear_forward` | `LinearAnchor` | `linear_anchor_forward_cpu` |
| **Parity Tolerance** | $< 10^{-5}$ | $< 10^{-5}$ | $< 10^{-5}$ | $< 10^{-5}$ |

---

## 5. Pathological Failure Modes & Postmortems

1. **High-Frequency Phase Wrapping in Long-Context RoPE:** When extrapolating past the maximum training length $L_{\text{train}}$, token coordinates $m > L_{\text{train}}$ rotate high-frequency dimensions past $2\pi$, causing attention scores to degrade into out-of-distribution noise. Frequency scaling methods (NTK-aware interpolation, YaRN) dynamically rescale $\theta_i$ to prevent phase collision.
2. **ALiBi Monotonic Distance Assumption:** ALiBi penalizes all distant tokens linearly. In tasks requiring long-range periodic or copy-memory lookups, strong slope heads completely suppress distant tokens regardless of content relevance.
3. **Graph Disconnected Components in RWPE:** If a node has degree zero ($D_{ii} = 0$), $D^{-1}$ produces numerical division by zero. Normalization must clamp degrees to $\max(D_{ii}, 1)$ and set $p_{ii}^{(k)} = 0$ for isolated components.

---

## 6. Cross-Domain Generalisation

- **Large Language Models (LLaMA 3, Mistral, Gemma):** RoPE is the universal standard for modern open-weights autoregressive language models, paired with base scaling ($b = 500,000$) for long-context windows.
- **Vision Transformers (2D RoPE, Axial Positional Encodings):** Splitting head dimension into $D/4$ for vertical coordinate $y$ and $D/4$ for horizontal coordinate $x$ allows RoPE to operate seamlessly over arbitrary resolution 2D image patches.
- **Molecular & Material Graph Transformers (GPS, GraphGPS):** RWPE provides local cycle count signatures (e.g. 3-step triangles, 5-step rings, 6-step aromatics) directly distinguishing molecular functional groups.

---

## 7. Verification & Empirical Parity

| Comparison Pair | Max Absolute Difference ($\|Y_A - Y_B\|_\infty$) | Test Status |
| :--- | :--- | :--- |
| **PyTorch 2.x vs Golden Safetensors** | $0.00 \times 10^0$ | **PASSED** |
| **JAX vs Golden Safetensors** | $5.96 \times 10^{-8}$ | **PASSED** |
| **Rust Candle vs Golden Safetensors** | $5.96 \times 10^{-8}$ | **PASSED** |
| **Hardware Kernel (CPU/WGPU) vs Golden** | $5.96 \times 10^{-8}$ | **PASSED** |

All implementations satisfy the curriculum parity threshold ($\|Y_A - Y_B\|_\infty < 10^{-5}$).

---

## 8. References & Further Reading

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
2. Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.
3. Press, O., Smith, N. A., & Lewis, M. (2021). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. ICLR 2022.
4. Dwivedi, V. P., et al. (2022). *Graph Neural Networks with Learnable Structural and Positional Representations*. ICLR 2022.
5. Tancik, M., et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS 2020.
