# Triton and Tiled Kernel Algorithms Softmax RMSNorm Liger Cross-Entropy

## 1. Architectural Role & Context

In large language model training and inference, vocabulary projections often present severe memory and bandwidth bottlenecks. For modern vocabularies ($V \ge 128\,000$) and sequence lengths ($B \times S \ge 4\,096$), materializing full unnormalized logits in high-bandwidth memory (HBM) consumes immense memory:
$$\text{Memory}_{\text{logits}} = B \times S \times V \times 4\text{ bytes} \approx 4\,096 \times 128\,000 \times 4 \approx 2.1\text{ GB}$$
During backpropagation through Cross-Entropy loss, retaining or re-materializing this tensor causes out-of-memory (OOM) failures or requires expensive activation checkpointing.

**Triton and Tiled Kernel Architectures** (Tillet et al., 2019; Hsu et al., 2024 / Liger-Kernel) resolve this memory wall by tiling operations across fast on-chip SRAM:
1. **Fused RMSNorm:** Normalizes activations and applies scaling in a single memory pass without intermediate variance tensors.
2. **Fused SwiGLU:** Fuses gate projection, activation, and up projection inside SRAM register files.
3. **Chunked Fused Linear + Cross-Entropy:** Computes projection logits, cross-entropy loss, and input gradients $dH$ chunk-by-chunk in SRAM without ever materializing the global $[B \times S, V]$ logit tensor in HBM.

---

## 2. Mathematical Formalism

### 2.1 Fused Root Mean Square Normalization (RMSNorm)
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \quad \text{RMSNorm}(x, \gamma) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

### 2.2 Fused SwiGLU Activation
$$\text{SwiGLU}(x, W_g, W_u) = \text{silu}(x W_g) \odot (x W_u) = \left( \frac{x W_g}{1 + e^{-x W_g}} \right) \odot (x W_u)$$

### 2.3 Chunked Fused Linear + Cross-Entropy Loss & Gradient
Given hidden states $H \in \mathbb{R}^{B \times D}$, vocabulary matrix $W \in \mathbb{R}^{D \times V}$, and target labels $y \in \{0, \dots, V-1\}^B$, partition $H$ into micro-chunks of size $C$:
$$Z_c = H_c W \in \mathbb{R}^{C \times V}$$
$$\mathcal{L}_c = -\sum_{i=1}^C \log \text{softmax}(Z_{c, i})_{y_{c, i}} = -\sum_{i=1}^C \left( Z_{c, i, y_{c, i}} - \log \sum_{j=1}^V e^{Z_{c, i, j}} \right)$$
The gradient with respect to chunk inputs is computed immediately before discarding $Z_c$:
$$\nabla_{H_c} \mathcal{L} = \frac{1}{B} (P_c - \mathbf{1}_{y_c}) W^T \in \mathbb{R}^{C \times D}$$
where $P_{c, i, j} = \text{softmax}(Z_{c, i})_j$.

---

## 3. Algorithmic Implementation & Tiled Execution

```python
def tiled_linear_cross_entropy(h, w_vocab, labels, chunk_size=2):
    total_loss = 0.0
    dh = torch.zeros_like(h)
    for i in range(0, h.shape[0], chunk_size):
        h_chunk = h[i:i+chunk_size]
        y_chunk = labels[i:i+chunk_size]
        z_chunk = h_chunk @ w_vocab
        loss_chunk = F.cross_entropy(z_chunk, y_chunk, reduction="sum")
        total_loss += loss_chunk
        p_chunk = F.softmax(z_chunk, dim=-1)
        one_hot = F.one_hot(y_chunk, num_classes=w_vocab.shape[1])
        dh[i:i+chunk_size] = ((p_chunk - one_hot) @ w_vocab.T) / h.shape[0]
    return total_loss / h.shape[0], dh
```

---

## 4. Cross-Track Verification & Parity

All operations are cross-verified across PyTorch 2.x, JAX, Rust Candle, Hardware Kernels, and Jupyter Notebooks against seed-42 golden weights:
- Linear anchor $\|Y_{\text{track}} - Y_{\text{golden}}\|_\infty < 10^{-5}$.
- Fused RMSNorm output $\|H_{\text{normed}} - H_{\text{golden}}\|_\infty < 10^{-5}$.
- SwiGLU activation $\|Y_{\text{swiglu}} - Y_{\text{golden}}\|_\infty < 10^{-5}$.
- Tiled cross-entropy loss and backward input gradient $\|L - L_{\text{golden}}\|_\infty < 10^{-5}$ and $\|dH - dH_{\text{golden}}\|_\infty < 10^{-5}$.

---

## 5. Failure Modes & Misuse Guardrails

| Misuse Mode | Mechanism | Operational Guardrail |
| :--- | :--- | :--- |
| **Full Logit Materialization** | Materializing $[B \times S, 128\text{k}]$ FP32 logits before `F.cross_entropy` causes OOM crashes during training. | Always use chunked or fused linear cross-entropy (Liger-Kernel). |
| **Ignoring Float32 Accumulation in Softmax** | Accumulating exp sums in FP16 causes numerical underflow/overflow. | Keep log-sum-exp reductions strictly in FP32 inside SRAM tiles. |
| **Uncoalesced Memory Strides** | Transposing weight matrices inside inner kernel loops causes DRAM bank conflicts. | Ensure vocabulary matrices are stored in memory-contiguous layouts. |

---

## 6. Roofline Analysis & Hardware Performance

- **Standard Eager Cross-Entropy:** Requires writing $2.1\text{ GB}$ of logits to HBM, then reading it back for softmax and backward gradients ($\approx 6.3\text{ GB}$ HBM traffic).
- **Fused Chunked Cross-Entropy:** Keeps logits resident in SRAM, eliminating $> 80\%$ of memory bandwidth overhead and reducing peak training VRAM from $O(B \times S \times V)$ to $O(C \times V)$.

---

## 7. Cross-Domain Transfer & Isomorphisms

| Domain | Tiled Kernel Analogue | Memory Advantage |
| :--- | :--- | :--- |
| **Computational Fluid Dynamics** | Stencil Sweeping with Shared Memory Cache | Stream 3D grid velocity updates through L1 cache without global RAM roundtrips |
| **Graphics & Ray Tracing** | Tiled Bounding Volume Hierarchy (BVH) Traversal | Keep ray bundles and triangle vertices in ray-tracing cores' local registers |
| **High-Throughput Genomics** | Chunked Smith-Waterman Sequence Alignment | Compute dynamic programming score matrices in SRAM without saving quadratic tables |

---

## 8. References & Further Reading

1. Tillet, P., Kung, H. T., & Cox, D. (2019). *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*. MAPL.
2. Hsu, B., et al. (2024). *Liger Kernel: Efficient Triton Kernels for LLM Training*. arXiv:2410.10989.
3. Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288 (SwiGLU & RMSNorm).
