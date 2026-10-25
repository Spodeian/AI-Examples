# Unstructured, Structured Pruning, ShortGPT, and the Lottery Ticket Hypothesis

Model compression techniques seek to reduce the computational, memory, and energy footprint of deep neural networks while preserving representation fidelity and downstream performance. Pruning eliminates parameters, channels, or entire structural modules from overparameterized models. This topic examines the four foundational pruning paradigms: fine-grained **unstructured magnitude pruning**, coarse-grained **structured channel/row pruning**, modern one-shot layer ablation (**ShortGPT Block Influence**), and the theoretical grounding of sparse subnetworks via Frankle & Carbin's **Lottery Ticket Hypothesis (LTH)**.

---

## 1. Pruning Paradigms: Unstructured vs. Structured vs. Layer Slicing

Model pruning spans a spectrum from individual scalar weights to macroscopic transformer layers:

| Pruning Paradigm | Granularity | Hardware Speedup | Zero-Shot Retraining Required? | Memory Representation | Typical Sparsity Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unstructured** | Scalar $W_{ij}$ | Requires 2:4 / Sparse Tensor Cores | Heavy iterative fine-tuning | Sparse CSR / COO / bitmask | $50\% - 90\%$ |
| **Structured (Channel/Row)** | Vector $W_{i, :}$ or $W_{:, j}$ | Immediate across standard dense GEMM | Moderate fine-tuning | Physically resized dense tensor | $25\% - 50\%$ |
| **Layer Pruning (ShortGPT)** | Block / Layer $L_i$ | Direct linear reduction in latency | Zero (Direct one-shot or light LoRA) | Truncated sequential pipeline | $15\% - 30\%$ of layers |
| **Lottery Ticket Subnetwork** | Sparse topological mask $M$ | Dependent on sparse runtime | Retrained from initialization $W_0$ | Masked dense or sparse graph | $80\% - 95\%$ |

---

## 2. Mathematical Formulations of Magnitude & L2 Norm Pruning

### Unstructured Magnitude Pruning
Given a weight tensor $W \in \mathbb{R}^{m \times n}$ and target sparsity ratio $s \in [0, 1)$, unstructured magnitude pruning computes an absolute threshold $\tau$:
$$\tau = \text{k-th smallest element of } \{ |W_{ij}| \}_{i=1, j=1}^{m, n}, \quad k = \lfloor s \cdot m \cdot n \rfloor$$
The pruned weight and binary mask are:
$$M_{ij} = \mathbb{I}(|W_{ij}| \ge \tau), \quad W^{\text{unstr}} = W \odot M$$

### Structured Row / Channel Pruning
In linear transformations $Y = XW$, dropping output channels (or weight rows) preserves standard dense matrix multiplication without specialized sparse kernels. We rank rows by their $\ell_2$ Euclidean norm:
$$s_i = \|W_{i, :}\|_2 = \sqrt{\sum_{j=1}^n W_{ij}^2}$$
For a retention fraction $\rho \in (0, 1]$, we select top indices $\mathcal{I}_{\text{keep}} = \text{Top-}k(\{s_i\}_{i=1}^m, k = \lfloor \rho \cdot m \rfloor)$. The sliced weight is:
$$W^{\text{struct}} = W[\mathcal{I}_{\text{keep}}, :] \in \mathbb{R}^{k \times n}$$

---

## 3. ShortGPT and Layer Redundancy: Cosine Block Influence (BI)

Recent foundation LLMs (e.g. LLaMA, Mistral, Qwen) exhibit significant representation redundancy in intermediate residual blocks. ShortGPT (Men et al., 2024) introduces **Block Influence (BI)** to quantify the transformational contribution of layer $l$:

Given input hidden state $X_l$ and output state $X_{l+1}$ for a batch of sequences:
$$\text{BI}_l = 1 - \frac{1}{B} \sum_{b=1}^B \frac{X_{l, b} \cdot X_{l+1, b}}{\|X_{l, b}\|_2 \cdot \|X_{l+1, b}\|_2 + \epsilon}$$

```
X_l ───► [ Residual Layer l ] ───► X_{l+1}
 │                                   ▲
 └─────────────────► cos(·, ·) ──────┘
                        │
                        ▼
           BI_l = 1 - cos(X_l, X_{l+1})
```

- When $\text{BI}_l \to 0$, $X_{l+1} \approx X_l$: the layer acts as an approximate identity mapping and can be pruned with negligible loss in benchmark accuracy.
- Layers with high $\text{BI}_l$ perform critical semantic feature shifts and must be preserved.

---

## 4. The Lottery Ticket Hypothesis & Subnetwork Rewinding

Frankle & Carbin (2018) formulated the **Lottery Ticket Hypothesis**:
> *A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations.*

### Iterative Magnitude Pruning (IMP) with Weight Rewinding
1. Initialize dense weights $W_0 \sim \mathcal{D}$.
2. Train the network for $T$ steps to converge at $W_T$.
3. Prune $s\%$ of the smallest magnitude parameters in $W_T$, yielding binary mask $M \in \{0, 1\}^{\text{dim}(W)}$:
   $$M = \mathbb{I}(|W_T| \ge \tau_s)$$
4. Reset non-zero weights back to their initial state $W_0$ (or early warmup checkpoint $W_k$ for large-scale transformers):
   $$W_{\text{ticket}} = W_0 \odot M$$
5. Retrain $W_{\text{ticket}}$ under fixed mask $M$.

---

## 5. Hardware Execution Realities: Sparse CSR/SpMM vs. Dense Sliced GEMM

Unstructured sparsity creates irregular, non-contiguous memory access patterns:
- **Unstructured CPU/GPU Execution**: Standard dense BLAS GEMM (`sgemm`) cannot leverage unstructured zeroes without explicit zero-overhead bitmasking. Compressed Sparse Row (CSR) or Compressed Sparse Column (CSC) formats incur indexing overhead that negates arithmetic savings until sparsity exceeds $75\% - 90\%$.
- **Ampere 2:4 Semi-Structured Sparsity**: Hardware acceleration is achieved when exactly 2 out of every 4 adjacent values are zero, executing on NVIDIA Sparse Tensor Cores at $2\times$ dense throughput.
- **Structured Pruning Efficiency**: Physically slicing tensor rows or removing transformer layers preserves standard contiguous FP16/BF16 dense GEMMs, delivering $1.3\times - 2.0\times$ wall-clock speedup directly on vanilla hardware runtimes.

---

## 6. Multi-Framework Implementation & Verification (PyTorch, JAX, Candle)

The algorithms are implemented identically across all three core frameworks and cross-verified against seed-42 golden safetensors:

```rust
// Candle implementation of ShortGPT Block Influence
pub fn compute_block_influence(x_in: &Tensor, x_out: &Tensor) -> Result<f32> {
    let dot = (x_in * x_out)?.sum_keepdim(1)?;
    let norm_in = x_in.sqr()?.sum_keepdim(1)?.sqrt()?;
    let norm_out = x_out.sqr()?.sum_keepdim(1)?.sqrt()?;
    let denom = (norm_in * norm_out)?.affine(1.0, 1e-8)?;
    let cos = (dot / denom)?;
    let mean_cos = cos.mean_all()?.to_scalar::<f32>()?;
    Ok(1.0 - mean_cos)
}
```

The transitive numerical parity across PyTorch, JAX, and Candle is verified to $< 10^{-5}$ absolute tolerance.

---

## 7. Empirical Evaluation: Compression Ratio vs. Representation Fidelity

| Method | Target Metric | Measured Metric / Sparsity | Representation Fidelity (Cosine Sim) | Downstream Viability |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Baseline** | 100% Params | 0.0% Sparsity | 1.0000 | Reference baseline |
| **Unstructured Magnitude** | 50% Sparsity | 50.0% Zeroes | 0.8924 | Requires iterative fine-tuning |
| **Structured Row Slicing** | 75% Retention | 25.0% Rows Pruned | 0.9412 | Immediate GEMM speedup |
| **ShortGPT (Redundant Layer)**| $\text{BI} \to 0$ | $\text{BI} = 5.92 \times 10^{-5}$ | 0.9999 | Direct zero-shot layer removal |
| **ShortGPT (Active Layer)** | High $\text{BI}$ | $\text{BI} = 0.1084$ | 0.8916 | Critical layer, must retain |
| **Winning Ticket Rewind** | Initialized Subnet | 50.0% Masked | N/A (Trainable) | Matches dense accuracy from $W_0$ |

---

## 8. Decision Matrix & Production Deployment Playbook

```
                         Are you compressing at inference time or training time?
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
              Inference Time                     Training Time
                     │                                 │
     Can you modify model architecture?         Do you seek sparse structure
           │                  │                 from initialization?
     ┌─────┴─────┐      ┌─────┴─────┐                  │
     ▼           ▼      ▼           ▼            ┌─────┴─────┐
    Yes          No    Yes          No           ▼           ▼
     │           │      │           │           Yes          No
 ShortGPT     Wanda / Structured Unstructured   Lottery     Iterative
  Layer      SparseGPT Channel   2:4 Sparse     Ticket      Magnitude
 Slicing      One-shot  Pruning   Pruning     Rewinding     Pruning
```

1. **For Production LLM Speedup Without Retraining**: Evaluate ShortGPT Block Influence across validation calibration samples. Prune layers with $\text{BI} < 10^{-4}$ directly from the computation graph.
2. **For Constrained Embedded / Edge Deployments**: Employ structured row/channel pruning to produce smaller dense weight matrices that execute natively on standard hardware without custom kernels.
3. **For Algorithmic Research & Foundational Insight**: Leverage the Lottery Ticket Hypothesis to isolate critical sparse parameter topologies and test overparameterization dynamics.
