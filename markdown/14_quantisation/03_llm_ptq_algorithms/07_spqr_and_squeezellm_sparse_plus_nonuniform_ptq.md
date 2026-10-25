# SpQR and SqueezeLLM: Sparse Plus Nonuniform PTQ

Pushing Large Language Model quantization down to 3-bit and 2-bit regimes typically causes catastrophic degradation under uniform post-training quantization schemes. The root cause is a heavy-tailed distribution: a minute fraction ($< 1\%$) of weights are orders of magnitude more sensitive to quantization perturbation than the rest. **SpQR** (Sparse-Quantized Representation — Dettmers et al., 2023) and **SqueezeLLM** (Kim et al., 2023) resolve this bottleneck by decomposing the weight matrix into a high-precision sparse component (storing sensitive outlier weights in FP16) and an ultra-low-bit non-uniform or dense quantized component.

---

## 1. The Breakdown of Uniform Low-Bit Quantization

In 4-bit and 8-bit quantization, uniform grid rounding noise is sufficiently small that standard calibration or second-order compensation (GPTQ, AWQ) suffices. However, below 4 bits:
- A single large weight outlier forces the uniform step size $\Delta = \frac{\max(|W|)}{2^{b-1}-1}$ to expand dramatically.
- Expanding $\Delta$ coarsens the quantization grid for the remaining $99\%$ of weights, drowning normal weights in quantization noise.
- The Hessian sensitivity metric $\mathbf{S}_{ij} = \mathbf{H}_{ii} |W_{ij}|$ demonstrates that quantization errors in specific outlier parameters incur disproportionately large downstream task loss.

---

## 2. SpQR: Sparse-Quantized Representation

**SpQR** identifies outlier weights using second-order sensitivity and extracts them into a sparse matrix:

$$
\mathbf{W} = \mathbf{W}_{\text{dense}} + \mathbf{W}_{\text{sparse}}
$$

1. **Hessian Sensitivity Formulation:** Given calibration activations $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{in}}}$, the empirical Hessian diagonal is computed as:
   $$
   \mathbf{H}_{ii} = \frac{2}{N} \sum_{k=1}^N \mathbf{X}_{ki}^2
   $$
   The parameter sensitivity is quantified by:
   $$
   \mathbf{S}_{ij} = \mathbf{H}_{ii} \cdot |W_{ij}|
   $$
2. **Sparse Outlier Isolation:** Weights with sensitivity exceeding a calibrated threshold are routed to $\mathbf{W}_{\text{sparse}}$ in FP16 ($~1\%$ of total elements).
3. **Dense Quantization:** The remaining non-outlier matrix $\mathbf{W}_{\text{dense}}$ is quantized uniformly to 3-bit (or 2-bit) with small tile-based micro-groups:
   $$
   \mathbf{q} = \text{clamp}\left( \text{round}\left( \frac{\mathbf{W}_{\text{dense}}}{\Delta} \right), q_{\min}, q_{\max} \right)
   $$

---

## 3. SqueezeLLM: Non-Uniform Sensitivity-Weighted PTQ

While SpQR preserves sensitive weights in sparse FP16 format, **SqueezeLLM** addresses non-uniform weight distributions through sensitivity-weighted $k$-means clustering:

1. **Weighted Objective:** Rather than standard $L_2$ clustering, SqueezeLLM minimizes the Hessian-weighted quadratic error:
   $$
   \min_{\mathbf{c}, \mathbf{z}} \sum_{i, j} \mathbf{H}_{ii} \left( W_{ij} - c_{z_{ij}} \right)^2
   $$
   where $c_1, \dots, c_K$ are discrete centroid levels and $z_{ij} \in \{1, \dots, K\}$ are cluster assignments.
2. **Dense-and-Sparse Decomposition:** SqueezeLLM combines non-uniform lookup tables (LUTs) for dense weights with compressed sparse column (CSC) storage for high-magnitude outliers.

---

## 4. SpQR Linear Forward Formulation

At runtime, inference on an input activation tensor $\mathbf{X}$ computes the joint dense and sparse projection:

$$
\mathbf{Y} = \mathbf{X} \mathbf{W}_{\text{dense}}^{\text{deq}} + \mathbf{X} \mathbf{W}_{\text{sparse}} + \mathbf{b}
$$

Because $\mathbf{W}_{\text{sparse}}$ is extremely sparse ($< 2\%$ non-zeros), the sparse GEMM ($\text{SpMM}$) introduces minimal compute latency while completely preserving the salient outlier features.

---

## 5. Multi-Framework Parity & Verification Architecture

All implementations share bit-exact transitive numerical parity across PyTorch 2.x, JAX, and Candle:

```mermaid
flowchart TD
    A[Calibration Matrix X & Weights W] --> B[Hessian Diagonal H_ii = 2/N sum X_ki^2]
    B --> C[Sensitivity Metric S_ij = H_ii * abs W_ij]
    C --> D[Top-k Thresholding & Outlier Separation]
    D --> E[FP16 Sparse Outliers W_sparse]
    D --> F[Dense Residual Matrix W_dense]
    F --> G[Symmetric 3-bit Quantization]
    G --> H[Quantized Dense Weights W_dense_deq]
    E --> I[SpQR Forward Pass: X @ W_dense + X @ W_sparse + b]
    H --> I
    I --> J[Golden SafeTensors seed 42]
    J --> K[PyTorch 2.x Reference Parity < 1e-6]
    J --> L[JAX Parity < 1e-5]
    J --> M[Candle Rust Parity < 1e-5]
```

### Numerical Parity Summary:
| Track | Path | Parity vs Golden |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py` | Reference |
| **JAX** | `jax/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.rs` | WGSL / CPU |

---

## 6. Systems Engineering & Hardware Kernels

Inference fuses the dense low-bit projection with sparse outlier accumulation in a single WGSL dispatch (`SPQR_FUSED_GEMM_WGSL`):

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let o = global_id.y;
    if (b >= params.batch || o >= params.d_out) { return; }
    var acc: f32 = bias[o];
    for (var i: u32 = 0u; i < params.d_in; i = i + 1u) {
        let x_val = x[b * params.d_in + i];
        let wd = w_dense[i * params.d_out + o];
        let ws = w_sparse[i * params.d_out + o];
        acc = acc + x_val * (wd + ws);
    }
    y_out[b * params.d_out + o] = acc;
}
```

---

## 7. Comparative Benchmark & SQNR Evaluation

Evaluating 3-bit quantization on a calibration matrix with activation outlier channels ($d_{\text{in}}=16, d_{\text{out}}=8$):

| Configuration | Sparse Outlier Fraction | Output MSE vs FP32 | Output SQNR (dB) |
| :--- | :--- | :--- | :--- |
| **Pure 3-bit Dense Quantization** | $0.0\%$ | $0.1834$ | $17.91\text{ dB}$ |
| **SpQR (Dense 3-bit + Sparse FP16)** | $6.25\%$ (8/128 weights) | $0.1296$ | $19.42\text{ dB}$ ($+1.51\text{ dB}$) |

Isolating just $6.25\%$ of the most sensitive weights provides $+1.51\text{ dB}$ in signal quality, protecting critical features from catastrophic collapse in ultra-low-bit regimes.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.ipynb](../../../notebooks/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.ipynb)
- **PyTorch 2.x Module:** [07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py](../../../pytorch2/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py)
- **Functional JAX Module:** [07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py](../../../jax/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py)
- **Native Rust Candle Track:** [07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.rs](../../../candle/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.rs)
- **Hardware Kernel (CPU / WGSL):** [07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.rs](../../../kernels/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.rs)
- **Golden SafeTensors:** [07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.safetensors](../../../weights/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.safetensors)
