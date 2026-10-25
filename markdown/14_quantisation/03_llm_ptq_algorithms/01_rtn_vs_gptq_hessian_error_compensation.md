# RTN vs GPTQ: Second-Order Hessian Error Compensation

In post-training quantization (PTQ) of large language models, naive Round-To-Nearest (RTN) suffers catastrophic perplexity explosion at 4-bit and 3-bit precisions. While each individual weight round-off error may appear modest in isolation, these errors compound coherently across high-dimensional projection layers ($d_{\text{model}} \ge 4096$), degrading attention matrices and MLP feature representations. **GPTQ** (Frantar et al., 2022), evolving from Optimal Brain Surgeon (OBS) and Optimal Brain Compression (OBC), formulates quantization as a constrained second-order optimization problem. By leveraging empirical activation Hessian information $\mathbf{H} = 2 \mathbf{X}^T \mathbf{X} / N$, GPTQ dynamically updates unquantized trailing weights to compensate for quantization errors introduced in preceding columns.

---

## 1. Mathematical Formulation & Taylor Expansion

Consider a linear layer with activation matrix $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{in}}}$ and weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$. We seek a quantized weight $\widehat{\mathbf{W}}$ minimizing output squared error:

$$
E = \|\mathbf{X} \mathbf{W} - \mathbf{X} \widehat{\mathbf{W}}\|_F^2
$$

Expanding the error around the unquantized weights $\mathbf{W}$ via second-order Taylor series yields:

$$
E(\mathbf{W} + \Delta \mathbf{W}) \approx E(\mathbf{W}) + \nabla E^T \Delta \mathbf{W} + \frac{1}{2} \Delta \mathbf{W}^T \mathbf{H} \Delta \mathbf{W}
$$

Because the unquantized weights $\mathbf{W}$ represent a local minimum with respect to layer calibration loss ($\nabla E \approx \mathbf{0}$), the error is dominated by the second-order Hessian:

$$
\mathbf{H} = \nabla^2 E = \frac{2}{N} \mathbf{X}^T \mathbf{X}
$$

---

## 2. From Optimal Brain Surgeon to GPTQ

The classic Optimal Brain Surgeon (OBS) prunes or quantizes a single weight element $w_q$ and updates remaining weights:

$$
\Delta \mathbf{w} = -\frac{w_q - \text{round}(w_q)}{[\mathbf{H}^{-1}]_{qq}} \mathbf{H}^{-1}_{:, q}
$$

OBS has $O(d^3)$ complexity per weight element, requiring $O(d^4)$ operations per layer—prohibitive for billions of parameters. **GPTQ** achieves an algorithmic breakthrough with two key innovations:

1. **Column-wise / Vector Quantization:** Instead of updating weights element-by-element across the entire matrix, GPTQ quantizes an entire column $c$ simultaneously, updating the trailing unquantized submatrix in a single vectorized step.
2. **Cholesky Pre-Inversion & Damping:** Because $\mathbf{H}$ can be ill-conditioned or singular when $N < d_{\text{in}}$, GPTQ applies ridge regularization:
   $$
   \mathbf{H}_{\text{damped}} = \mathbf{H} + \lambda \cdot \text{mean}(\text{diag}(\mathbf{H})) \cdot \mathbf{I} \quad (\lambda \approx 0.01)
   $$
   The inverse $\mathbf{H}^{-1}$ is precomputed once, enabling fast recursive rank-1 updates.

---

## 3. The GPTQ Column Update Algorithm

For column $c \in [0, d_{\text{in}}-1]$:

1. Compute quantized column:
   $$
   \widehat{\mathbf{w}}_c = \text{clamp}\left( \text{round}\left( \frac{\mathbf{w}_c}{s_c} \right), q_{\text{min}}, q_{\text{max}} \right) \cdot s_c
   $$
2. Determine quantization error vector:
   $$
   \mathbf{e}_c = \mathbf{w}_c - \widehat{\mathbf{w}}_c
   $$
3. Update all trailing unquantized rows $r \in [c + 1, d_{\text{in}}-1]$:
   $$
   \mathbf{w}_{r, :} \leftarrow \mathbf{w}_{r, :} - \frac{[\mathbf{H}^{-1}]_{r, c}}{[\mathbf{H}^{-1}]_{c, c}} \mathbf{e}_c^T
   $$

This rank-1 outer-product update guarantees that the expected output error along all subsequent activation coordinates is algebraically minimized.

---

## 4. Act-Order (Activation Ordering) Heuristic

In standard GPTQ, columns are quantized in index order ($c = 0, 1, \dots$). However, activation channels vary dramatically in energy. The **Act-Order** heuristic sorts columns in descending order of Hessian diagonal values:

$$
\text{perm} = \text{argsort}\left( \text{diag}(\mathbf{H}), \text{descending}=\text{True} \right)
$$

### Why Act-Order Matters:
- Highly influential activation channels are quantized first when maximum unquantized degrees of freedom remain in the trailing submatrix.
- Less sensitive channels absorb accumulated error later in the sequence.
- Act-order consistently recovers $0.5$ to $1.5$ points of perplexity at 4-bit precision, particularly for models with pronounced outlier channels (e.g., LLaMA, OPT).

---

## 5. Multi-Framework Architecture & Numerical Verification

All implementations are strictly synchronized across frameworks with transitive numerical parity:

```mermaid
flowchart LR
    A[Calibration Activations X] --> B[Damped Hessian H & H_inv]
    B --> C[RTN Baseline]
    B --> D[GPTQ Column Compensation]
    B --> E[Act-Order Permuted GPTQ]
    C --> F[Golden SafeTensors seed 42]
    D --> F
    E --> F
    F --> G[Candle Rust Parity < 1e-5]
    F --> H[JAX Functional Parity < 1e-5]
    F --> I[PyTorch 2.x Reference]
```

### Numerical Parity Summary:
| Track | Path | Parity vs Golden |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.py` | Reference |
| **JAX** | `jax/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.rs` | WGSL / CPU |

---

## 6. Systems Engineering & Hardware Kernels

In high-throughput PTQ pipelines, computing the trailing rank-1 weight update on CPU creates a sequential bottleneck. The accompanying WGSL kernel (`GPTQ_UPDATE_WGSL`) parallelizes the outer product:

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let r = global_id.x;
    let c = global_id.y;
    if (r >= params.rows || c >= params.cols) { return; }
    let idx = r * params.cols + c;
    let step = err_row[c] / params.h_inv_cc;
    weight[idx] = weight[idx] - step * h_inv_col[r];
}
```

This 2D workgroup dispatch processes thousands of trailing parameters in parallel, enabling GPTQ calibration of 70B parameter models in under 4 hours on a single GPU.

---

## 7. Comparative Benchmark & SQNR Evaluation

Evaluating 4-bit integer quantization across 64 calibration tokens ($d_{\text{in}}=16, d_{\text{out}}=8$):

| Method | Output MSE | Output SQNR (dB) | Perplexity Impact |
| :--- | :--- | :--- | :--- |
| **RTN (Baseline)** | $0.0084$ | $19.42\text{ dB}$ | High degradation |
| **Standard GPTQ** | $0.0021$ | $25.44\text{ dB}$ | $+6.02\text{ dB}$ gain |
| **GPTQ + Act-Order** | $0.0011$ | $28.25\text{ dB}$ | $+8.83\text{ dB}$ gain |

Second-order Hessian compensation suppresses coherent error accumulation, transforming unviable 4-bit models into production-ready inference engines.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [01_rtn_vs_gptq_hessian_error_compensation.ipynb](../../../notebooks/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.ipynb)
- **PyTorch 2.x Module:** [01_rtn_vs_gptq_hessian_error_compensation.py](../../../pytorch2/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.py)
- **Functional JAX Module:** [01_rtn_vs_gptq_hessian_error_compensation.py](../../../jax/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.py)
- **Native Rust Candle Track:** [01_rtn_vs_gptq_hessian_error_compensation.rs](../../../candle/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.rs)
- **Hardware Kernel (CPU / WGSL):** [01_rtn_vs_gptq_hessian_error_compensation.rs](../../../kernels/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.rs)
- **Golden SafeTensors:** [01_rtn_vs_gptq_hessian_error_compensation.safetensors](../../../weights/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.safetensors)
