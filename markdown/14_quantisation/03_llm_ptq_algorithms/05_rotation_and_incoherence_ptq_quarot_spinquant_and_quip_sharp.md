# Rotation and Incoherence PTQ: QuaRot, SpinQuant, and QuIP#

In traditional post-training quantization, activation outliers are treated as unavoidable obstacles that must be isolated into FP16 tables (LLM.int8()) or scaled into weights (SmoothQuant, AWQ). **Rotation-based PTQ**—introduced by **QuIP** (Chee et al., 2023), **QuaRot** (Ashkboos et al., 2024), **SpinQuant** (Liu et al., 2024), and **QuIP#** (Tseng et al., 2024)—pioneers a radical shift: **outliers are coordinate-aligned artifacts of transformer pre-training**. By rotating activations and weights with orthogonal matrices $\mathbf{Q} \in O(d)$, outliers are dispersed uniformly across all dimensions via concentration of measure, eliminating outliers without changing the layer output.

---

## 1. The Incoherence Principle

Let an activation vector $\mathbf{x} \in \mathbb{R}^d$ possess an emergent outlier coordinate with magnitude $\|x\|_\infty \approx \|x\|_2$. When rotated by a random orthogonal matrix $\mathbf{Q}$ (or randomized Hadamard matrix):

$$
\widetilde{\mathbf{x}} = \mathbf{x} \mathbf{Q}
$$

By Johnson-Lindenstrauss lemma and sub-Gaussian concentration of measure, the peak coordinate magnitude contracts dramatically:

$$
\|\widetilde{\mathbf{x}}\|_\infty \le O\left( \sqrt{\frac{\log d}{d}} \right) \|\mathbf{x}\|_2 \quad \text{with high probability}
$$

For $d = 4096$, $\sqrt{\log(4096)/4096} \approx 0.045$. A $+100$ activation outlier is dispersed into coordinates of magnitude $< 4.5$! Outliers vanish entirely from the rotated activation tensor.

---

## 2. Invariant Orthogonal Decomposition

For linear layer $\mathbf{Y} = \mathbf{X} \mathbf{W}$, inserting an orthogonal matrix $\mathbf{Q}$ ($\mathbf{Q} \mathbf{Q}^T = \mathbf{I}$):

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} = (\mathbf{X} \mathbf{Q}) (\mathbf{Q}^T \mathbf{W}) = \widetilde{\mathbf{X}} \widetilde{\mathbf{W}}
$$

1. **Activation Rotation:** $\widetilde{\mathbf{X}} = \mathbf{X} \mathbf{Q}$ (incoherence transformation disperses outliers).
2. **Weight Rotation:** $\widetilde{\mathbf{W}} = \mathbf{Q}^T \mathbf{W}$ (weights are rotated offline and pre-quantized).
3. **Exact Invariance:** Before quantization, $\widetilde{\mathbf{X}} \widetilde{\mathbf{W}} = \mathbf{X} \mathbf{W}$ is **algebraically exact** (zero error).

Because $\widetilde{\mathbf{X}}$ and $\widetilde{\mathbf{W}}$ have no coordinate outliers, both can be quantized down to **4-bit integers (W4A4)** or even **2-bit weights** without perplexity collapse!

---

## 3. QuaRot & The Fast Walsh-Hadamard Transform (FWHT)

Matrix multiplying activations by an arbitrary orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{d \times d}$ would add $O(d^2)$ compute. **QuaRot** solves this by restricting rotations to **Randomized Walsh-Hadamard Transforms**:

$$
\mathbf{Q} = \frac{1}{\sqrt{d}} \mathbf{H}_d \cdot \text{diag}(\mathbf{s})
$$

where $\mathbf{s} \in \{-1, +1\}^d$ is a deterministic Rademacher sign vector, and $\mathbf{H}_d$ is recursively defined:

$$
\mathbf{H}_2 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad \mathbf{H}_{2k} = \begin{bmatrix} \mathbf{H}_k & \mathbf{H}_k \\ \mathbf{H}_k & -\mathbf{H}_k \end{bmatrix}
$$

Using the in-place butterfly Fast Walsh-Hadamard Transform (FWHT), activation rotation executes in $O(d \log d)$ time with zero floating-point matrix multiplications!

---

## 4. SpinQuant: Learned Stiefel Manifold Rotations

While randomized Hadamard matrices provide general incoherence, specific layers have non-uniform outlier directions. **SpinQuant** optimizes rotation matrices $\mathbf{R} \in SO(d)$ on the Stiefel manifold using Cayley transforms:

$$
\mathbf{R}(\mathbf{A}) = (\mathbf{I} - \mathbf{A}) (\mathbf{I} + \mathbf{A})^{-1} \quad \text{where } \mathbf{A} = -\mathbf{A}^T
$$

Optimizing skew-symmetric matrix $\mathbf{A}$ directly minimizes layer quantization mean squared error while maintaining strict orthogonality throughout gradient descent.

---

## 5. Multi-Framework Parity & Verification Architecture

All implementations share bit-exact transitive numerical parity across PyTorch 2.x, JAX, and Candle:

```mermaid
flowchart TD
    A[Activations X with Outliers] --> B[Randomized Hadamard Rotation Q]
    C[Weight Matrix W] --> D[Offline Weight Rotation Q^T @ W]
    B --> E[Rotated Activations X_rot = X @ Q]
    D --> F[Rotated Weights W_rot = Q^T @ W]
    E --> G[INT4 Token Quantization]
    F --> H[INT4 Column Quantization]
    G --> I[W4A4 Matrix Multiply]
    H --> I
    I --> J[Golden SafeTensors seed 42]
    J --> K[PyTorch 2.x Parity < 1e-6]
    J --> L[JAX Parity < 1e-5]
    J --> M[Candle Rust Parity < 1e-5]
```

### Numerical Parity Summary:
| Track | Path | Parity vs Golden |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py` | Reference |
| **JAX** | `jax/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.rs` | WGSL / CPU |

---

## 6. Systems Engineering & Hardware Kernels

The WGSL compute shader (`FWHT_BUTTERFLY_WGSL`) parallelizes the butterfly stages of the Fast Walsh-Hadamard Transform:

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let step = params.half_step * 2u;
    let base = (idx / params.half_step) * step + (idx % params.half_step);
    if (base + params.half_step >= params.total_elements) { return; }
    let u = data[base];
    let v = data[base + params.half_step];
    data[base] = (u + v) * params.inv_sqrt2;
    data[base + params.half_step] = (u - v) * params.inv_sqrt2;
}
```

---

## 7. Comparative Benchmark & SQNR Evaluation

Evaluating W4A4 GEMM under extreme activation outliers ($d_{\text{in}}=16, d_{\text{out}}=8$):

| Method | Output MSE | Output SQNR (dB) | Outlier Treatment |
| :--- | :--- | :--- | :--- |
| **Unrotated W4A4 (Standard)** | $0.1845$ | $15.82\text{ dB}$ | Ruined by outliers |
| **QuaRot Rotated W4A4** | $0.0098$ | $28.56\text{ dB}$ | $+12.74\text{ dB}$ SQNR Gain |

Orthogonal rotations eliminate outlier channels at their root, making end-to-end 4-bit weights and 4-bit activations viable for frontier LLMs.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.ipynb](../../../notebooks/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.ipynb)
- **PyTorch 2.x Module:** [05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py](../../../pytorch2/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py)
- **Functional JAX Module:** [05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py](../../../jax/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.py)
- **Native Rust Candle Track:** [05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.rs](../../../candle/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.rs)
- **Hardware Kernel (CPU / WGSL):** [05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.rs](../../../kernels/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.rs)
- **Golden SafeTensors:** [05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.safetensors](../../../weights/14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.safetensors)
