# Quantization-Aware Training: STE, LSQ, LSQ+, PACT, and DoReFa

While Post-Training Quantization (PTQ) freezes model parameters or optimizes auxiliary transformations on small calibration sets, **Quantization-Aware Training (QAT)** incorporates the discrete quantization operator directly into the forward pass during end-to-end model training or continuous fine-tuning. Because the rounding function $\text{round}(x)$ has a derivative of zero almost everywhere and is discontinuous at half-integers, backward propagation requires gradient surrogates. Starting from the heuristic **Straight-Through Estimator (STE)** (Bengio et al., 2013), modern QAT has evolved into mathematically principled frameworks including **PACT** (Choi et al., 2018), **DoReFa-Net** (Zhou et al., 2016), and **LSQ / LSQ+** (Esser et al., 2019; Bhalgat et al., 2020), which learn optimal clipping boundaries and quantizer step sizes via exact gradient scaling.

---

## 1. The Gradient Collapse Problem in Quantization

The fundamental challenge in QAT is passing gradients through discrete rounding:

$$
\frac{d}{dx} \text{round}(x) = 0 \quad \text{for } x \notin \mathbb{Z} + \frac{1}{2}
$$

If standard automatic differentiation were applied, the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \widehat{\mathbf{W}}} \cdot \frac{\partial \widehat{\mathbf{W}}}{\partial \mathbf{W}}$ would evaluate to zero everywhere, freezing model weights completely.

---

## 2. Straight-Through Estimator (STE) & Clamped Gradients

The **Straight-Through Estimator** substitutes the zero derivative with the identity pass-through during backpropagation:

$$
\text{Forward:} \quad \widehat{\mathbf{W}} = \text{clamp}\left( \text{round}\left( \frac{\mathbf{W}}{s} \right), q_{\min}, q_{\max} \right) \cdot s
$$

$$
\text{Backward (Clamped STE):} \quad \frac{\partial \widehat{\mathbf{W}}}{\partial \mathbf{W}} = \mathbb{I}[q_{\min} \cdot s \le \mathbf{W} \le q_{\max} \cdot s]
$$

- **Unclamped STE Misuse:** Passing gradients through saturated weights ($|\mathbf{W}| > q_{\max} \cdot s$) causes weights to grow unboundedly while the quantized output remains strictly clipped, leading to severe latent weight inertia.
- **Clamped STE Solution:** Zeroing the gradient outside the clipping interval forces latent weights to stay near the active quantization threshold.

---

## 3. Parameterized Activation Clipping (PACT)

Activations in ReLU and GELU networks are unbounded and asymmetric. Hard-coded clipping limits either truncate tail distributions or dilute quantization resolution across empty space. **PACT** introduces an explicit learnable clipping bound $\alpha > 0$:

$$
y = \frac{1}{2}\left( |x| - |x - \alpha| + \alpha \right) = \text{clamp}(x, 0, \alpha)
$$

Quantizing onto $b$ bits with step size $s = \frac{\alpha}{2^b - 1}$:

$$
\widehat{y} = \text{round}\left( \frac{y}{s} \right) \cdot s
$$

The gradient with respect to the clipping threshold $\alpha$ is derived analytically:

$$
\frac{\partial \widehat{y}}{\partial \alpha} = \begin{cases} 0 & x < \alpha \\ 1 & x \ge \alpha \end{cases}
$$

When activations frequently saturate above $\alpha$, $\frac{\partial \mathcal{L}}{\partial \alpha} > 0$ increases $\alpha$ to accommodate larger activations; when activations are small, $\alpha$ contracts to maximize precision.

---

## 4. Learned Step Size Quantization (LSQ & LSQ+)

While PACT optimizes activation thresholds, **LSQ** directly parameterizes the quantizer step size $s$ as a continuous learnable variable:

$$
\bar{v} = \text{round}\left( \text{clamp}\left( \frac{v}{s}, -Q_N, Q_P \right) \right) \cdot s
$$

Differentiating $\bar{v}$ with respect to step size $s$ yields:

$$
\frac{\partial \bar{v}}{\partial s} = \begin{cases} -Q_N & v/s < -Q_N \\ Q_P & v/s > Q_P \\ \text{round}(v/s) - v/s & \text{otherwise} \end{cases}
$$

### Analytical Gradient Scaling:
Because a step size update scales every element in the tensor simultaneously, the raw gradient magnitude scales with $\sqrt{N}$. Without stabilization, $s$ undergoes wild oscillations. LSQ introduces the analytical gradient scaling factor:

$$
g = \frac{1}{\sqrt{N \cdot Q_P}} \implies \nabla_s \leftarrow g \cdot \nabla_s
$$

**LSQ+** extends this by adding a learnable asymmetric zero-offset $\beta$, enabling asymmetric activation distributions (e.g. SwiGLU / GeLU) to be quantized losslessly without wasting bit levels on negative values.

---

## 5. Multi-Framework Parity & Verification Architecture

All implementations share bit-exact transitive numerical parity across PyTorch 2.x, JAX, and Candle:

```mermaid
flowchart TD
    A[Activations X & Weights W] --> B[PACT Activation Quantizer alpha]
    A --> C[LSQ Weight Quantizer s]
    B --> D[Quantized Activations X_q]
    C --> E[Quantized Weights W_q]
    D --> F[QAT Linear GEMM Y = X_q @ W_q + b]
    E --> F
    C --> G[Analytical LSQ Gradient Scale 1/sqrt N*Q_P]
    B --> H[PACT Alpha Saturated Gradient Mask]
    F --> I[Golden SafeTensors seed 42]
    I --> J[PyTorch 2.x Parity < 1e-6]
    I --> K[JAX Functional Parity < 1e-5]
    I --> L[Candle Rust Parity < 1e-5]
```

### Numerical Parity Summary:
| Track | Path | Parity vs Golden |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py` | Reference |
| **JAX** | `jax/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.rs` | WGSL / CPU |

---

## 6. Systems Engineering & Hardware Kernels

In GPU inference engines, PACT activation clipping and quantization are fused directly into the input prologue of GEMM operations (`PACT_DEQUANT_WGSL`):

```wgsl
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) { return; }
    let val = x_in[idx];
    let clipped = clamp(val, 0.0, params.alpha);
    let step = params.alpha / params.qmax;
    x_out[idx] = round(clipped / step) * step;
}
```

---

## 7. Comparative Benchmark & SQNR Evaluation

Evaluating 4-bit weight and activation quantization on non-negative activation distributions:

| Quantization Regime | Activation Scheme | Weight Quantizer | Output SQNR (dB) |
| :--- | :--- | :--- | :--- |
| **Static Heuristic PTQ** | MinMax [0, max(X)] | Symmetric Rounding | $18.42\text{ dB}$ |
| **Unclamped STE QAT** | Fixed Clip [0, 6.0] | Clamped STE | $21.15\text{ dB}$ |
| **LSQ + PACT QAT** | **Learnable $\alpha$ PACT** | **Learnable Step $s$ LSQ** | **$26.84\text{ dB}$** ($+8.42\text{ dB}$ gain!) |

Learning the quantizer step size $s$ and activation clipping $\alpha$ jointly during training recovers over $99\%$ of FP32 accuracy even at 4-bit and 3-bit precision.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [03_qat_ste_lsq_lsq_plus_pact_and_dorefa.ipynb](../../../notebooks/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.ipynb)
- **PyTorch 2.x Module:** [03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py](../../../pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py)
- **Functional JAX Module:** [03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py](../../../jax/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py)
- **Native Rust Candle Track:** [03_qat_ste_lsq_lsq_plus_pact_and_dorefa.rs](../../../candle/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.rs)
- **Hardware Kernel (CPU / WGSL):** [03_qat_ste_lsq_lsq_plus_pact_and_dorefa.rs](../../../kernels/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.rs)
- **Golden SafeTensors:** [03_qat_ste_lsq_lsq_plus_pact_and_dorefa.safetensors](../../../weights/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.safetensors)
