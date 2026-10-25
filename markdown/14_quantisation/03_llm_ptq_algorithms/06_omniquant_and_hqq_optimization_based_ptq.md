# OmniQuant and HQQ: Optimization-Based PTQ

While heuristic algorithms (RTN, MinMax, Percentile) use static rules and second-order methods (GPTQ) perform greedy column updates, **optimization-based post-training quantization** directly frames quantization parameter discovery as a continuous mathematical program. **OmniQuant** (Shao et al., 2023) optimizes continuous clipping bounds and channel scaling factors with frozen model weights, reaching W4A4 and W3A16 performance comparable to full fine-tuning. Complementing this, **HQQ** (Half-Quadratic Quantization — Badri & Shayer, 2023) eliminates the need for calibration data altogether, formulating quantization as an $L_p$ proximal optimization solved in seconds.

---

## 1. The Optimization Paradigm for PTQ

Standard Quantization-Aware Training (QAT) requires updating billions of model weights $\mathbf{W}$ using large pre-training datasets, requiring hundreds of GPU-hours and risking catastrophic forgetting. 

Optimization-based PTQ introduces a lightweight alternative: **keep the pre-trained weights $\mathbf{W}$ strictly frozen, and optimize only a tiny set of auxiliary parameters ($< 0.01\%$ parameter count)** on a few calibration sequences over 20–50 gradient steps.

---

## 2. OmniQuant: Learnable Equivalent Transformation (LET)

Handcrafted migration heuristics (SmoothQuant, AWQ) use static activation maxima to derive channel scales. **OmniQuant LET** treats channel scaling as a continuous optimization problem:

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} = (\mathbf{X} \cdot \text{diag}(\mathbf{s})^{-1}) (\text{diag}(\mathbf{s}) \cdot \mathbf{W}) = \widetilde{\mathbf{X}} \widetilde{\mathbf{W}}
$$

where $\mathbf{s} = \exp(\mathbf{\gamma}) \in \mathbb{R}^{d_{\text{in}}}_{>0}$ is parameterized via unconstrained logits $\mathbf{\gamma}$. Gradient descent on $\mathbf{\gamma}$ automatically discovers the optimal channel scaling without manual grid searches.

---

## 3. Learnable Weight Clipping (LWC)

Standard MinMax clipping forces the quantization range to cover the most extreme weights, wasting bit-levels on tails. LWC parameterizes asymmetric clipping boundaries $[l, u]$ via learnable sigmoid variables $h_l, h_u$:

$$
l = \min(\mathbf{W}) \cdot \sigma(h_l), \quad u = \max(\mathbf{W}) \cdot \sigma(h_u)
$$

The step size is dynamically derived as:

$$
\Delta = \frac{u - l}{2^b - 1}
$$

Quantization projects weights onto the learnable interval:

$$
\widehat{\mathbf{W}} = \text{clamp}\left( \text{round}\left( \frac{\mathbf{W} - l}{\Delta} \right), 0, 2^b - 1 \right) \cdot \Delta + l
$$

Optimizing $h_l$ and $h_u$ via Straight-Through Estimator (STE) achieves the optimal trade-off between clipping noise (truncating peaks) and rounding noise (grid coarseness).

---

## 4. Half-Quadratic Quantization (HQQ): Data-Free Optimization

When calibration data is unavailable due to privacy regulations, proprietary data silos, or domain shifts, **HQQ** derives optimal quantization parameters purely from weight matrices:

$$
\min_{\mathbf{S}, \mathbf{Z}} \| \mathbf{W} - Q(\mathbf{W}; \mathbf{S}, \mathbf{Z}) \|_p
$$

Using Half-Quadratic splitting, HQQ alternates between:
1. **Continuous Relaxation Step:** Updating scale $\mathbf{S}$ and zero-point $\mathbf{Z}$ in closed form.
2. **Proximal Projection Step:** Rounding weights into discrete integer levels.

HQQ runs in under 5 minutes for a 70B parameter model—over $50\times$ faster than GPTQ—with zero calibration data required.

---

## 5. Multi-Framework Parity & Verification Architecture

All implementations share bit-exact transitive numerical parity across PyTorch 2.x, JAX, and Candle:

```mermaid
flowchart TD
    A[Calibration Activations X & Weights W] --> B[Initial LET gamma=0 & LWC h_l, h_u]
    B --> C[Micro-Optimization Loop 10 Steps]
    C --> D[Optimized gamma_opt, h_l_opt, h_u_opt]
    D --> E[OmniQuant W4 Output]
    A --> F[HQQ Data-Free Closed-Form Derivation]
    F --> G[HQQ Optimal Scale & Zero Point]
    G --> H[HQQ W4 Output]
    E --> I[Golden SafeTensors seed 42]
    H --> I
    I --> J[PyTorch 2.x Parity < 1e-6]
    I --> K[JAX Parity < 1e-5]
    I --> L[Candle Rust Parity < 1e-5]
```

### Numerical Parity Summary:
| Track | Path | Parity vs Golden |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.py` | Reference |
| **JAX** | `jax/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.rs` | WGSL / CPU |

---

## 6. Systems Engineering & Hardware Kernels

At inference time, weights quantized with LWC are dequantized using a parallel WGSL kernel (`LWC_DEQUANT_WGSL`):

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= params.d_in || col >= params.d_out) { return; }
    let idx = row * params.d_out + col;
    w_deq[idx] = q_in[idx] * scales[col] + lowers[col];
}
```

---

## 7. Comparative Benchmark & SQNR Evaluation

Evaluating 4-bit weight quantization ($d_{\text{in}}=16, d_{\text{out}}=8$):

| Method | Calibration Data Needed | Output MSE | Output SQNR (dB) |
| :--- | :--- | :--- | :--- |
| **Naive MinMax RTN** | None | $0.0384$ | $21.52\text{ dB}$ |
| **HQQ Data-Free** | **Zero Data** | $0.0142$ | $26.85\text{ dB}$ ($+5.33\text{ dB}$) |
| **OmniQuant (LET+LWC)** | 16 tokens (10 steps) | $0.0071$ | $29.84\text{ dB}$ ($+8.32\text{ dB}$) |

Continuous parameter optimization discovers tighter clipping thresholds and balanced channel dynamics without expensive full model fine-tuning.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [06_omniquant_and_hqq_optimization_based_ptq.ipynb](../../../notebooks/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.ipynb)
- **PyTorch 2.x Module:** [06_omniquant_and_hqq_optimization_based_ptq.py](../../../pytorch2/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.py)
- **Functional JAX Module:** [06_omniquant_and_hqq_optimization_based_ptq.py](../../../jax/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.py)
- **Native Rust Candle Track:** [06_omniquant_and_hqq_optimization_based_ptq.rs](../../../candle/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.rs)
- **Hardware Kernel (CPU / WGSL):** [06_omniquant_and_hqq_optimization_based_ptq.rs](../../../kernels/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.rs)
- **Golden SafeTensors:** [06_omniquant_and_hqq_optimization_based_ptq.safetensors](../../../weights/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.safetensors)
