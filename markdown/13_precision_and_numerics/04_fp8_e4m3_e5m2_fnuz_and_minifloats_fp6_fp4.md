# FP8 (E4M3, E5M2, FNUZ) and Minifloats (FP6, FP4)

## 1. Mathematical Foundations & Minifloat Taxonomy

Modern deep learning accelerators (NVIDIA Hopper H100, Blackwell B200, AMD MI300, Intel Gaudi) introduce sub-8-bit floating point formats to multiply compute density and reduce memory traffic:
1. **The 8-bit Frontier (Micikevicius et al., 2022)**: Moving from 16-bit floats to 8-bit floats doubles matrix multiplication compute throughput (e.g. 2000 TFLOPS on H100 FP8 Tensor Cores) and halves memory bandwidth demand while preserving convergence.
2. **Asymmetric Precision Specialization**: Unlike FP16, which uses a single universal bit allocation, 8-bit training splits into two specialized complementary formats:
   - **FP8 E4M3**: 1 sign bit, 4 exponent bits, 3 mantissa bits. High mantissa precision for forward activations and model weights.
   - **FP8 E5M2**: 1 sign bit, 5 exponent bits, 2 mantissa bits. Wide dynamic range matching FP16 for backpropagated gradients.
3. **Sub-8-Bit Minifloats (FP6 and FP4)**: As LLMs scale to hundreds of billions of parameters, post-training quantization and microscaling push towards 6-bit (E3M2, E2M3) and 4-bit (E2M1, NVFP4) minifloats, providing superior non-linear resolution compared to uniform integer quantization (INT4).

---

## 2. Bitwise Anatomy & Encoding Rules

The general minifloat formulation is parameterized by $(S, E, M, \text{bias})$:

$$x = (-1)^s \times 2^{e - \text{bias}} \times \left(1 + \sum_{i=1}^M m_i 2^{-i}\right)$$

| Format | Total Bits | Sign ($S$) | Exponent ($E$) | Mantissa ($M$) | Exponent Bias | Max Normal | Min Subnormal | Special Values |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FP8 E4M3FN** | 8 | 1 | 4 | 3 | 7 | $448.0$ | $2^{-9} \approx 0.00195$ | No Inf; $0\text{x}7\text{F}, 0\text{xFF}$ are NaN |
| **FP8 E5M2** | 8 | 1 | 5 | 2 | 15 | $57344.0$ | $2^{-16} \approx 1.53 \times 10^{-5}$ | IEEE standard $\pm \infty$ and NaN |
| **FP8 E4M3FNUZ** | 8 | 1 | 4 | 3 | 8 | $240.0$ | $2^{-10} \approx 0.00098$ | Unsigned Zero; $0\text{x}80$ is NaN |
| **FP6 E3M2** | 6 | 1 | 3 | 2 | 3 | $28.0$ | $2^{-4} = 0.0625$ | No Inf |
| **FP4 E2M1** | 4 | 1 | 2 | 1 | 1 | $6.0$ | $2^{-1} = 0.5$ | $\{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$ |

---

## 3. Subnormal Numbers & Underflow Mechanics

In standard floating point formats, when the exponent $e$ reaches the minimum representable code $0$, the implicit leading bit transitions from $1$ to $0$:

$$\text{If } e < 1 - \text{bias} \implies x_{\text{subnormal}} = (-1)^s \times 2^{1 - \text{bias}} \times \left(0 + \sum_{i=1}^M m_i 2^{-i}\right)$$

- **Gradual Underflow**: Subnormals provide equispaced quantization bins between zero and the minimum normalized positive number $2^{1 - \text{bias}}$.
- **Quantization Step Size**:
  $$\Delta = \begin{cases} 2^{1 - \text{bias} - M}, & \text{subnormal regime} \\ 2^{\lfloor \log_2(|x|) \rfloor - M}, & \text{normal regime} \end{cases}$$

---

## 4. Delayed Scaling Dynamics in Tensor Core FP8 GEMM

Because FP8 E4M3 has a limited dynamic range ($\pm 448.0$), raw activations and weights must be dynamically scaled into the optimal range of the format:

```
  Matrix A (FP32/FP16)                      Matrix B (FP32/FP16)
         │                                         │
         ▼ (Scale: 1 / s_A)                        ▼ (Scale: 1 / s_B)
  Quantize to E4M3FN                        Quantize to E4M3FN
         │                                         │
         └─────────────► FP8 Tensor Core ◄─────────┘
                               │ (A_q @ B_q)
                               ▼
                        Scaled Accumulator
                               │
                               ▼ (Rescale: * s_A * s_B)
                         Output (FP32 / FP16)
```

### The Delayed Scale Update Rule
To avoid synchronizing the GPU across all threads to compute the absolute maximum ($\operatorname{amax}$) on the critical path of each GEMM, NVIDIA Transformer Engine utilizes **delayed scaling**:
1. GEMM at step $t$ executes using the scale factors $s_t$ computed at step $t-1$.
2. Simultaneously, an asynchronous reduction tracks the tensor's current $\operatorname{amax}_t = \max(|X_t|)$.
3. The scale factor for step $t+1$ is updated asynchronously:
   $$s_{t+1} = \frac{\operatorname{amax}_t}{\text{max\_representable}} \times \text{margin}$$

---

## 5. E4M3 vs E5M2 Role Allocation

1. **Forward Pass (Activations & Weights $\to$ E4M3)**:
   - Neural network activations and weights follow bell-shaped (Gaussian-like) distributions centered around 0.
   - 3 mantissa bits provide $8$ quantization bins per octave, yielding a Signal-to-Quantization-Noise Ratio (SQNR) $\approx 6 \times 3 = 18\text{ dB}$, which preserves downstream cross-entropy loss.
2. **Backward Pass (Gradients $\to$ E5M2)**:
   - Backpropagated error gradients exhibit heavy tails and span many orders of magnitude across layers.
   - 5 exponent bits prevent catastrophic underflow, ensuring gradients down to $1.5 \times 10^{-5}$ do not flush to zero.

---

## 6. Sub-8-Bit Minifloats: FP6 and FP4 Bit Layouts

1. **FP6 (E3M2 / E2M3)**:
   - Used in LLM inference caching (FP6 KV-cache compression) where INT4 degrades perplexity while INT8 wastes DRAM capacity.
   - Packed efficiently into 32-bit words (e.g. five 6-bit numbers per 32-bit word with 2 padding bits, or interleaved 3-byte pairs).
2. **FP4 (OCP E2M1)**:
   - Employs 1 sign bit, 2 exponent bits (bias = 1), and 1 mantissa bit.
   - Produces the exact non-negative grid:
     $$\{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$$
   - Paired with micro-block scaling factors (e.g. 1 scale per 16 or 32 values) in the OCP Microscaling (MX) standard.

---

## 7. Numerical Invariants and Precision Contracts

- **Subnormal Linearity**: For all $|x| < 2^{1 - \text{bias}}$, quantized output must be an exact integer multiple of the subnormal step $\Delta = 2^{1 - \text{bias} - M}$.
- **Saturation Bounds**: For all $|x| > \text{max\_val}$, quantized output must clamp strictly to $\operatorname{sgn}(x) \times \text{max\_val}$ (in E4M3FN, clamping to $448.0$).
- **Transitive Parity**: All minifloat simulations and delayed scaled GEMMs must achieve exact bit-level parity or max absolute error $< 10^{-5}$ across PyTorch, JAX, and Candle.

---

## 8. Hardware Dispatch & Empirical Validation

- **CUDA / Tensor Cores**: `tl.float8e4nv` / `torch.float8_e4m3fn` dispatched via cuBLAS and CUTLASS kernels.
- **CPU / WebGPU**: Software emulation with bit-field manipulation or arithmetic step-rounding with WGSL parallel compute shaders.
