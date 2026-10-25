# Block Floating Point (BFP), Microscaling (MXFP), and NVFP4

## 1. Mathematical Foundations & Microscaling Taxonomy

As large language models scale into hundreds of billions of parameters, standard uniform low-precision representations face severe quantization degradation:
1. **The Outlier Dilemma in Sub-8-bit Quantization**: In tensor-wide quantization, a few extreme outlier activation spikes (e.g. $|\mathbf{x}| > 50\sigma$) force the entire tensor's scale factor to blow up, compressing the remaining $99.9\%$ of normal features into only $1$ or $2$ quantization buckets, causing catastrophic degradation.
2. **Block Floating Point (BFP, Wilkinson 1963)**: Divides tensors into small contiguous tiles/blocks of length $K \in \{16, 32\}$. A single common exponent is shared across all elements in the block, allowing local scale adaptation without per-element exponent overhead.
3. **OCP Microscaling (MX) Specification**: An open industry standard jointly developed by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm (2023) establishing formal representations for:
   - **MXFP8** (E5M2 or E4M3 with block size 32)
   - **MXFP6** (E3M2 or E2M3 with block size 32)
   - **MXFP4** (E2M1 with block size 32)
   - **MXINT8** (8-bit signed integer with block size 32)
4. **NVIDIA Blackwell NVFP4**: Hardware-accelerated 4-bit floating point format with two-level scaling (tensor-level FP8 scale + block-level 8-bit scale per 16 elements), achieving up to $20$ PFLOPS of FP4 compute per GPU.

---

## 2. Block Floating Point (BFP) Shared Exponent Mechanics

Given a vector block $\mathbf{x} = [x_1, x_2, \dots, x_K] \in \mathbb{R}^K$:

### A. Shared Exponent Extraction
The block identifies the maximum absolute element:

$$a_{\text{max}} = \max_{i=1}^K |x_i|$$

The shared power-of-2 exponent is:

$$E_{\text{shared}} = \lfloor \log_2(a_{\text{max}}) \rfloor, \quad S_{\text{shared}} = 2^{E_{\text{shared}}}$$

### B. Mantissa Quantization
Elements are normalized relative to the shared scale and rounded to an $M$-bit signed integer ($M \in \{4, 8, 12, 16\}$):

$$m_i = \operatorname{clamp}\left( \operatorname{round}\left( \frac{x_i}{S_{\text{shared}}} \times (2^{M-1} - 1) \right), \; -(2^{M-1} - 1), \; 2^{M-1} - 1 \right)$$

### C. Dequantized Reconstruction
$$\hat{x}_i = \frac{m_i}{2^{M-1} - 1} \times S_{\text{shared}}$$

---

## 3. The OCP Microscaling (MX) Specification

The OCP Microscaling standard defines **E8M0 shared scales**:
- Scale factor is an 8-bit unsigned power of two: $S \in \{2^{-127}, \dots, 2^{127}\}$.
- Eliminates expensive hardware multipliers for scale application (implemented as simple exponent addition).

```
┌────────────────────────────────────────────────────────────────────────┐
│ OCP MX Block (32 elements)                                             │
├────────────────────────────────┬───────────────────────────────────────┤
│ Shared Scale (8-bit E8M0)      │ 32 x Minifloat Elements (e.g. FP4)    │
│ S = 2^(e - 127)                │ 32 x 4 bits = 128 bits (16 bytes)     │
└────────────────────────────────┴───────────────────────────────────────┘
Total footprint: 1 byte (scale) + 16 bytes (elements) = 17 bytes -> 4.25 bits/element!
```

### Scale Computation Formula for MXFP4
For FP4 E2M1 (where maximum representable element is $6.0$):

$$e_{\text{scale}} = \left\lceil \log_2\left( \frac{\max_{i \in \text{block}} |x_i|}{6.0} \right) \right\rceil, \quad S_{\text{block}} = 2^{e_{\text{scale}}}$$

---

## 4. NVIDIA Blackwell NVFP4 Two-Level Architecture

To maximize dynamic range while keeping local registers compact, NVIDIA Blackwell implements **two-level scaling**:

$$\hat{x}_{i} = \tilde{x}_i \times S_{\text{local}} \times S_{\text{global}}$$

1. **Global Scale ($S_{\text{global}}$)**: Stored per-tensor or per-channel in FP8/FP16.
2. **Local Scale ($S_{\text{local}}$)**: Stored per micro-block of 16 elements in an 8-bit minifloat (E4M3 or E8M0).
3. **Element Mantissa ($\tilde{x}_i$)**: 4-bit FP4 E2M1 format ($\{0, \pm 0.5, \pm 1.0, \pm 1.5, \pm 2.0, \pm 3.0, \pm 4.0, \pm 6.0\}$).

---

## 5. Numerical Error Analysis & SQNR

Microscaling achieves dramatically higher Signal-to-Quantization-Noise Ratio (SQNR) than tensor-wide quantization:

$$\text{SQNR} = 10 \log_{10} \frac{\sum_i x_i^2}{\sum_i (x_i - \hat{x}_i)^2}$$

- **Tensor-wide FP4**: When outliers exceed normal values by $10\times$, effective bit depth drops to $< 1.5$ bits ($\text{SQNR} < 8\text{ dB}$).
- **MXFP4 (Block size 32)**: Outliers are isolated within a single block of 32 elements. The remaining blocks adapt their scales locally, preserving $\text{SQNR} \approx 18-22\text{ dB}$, matching unquantized FP16 downstream perplexity.

---

## 6. Asymptotic Time & Memory Complexity

- **Compute Complexity**: Block scale extraction requires an intra-block reduction: $\mathcal{O}(K)$ comparisons per block $\implies \mathcal{O}(N)$ total for $N$ elements.
- **Memory Footprint**:
  - Uncompressed FP32: $32\text{ bits/element}$.
  - FP16/BF16: $16\text{ bits/element}$.
  - FP8: $8\text{ bits/element}$.
  - MXFP4 (block 32): $4 + \frac{8}{32} = 4.25\text{ bits/element}$ ($3.76\times$ compression over FP16).
  - NVFP4 (block 16): $4 + \frac{8}{16} = 4.5\text{ bits/element}$ ($3.55\times$ compression over FP16).

---

## 7. Numerical Invariants and Precision Contracts

- **Exact Scale Preservation**: All block scales must evaluate strictly to exact integer powers of 2 ($2^k$), ensuring zero floating point rounding error in scale application.
- **Outlier Containment**: An outlier in block $b$ must have zero numerical influence on the scale or quantized values of block $b+1$.
- **Transitive Parity**: Parity error must satisfy $\max |x_{\text{Candle}} - x_{\text{PyTorch}}| < 10^{-5}$ across all test cases.

---

## 8. Hardware Dispatch & Micro-Architecture Realities

- **NVIDIA Blackwell Tensor Cores**: Direct native execution of NVFP4 GEMMs via 5th-generation Tensor Cores with hardware-managed scale registers.
- **AMD CDNA3 / Intel Gaudi**: Native hardware support for OCP MXFP8 and MXFP4 primitives.
- **Software / WebGPU Fallback**: WGSL compute shaders unpack nibbles (2 elements per byte) and execute parallel dequantization with workgroup size 64.
