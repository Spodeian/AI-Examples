# INT, UINT, Fixed-Point (Q-Format), Dyadic Rescaling, and Residue Number System (RNS) Arithmetic

## 1. Mathematical Foundations & Historical Lineage

Deploying deep neural networks to edge microcontrollers, mobile NPUs, and inference servers requires replacing power-hungry 32-bit floating point multiply-accumulate (MAC) units with compact, energy-efficient integer ALUs:
1. **Residue Number System (RNS, Garner 1959)**: Pioneered carry-free modular arithmetic across pairwise coprime moduli, decomposing wide dynamic-range arithmetic into parallel independent narrow-bit channels.
2. **Dynamic Fixed-Point (DFP, Courbariaux et al., 2014; Gysel et al., 2016)**: Introduced layer-specific radix points ($Q_{m.n}$), bounding word lengths without floating point logic.
3. **Integer-Arithmetic-Only Quantization (Jacob et al., CVPR 2018)**: Formalized affine asymmetric quantization with integer-only dyadic multiplier rescaling, eliminating all floating-point hardware instructions from inference loops.

---

## 2. Uniform Quantization Algebra: Symmetric vs. Affine

Uniform quantization partitions a continuous real interval into $2^b$ uniform step intervals of width $s$:

```
Continuous Range: [xmin ───────────── 0 ────────────────── xmax]
                        │             │                   │
                        ▼             ▼                   ▼
Affine Quantized: [0 ──────────────── z ─────────────── 2^b - 1]
Symmetric Quant:  [-2^(b-1) ───────── 0 ──────────── 2^(b-1) - 1]
```

### A. Affine (Asymmetric) Quantization
Maps continuous range $[x_{\min}, x_{\max}]$ to unsigned integers $[0, 2^b - 1]$ using scale $s$ and integer zero-point $z$:

$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}, \quad z = \operatorname{clamp}\left( \operatorname{round}\left( -\frac{x_{\min}}{s} \right), \; 0, \; 2^b - 1 \right)$$

$$q = \operatorname{clamp}\left( \operatorname{round}\left(\frac{x}{s}\right) + z, \; 0, \; 2^b - 1 \right), \quad \hat{x} = (q - z) \cdot s$$

### B. Symmetric Quantization
Constrains $z = 0$, aligning the numerical zero with integer zero:

$$s = \frac{\max(|x_{\min}|, |x_{\max}|)}{q_{\max}}$$

$$\text{Restricted Range: } q \in [-(2^{b-1} - 1), \; 2^{b-1} - 1], \quad \text{Full Range: } q \in [-2^{b-1}, \; 2^{b-1} - 1]$$

> [!CAUTION]
> Applying symmetric quantization to strictly non-negative post-ReLU/SiLU activations wastes $50\%$ of all representable bins in the negative half-plane, losing $1.0\text{ full bit}$ of precision ($\approx 6\text{ dB SQNR}$ drop)! Always use affine quantization for one-sided activation distributions.

---

## 3. Fixed-Point Qm.n Arithmetic

The $Q_{m.n}$ format represents real numbers with $m$ integer bits (including sign) and $n$ fractional bits:

$$x \approx \frac{q}{2^n}, \quad q \in [-2^{m+n-1}, \; 2^{m+n-1} - 1]$$

- **Quantization Step**: $\Delta = 2^{-n}$.
- **Addition**: Directly computed as $q_A + q_B$ when $n_A = n_B$.
- **Multiplication**: $q_A \times q_B$ produces $2n$ fractional bits, requiring an arithmetic right-shift by $n$ bits: $q_{A \times B} = (q_A \times q_B) \gg n$.

---

## 4. Dyadic Multiplier Rescaling (GEMMLOWP)

In integer-only matrix multiplication, computing $\hat{Y} = \hat{X} \cdot \hat{W}$ produces an accumulator with effective scale $S_{\text{acc}} = S_X \cdot S_W$.
To rescale to output precision $S_Y$, the accumulator must be multiplied by the real scalar:

$$M = \frac{S_X \cdot S_W}{S_Y} \in (0, 1)$$

To execute this without floating point hardware, $M$ is decomposed into a 32-bit normalized integer multiplier $M_0 \in [2^{30}, 2^{31}-1]$ and a right-shift exponent $n \ge 0$:

$$M \approx M_0 \cdot 2^{-n}$$

The integer rescaling is evaluated using standard 32-bit high-multiplication:

$$\operatorname{Rescale}(A) = \left\lfloor \frac{A \cdot M_0 + 2^{30}}{2^{31}} \right\rfloor \gg (n - 31)$$

---

## 5. Residue Number System (RNS) Carry-Free Modular Arithmetic

Given a set of pairwise coprime integer moduli $\mathcal{M} = \{m_1, m_2, \dots, m_k\}$ with dynamic range $M = \prod_{i=1}^k m_i$:

### A. RNS Representation
Every integer $X \in [0, M-1]$ has a unique representation:

$$X \longleftrightarrow (x_1, x_2, \dots, x_k), \quad x_i = X \bmod m_i$$

### B. Carry-Free Parallel Arithmetic
Arithmetic operations execute independently in parallel across narrow bit-width channels:

$$(A \star B) \longleftrightarrow \left( (a_1 \star b_1) \bmod m_1, \; \dots, \; (a_k \star b_k) \bmod m_k \right)$$

No carry propagation occurs between channels, enabling low-latency, high-frequency DSP execution.

### C. Chinese Remainder Theorem (CRT) Reconstruction
Let $M_i = M / m_i$ and $y_i \equiv M_i^{-1} \pmod{m_i}$:

$$X = \left( \sum_{i=1}^k x_i \cdot M_i \cdot y_i \right) \bmod M$$

---

## 6. Quantization Granularities

1. **Per-Tensor**: Single scale and zero-point for the entire tensor. Lowest memory overhead, highest vulnerability to outlier distortion.
2. **Per-Channel (Per-Axis)**: Independent scale per output channel/row ($W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$). Standard for convolution filters and projection weights.
3. **Per-Token (Per-Row)**: Independent scale per activation vector. Essential for transformer self-attention tokens with dynamic magnitude variation.
4. **Grouped / Sub-Channel ($g=32, 128$)**: Independent scale per contiguous block of elements, isolating local spatial variance.

---

## 7. Numerical Invariants and Precision Contracts

- **Zero-Point Invariance**: For affine quantization, $\operatorname{Quantize}(0.0) = z$ exactly as an integer.
- **Symmetric Sign Invariance**: For symmetric quantization, $\operatorname{Quantize}(-x) = -\operatorname{Quantize}(x)$ for all $|x| < x_{\max}$.
- **Transitive Parity**: Parity error must satisfy $\max |x_{\text{Candle}} - x_{\text{PyTorch}}| < 10^{-5}$ across all test tensors.

---

## 8. Hardware Dispatch & Micro-Architecture Realities

- **x86 AVX-512 VNNI / AVX-VNNI**: Native execution of `_mm512_dpbusd_epi32` (dot product of unsigned 8-bit bytes and signed 8-bit bytes with 32-bit accumulation).
- **ARM Neon / Cortex-M**: `SDOT` / `UDOT` 4-way vector multiply-accumulate instructions.
- **WebGPU WGSL**: High-performance parallel integer dot products with workgroup memory caching.
