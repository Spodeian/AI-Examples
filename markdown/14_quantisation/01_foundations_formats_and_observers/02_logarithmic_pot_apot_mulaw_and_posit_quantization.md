# Logarithmic, Power-of-Two (PoT), Additive PoT (APoT), μ-Law, and Posit Quantization

## 1. Mathematical Foundations & Historical Lineage

Traditional uniform quantization places quantization levels at equidistant intervals, assuming a uniform probability density function (PDF). However, neural network parameters and audio/speech activations follow non-uniform, heavy-tailed bell distributions:
1. **Companding (ITU-T G.711 $\mu$-Law / A-Law, 1972)**: Non-linearly warps speech amplitudes prior to uniform quantization, equalizing Signal-to-Quantization-Noise Ratio (SQNR) across quiet whispers and loud bursts.
2. **Power-of-Two Quantization (LogNet / PoT, Miyashita et al., 2016)**: Constrains all weights to exact powers of two ($w = \pm 2^{-p}$), replacing hardware multipliers with single-cycle barrel bit-shifters ($x \gg p$).
3. **Additive Power-of-Two (APoT, Li et al., ICLR 2020)**: Solves PoT's rigid resolution dilemma near $\pm 1.0$ by representing weights as the sum of $n=2$ staggered power-of-two terms ($w = \pm (2^{-p_1} + 2^{-p_2})$), beating single-term PoT by $>6\text{ dB SQNR}$ at 4 bits.
4. **Posit Arithmetic (Gustafson, 2017)**: Tapers precision away from the "Golden Zone" ($\pm 1$), providing dynamic bit allocation without explicit exponent scales.

---

## 2. Power-of-Two (PoT) Multiplier-Free Mechanics

In single-term PoT quantization, every weight $w$ is represented by a sign and an integer exponent $p \in \{0, 1, \dots, 2^{b-1} - 2\}$:

$$w \in \{0\} \cup \left\{ \operatorname{sign}(w) \cdot 2^{-p} \right\}$$

### Hardware Shift Inference
During the forward matrix multiplication $y = \mathbf{x} \cdot \mathbf{W}$, multiplication by $2^{-p}$ is evaluated as an arithmetic right-shift:

$$x_i \cdot w_i = \operatorname{sign}(w_i) \cdot (x_i \gg p_i)$$

Eliminates digital signal processors (DSPs) and integer MACs entirely, drastically reducing silicon area and dynamic power consumption on FPGAs and ASICs.

---

## 3. Additive Power-of-Two (APoT) Staggered Resolution

Single-term PoT allocates $75\%$ of its levels below $2^{-3} = 0.125$, leaving only $\{0.25, 0.5, 1.0\}$ to cover the vital $[0.2, 1.0]$ range where normalized neural weights cluster.

**APoT** solves this by summing $n$ power-of-two terms:

$$w = \operatorname{sign}(w) \sum_{i=1}^n 2^{-p_i}, \quad p_i \in \mathcal{P}_i$$

For 4-bit APoT with $n=2$ terms, the codebook forms 8 non-negative levels:

$$\mathcal{C} = \{0.0, \; 0.125, \; 0.25, \; 0.375, \; 0.5, \; 0.625, \; 0.75, \; 1.0\}$$

### Hardware Execution
Multiplication evaluates as a single shift-and-add operation:

$$x \cdot w = \operatorname{sign}(w) \cdot \left( (x \gg p_1) + (x \gg p_2) \right)$$

Preserves multiplier-free hardware while matching standard uniform INT4 task accuracy.

---

## 4. Base-$\sqrt{2}$ Logarithmic Scaling

Standard base-2 logarithmic quantization places levels at octave boundaries ($1, 1/2, 1/4, 1/8$), which is too coarse for sensitive transformer attention heads.

**Base-$\sqrt{2}$ Logarithmic Quantization** provides 2 quantization bins per octave:

$$q = \operatorname{sign}(x) \cdot (\sqrt{2})^k = \operatorname{sign}(x) \cdot 2^{k/2}, \quad k \in \mathbb{Z}$$

In hardware, $k = 2m + r$ where $r \in \{0, 1\}$:
- If $r = 0$: $x \gg m$.
- If $r = 1$: $(x \gg m) \times \sqrt{2} \approx (x \gg m) + (x \gg (m+1)) - (x \gg (m+4))$.

---

## 5. $\mu$-Law and A-Law Companding (WaveNet)

The continuous logarithmic companding transformation maps $x \in [-1, 1]$ into $y \in [-1, 1]$:

$$F(x) = \operatorname{sign}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \quad \mu = 255$$

```
Linear Domain x [-1, 1] ──► [ Compand: F(x) ] ──► Uniform INT8 Quantizer
                                                         │
De-quantized Output     ◄── [ Expand: F^(-1)(y) ] ◄──────┘
```

The analytical decompanding expansion inverse is:

$$F^{-1}(y) = \operatorname{sign}(y) \frac{(1 + \mu)^{|y|} - 1}{\mu}$$

Provides constant relative quantization noise $\Delta x / |x| \approx \text{const}$ across several decades of dynamic range.

---

## 6. Post-Training Posit4 & Posit8 Quantization

Posit numbers employ a variable-length regime prefix ($k$) that dynamically scales precision:
- For weights near $\pm 1$, the regime length is minimal ($r=1$), maximizing fraction bits ($f = N - 1 - r - es$).
- In Posit4 ($N=4, es=1$), levels span:
  $$\{0.0, \; 0.25, \; 0.5, \; 0.75, \; 1.0, \; 1.5, \; 2.0, \; 4.0\}$$
- Yields higher SQNR than uniform INT4 for Gaussian-distributed neural weights.

---

## 7. Numerical Invariants and Precision Contracts

- **Zero Preservation**: $\operatorname{Quantize}(0.0) = 0.0$ across all logarithmic and power-of-two schemes.
- **Symmetric Sign Equivariance**: $\operatorname{Quantize}(-x) = -\operatorname{Quantize}(x)$ for all inputs.
- **Transitive Parity**: Parity error must satisfy $\max |x_{\text{Candle}} - x_{\text{PyTorch}}| < 10^{-5}$ across all test tensors.

---

## 8. Hardware Dispatch & Micro-Architecture Realities

- **FPGA / Custom ASIC**: Bit-shift and small adders execute in $< 0.5\text{ ns}$ with $< 100\text{ gates}$, versus $> 2000\text{ gates}$ for a full 32-bit multiplier.
- **SIMD / GPU**: Emulated via table lookups or bitwise exponent manipulation.
