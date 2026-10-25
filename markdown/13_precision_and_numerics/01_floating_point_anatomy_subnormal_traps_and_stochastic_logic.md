# Floating Point Anatomy, Subnormal Traps, and Stochastic Logic

## 1. Mathematical Foundations & Floating-Point Lineage

Digital computers represent continuous real numbers using discrete, finite-precision floating-point formats governed by the IEEE 754 standard (IEEE 1985, revised 2019):
1. **The IEEE 754 Real Number Representation**: Any finite non-zero floating-point number is factored into three bitfields: a sign bit $s \in \{0, 1\}$, an exponent bitfield $E$ encoded with an integer bias $B$, and a fractional mantissa (significand) $M \in [0, 1)$:
   $$x = (-1)^s \times 2^{E - B} \times (1 + M)$$
   The leading 1 is implicit (the "hidden bit"), providing an extra bit of precision for normalized numbers.
2. **Subnormal / Denormal Numbers**: When $E = 0$, the implicit leading bit becomes 0:
   $$x_{\text{sub}} = (-1)^s \times 2^{1 - B} \times (0 + M)$$
   Subnormals fill the gap between zero and the smallest normalized float ($x_{\min} = 2^{1 - B}$), enabling gradual underflow.
3. **Stochastic Computing (Gaines 1969)**: Represents continuous numbers as probabilities $p = P(X = 1) \in [0, 1]$ in random Bernoulli bitstreams. Rather than using multi-bit binary multipliers with complex carry propagation, arithmetic operations map directly to basic logic gates: multiplication becomes a single two-input bitwise `AND` gate.

---

## 2. Bit Anatomy Across Deep Learning Formats

| Format | Total Bits | Sign Bits | Exponent Bits ($E$) | Exponent Bias ($B$) | Mantissa Bits ($M$) | Dynamic Range | Machine Epsilon ($\epsilon_{\text{mach}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP64** (Double) | 64 | 1 | 11 | 1023 | 52 | $10^{\pm 308}$ | $2^{-52} \approx 2.22 \times 10^{-16}$ |
| **FP32** (Single) | 32 | 1 | 8 | 127 | 23 | $10^{\pm 38}$ | $2^{-23} \approx 1.19 \times 10^{-7}$ |
| **TF32** (TensorFloat) | 19 | 1 | 8 | 127 | 10 | $10^{\pm 38}$ | $2^{-10} \approx 9.77 \times 10^{-4}$ |
| **FP16** (Half) | 16 | 1 | 5 | 15 | 10 | $10^{\pm 5}$ (Max 65,504) | $2^{-10} \approx 9.77 \times 10^{-4}$ |
| **BF16** (Brain Float) | 16 | 1 | 8 | 127 | 7 | $10^{\pm 38}$ | $2^{-7} \approx 7.81 \times 10^{-3}$ |

- **BF16 vs FP16 Trade-off**: BF16 preserves the full 8-bit dynamic range of FP32 ($10^{\pm 38}$), eliminating numerical underflow/overflow during deep network training without requiring loss scalers. However, its 7-bit mantissa offers lower precision ($\epsilon_{\text{mach}} \approx 7.8 \times 10^{-3}$) than FP16 ($\approx 9.8 \times 10^{-4}$).

---

## 3. Subnormal Traps: Flush-to-Zero (FTZ) and Denormals-are-Zero (DAZ)

Standard CPU and GPU floating-point units (ALUs) are optimized for normalized numbers with implicit leading ones.
1. **The Microcode Trap**: When an operation encounters subnormal inputs ($E = 0$) or produces a subnormal result, modern hardware pipelines cannot handle the shifted significand in single-cycle execution. The CPU/GPU triggers an interrupt and falls back to microcode exception handling, causing up to a **$100\times$ throughput penalty**.
2. **FTZ (Flush-To-Zero)**: Hardware flag causing any underflowing computation that would produce a subnormal result to be flushed directly to signed zero ($0.0$).
3. **DAZ (Denormals-Are-Zero)**: Hardware flag causing any subnormal input operand to be treated as zero before entering the execution unit.
4. Enabling FTZ/DAZ in PyTorch:
   ```python
   torch.set_flush_denormal(True)
   ```

---

## 4. Stochastic Computing & Bitstream Logic

Stochastic Computing (SC) encodes real values as time-domain or space-domain bitstreams:
1. **Unipolar Format ($x \in [0, 1]$)**: A real number $p$ is represented by a sequence of $L$ binary digits $B = (b_1, b_2, \dots, b_L)$ where each bit is generated from a uniform random source $U_k \sim \mathcal{U}[0, 1]$:
   $$b_k = \begin{cases} 1 & \text{if } U_k < p \\ 0 & \text{otherwise} \end{cases}, \quad \hat{p} = \frac{1}{L} \sum_{k=1}^L b_k$$
2. **Bipolar Format ($x \in [-1, +1]$)**: Maps symmetric signed numbers into probability space:
   $$p = \frac{x + 1}{2} \in [0, 1], \quad \hat{x} = 2 \hat{p} - 1 = \frac{2}{L} \sum_{k=1}^L b_k - 1$$

---

## 5. Stochastic Arithmetic Primitives

### A. Unipolar Multiplication (Single `AND` Gate)
Let $B_1$ and $B_2$ be two independent stochastic bitstreams encoding probabilities $p_1$ and $p_2$. Passing them through a digital AND gate yields:

$$P(B_1 \land B_2 = 1) = P(B_1 = 1) \cdot P(B_2 = 1) = p_1 \cdot p_2$$

$$\hat{p}_{\text{mul}} = \frac{1}{L} \sum_{k=1}^L (b_{1, k} \land b_{2, k})$$

A complex 32-bit floating-point multiplier requiring thousands of transistors collapses into a single 2-transistor logic gate!

### B. Bipolar Multiplication (Single `XNOR` Gate)
For signed numbers $x_1, x_2 \in [-1, +1]$ with corresponding probabilities $p_1, p_2$:

$$b_{\text{xnor}, k} = b_{1, k} \odot b_{2, k} = b_{1, k} b_{2, k} + (1 - b_{1, k})(1 - b_{2, k})$$

$$\hat{x} = 2 \left( \frac{1}{L} \sum_{k=1}^L b_{\text{xnor}, k} \right) - 1 = x_1 \cdot x_2$$

---

## 6. Numerical Invariants and Precision Contracts

- **Variance of Stochastic Estimation**: The variance of an $L$-bit stochastic bitstream estimating probability $p$ is:
  $$\operatorname{Var}(\hat{p}) = \frac{p(1 - p)}{L} \le \frac{1}{4L}$$
  The expected estimation error scales as $\mathcal{O}(1/\sqrt{L})$.
- **BF16 Rounding Parity**: Round-to-nearest-even (RTNE) conversion from FP32 to BF16 satisfies:
  $$|x - \text{BF16}(x)| \le \frac{1}{2} \text{ULP}(x)$$
- **Transitive Parity**: Bit-accurate FTZ and stochastic multipliers must agree to within $< 10^{-5}$ across PyTorch, JAX, and Candle.

---

## 7. PyTorch 2.x, JAX, and Candle Architectural Parity

| Feature | PyTorch 2.x | JAX | Rust Candle |
| :--- | :--- | :--- | :--- |
| **BF16 Support** | `torch.bfloat16` | `jnp.bfloat16` | `candle_core::DType::BF16` |
| **Denormal Control** | `torch.set_flush_denormal(True)` | Compiler flags (`-ffast-math`) | Standard CPU/LLVM flags / `ftz_daz` |
| **Bitwise Simulation** | Bitwise tensor ops on `torch.uint8` | `jnp.bitwise_and` / `jnp.bitwise_xor` | Native bitwise integer ops |
| **Stochastic Logic** | PRNG bitstream generation | `jax.random.bernoulli` | Seeded pseudo-random bitstreams |

---

## 8. Production Diagnostics and Numerical Failure Modes

1. **Catastrophic Cancellation**: Subtracting two nearly equal floating-point numbers ($a - b$ where $a \approx b$) causes severe loss of precision, exposing low-order rounding noise.
2. **BF16 Gradient Stagnation**: Adding small gradient updates $\eta g_t$ to large weights $w$ in BF16 when $|\eta g_t| < \frac{1}{2}\text{ULP}(w)$ results in $w + \eta g_t = w$, completely halting learning. Solved by Stochastic Rounding or FP32 master weights.
3. **Unscaled FP16 Attention Overflow**: Computing $Q K^T$ without $\frac{1}{\sqrt{d_k}}$ scaling frequently exceeds 65,504 in FP16, resulting in $\infty$ and subsequent $\text{NaN}$ outputs during Softmax.
