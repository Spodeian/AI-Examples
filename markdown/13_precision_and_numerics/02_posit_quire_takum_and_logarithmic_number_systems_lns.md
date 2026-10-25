# Posit, Quire, Takum, and Logarithmic Number Systems (LNS)

## 1. Mathematical Foundations & Non-IEEE Number Systems

While IEEE 754 floating-point arithmetic has dominated digital computing for four decades, its static bit allocation (fixed exponent and mantissa widths) introduces fundamental inefficiencies for deep learning and scientific computing:
1. **Posit™ Arithmetic (Gustafson 2017)**: Introduced as part of the Type III Universal Number (unum) framework. Posits replace the static exponent bitfield with a dynamic, run-length-encoded **Regime**, allowing the number of mantissa bits to vary smoothly according to the magnitude of the number.
2. **The Kulisch Quire (Kulisch 1981)**: Resolves catastrophic cancellation and rounding accumulation in dot products $\mathbf{a}^T \mathbf{b} = \sum_{i=1}^M a_i b_i$ by accumulating all product bits into an exact fixed-point accumulator register, deferring rounding until the final scalar output.
3. **Logarithmic Number Systems (LNS, Kingsbury & Rayner 1971)**: Represents continuous numbers by their base-2 logarithm: $x = (-1)^s 2^L$. By operating in the logarithmic domain, hardware multipliers are completely eliminated, replaced by simple integer adders.

---

## 2. Posit™ Bit Anatomy & Tapered Precision

A posit is parameterized by total word length $N$ and maximum exponent bits $es$: $\text{Posit}\langle N, es \rangle$.
```
  1-bit      Variable length (k)      es bits     Remaining bits (frac)
┌───────┬───────────────────────────┬───────────┬───────────────────────┐
│ Sign  │   Regime Bits (r...r ~r)  │ Exponent  │       Fraction        │
└───────┴───────────────────────────┴───────────┴───────────────────────┘
```

1. **Useed**: The base scaling factor governed by $es$:
   $$\text{useed} = 2^{2^{es}}$$
2. **Regime Value ($k$)**: Encoded by a run of identical bits terminated by an opposite bit (or by the end of the word):
   - A run of $m$ ones (`11...10`) represents $k = m - 1 \ge 0$.
   - A run of $m$ zeros (`00...01`) represents $k = -m < 0$.
3. **Decoded Posit Value**:
   $$x = (-1)^s \times \text{useed}^k \times 2^e \times (1 + f)$$
4. **The "Golden Zone"**: When numbers cluster near $\pm 1$ ($k \approx 0$), the regime requires only 2 bits (`10` or `01`), leaving the maximum possible number of bits for the fraction. For $\text{Posit}\langle 8, 1 \rangle$, numbers in $[0.5, 2.0]$ possess 4 fraction bits—yielding **3–5 dB higher SQNR than FP8 (E5M2 or E4M3)** on Gaussian-distributed neural weights!

---

## 3. The Kulisch Quire Exact Accumulator

In classical IEEE floating-point accumulation, evaluating $\sum_{i=1}^{10000} x_i y_i$ accumulates rounding error $\mathcal{O}(\sqrt{M} \epsilon_{\text{mach}})$, frequently suffering from catastrophic cancellation when large opposing terms cancel out:
1. **The Quire Register**: A dedicated 512-bit or 2048-bit fixed-point accumulator spanning the entire dynamic range from $(\text{useed}^{-(N-2)})^2$ to $(\text{useed}^{N-2})^2$, plus extra carry-guard bits $\lceil \log_2(M) \rceil$.
2. **Exact Dot-Product**:
   $$Q = \sum_{i=1}^M a_i \times b_i \quad (\text{Exact in Fixed-Point})$$
3. A single rounding operation is performed only when converting the final Quire value $Q$ back to a target Posit or IEEE float, providing provably deterministic mathematical exactness across all matrix operations.

---

## 4. Logarithmic Number Systems (LNS)

An LNS number represents non-zero real $x$ via a sign bit $s_x \in \{0, 1\}$ and a fixed-point logarithm $L_x = \log_2(|x|)$:

### A. Exact Multiplication and Division
Multiplication and division require zero hardware multipliers:

$$x \cdot y \iff s_{\text{mul}} = s_x \oplus s_y, \quad L_{\text{mul}} = L_x + L_y$$

$$x / y \iff s_{\text{div}} = s_x \oplus s_y, \quad L_{\text{div}} = L_x - L_y$$

### B. Gaussian Log-Addition
For positive quantities $a, b > 0$ with $L_a \ge L_b$:

$$a + b = 2^{L_a} + 2^{L_b} = 2^{L_a} \left(1 + 2^{-(L_a - L_b)}\right)$$

$$L_{a+b} = \log_2(a + b) = L_a + \Phi_+(d), \quad \text{where } d = L_a - L_b \ge 0$$

The non-linear correction function $\Phi_+(d) = \log_2(1 + 2^{-d})$ is computed via a compact lookup table (LUT) or piecewise polynomial approximation.

---

## 5. Mitchell's Approximation and Linear Interpolation

To avoid expensive transcendental evaluations, Mitchell (1962) proposed approximating log-addition by piecewise linear interpolation:
$$\log_2(1 + 2^{-d}) \approx \begin{cases} 1 - d & \text{if } 0 \le d < 1 \\ 0 & \text{if } d \ge 1 \end{cases}$$
While efficient in FPGA logic, uncorrected Mitchell's approximation incurs a systematic positive bias of up to $+0.086\text{ bits}$, which accumulates destructively across deep neural layers unless compensated by correction LUTs.

---

## 6. Numerical Invariants and Precision Contracts

- **Posit Tapering Invariant**: The number of available fraction bits $f(x)$ decreases monotonically with $|k|$:
  $$f(x) = \max\left(0, \; N - 1 - r(k) - es\right)$$
- **LNS Multiplicative Exactness**: For non-zero finite values, LNS multiplication satisfies:
  $$\max |(a \cdot b)_{\text{LNS}} - (a \cdot b)_{\text{exact}}| < 10^{-6}$$
- **Transitive Parity**: All Posit, Quire, and LNS kernels maintain parity $< 10^{-5}$ across PyTorch, JAX, and Candle.

---

## 7. PyTorch 2.x, JAX, and Candle Architectural Parity

| Feature | PyTorch 2.x | JAX | Rust Candle |
| :--- | :--- | :--- | :--- |
| **Posit Simulation** | Vectorized bitfield tensor simulation | Pure functional `jnp` math | Bit-accurate Rust structs |
| **Quire Accumulator** | `torch.float64` high-precision fallback | `jnp.float64` / XLA accumulator | Pure `f64` exact sum |
| **LNS Multiplier** | Native `torch.log2` / `exp2` | `jnp.log2` / `jnp.exp2` | Native Rust `log2` / `powf` loops |
| **Hardware Target** | Emulated / FPGA accelerator bindings | JAX custom primitives | Native SIMD / Wasm targets |

---

## 8. Production Diagnostics and Failure Modes

1. **Posit Exponent Scale Selection**: Setting $es$ too high ($es=3$ in an 8-bit posit) wastes bits on astronomical dynamic range ($10^{\pm 38}$), eroding fraction bits in the Golden Zone; setting $es=0$ restricts dynamic range to $10^{\pm 4}$, risking overflow on unnormalized activations.
2. **Log-Subtractions Near Zero**: In LNS, computing $a - b$ when $a \approx b$ requires evaluating $\Phi_-(d) = \log_2(1 - 2^{-d})$. As $d \to 0$, $\Phi_-(d) \to -\infty$, demanding very deep negative log representation to avoid underflow.
3. **Quire Memory Footprint**: Allocating permanent 512-bit registers for every intermediate activation exhausts register file capacity. Quires must be used strictly as localized thread-level accumulators during GEMM reduction loops.
