# Stochastic Rounding and Master-Weight-Free BF16 Training

## 1. Mathematical Foundations & The Vanishing Update Dilemma

In modern deep neural network training, weights and optimizer states dominate GPU High-Bandwidth Memory (HBM). In standard mixed-precision training:
1. **The FP32 Master Weight Burden**: While forward activations and gradients execute in 16-bit half precision (BF16 or FP16), standard training frameworks maintain a persistent 32-bit single precision (FP32) master copy of every trainable parameter:
   $$\theta_{t+1}^{\text{FP32}} = \theta_t^{\text{FP32}} - \eta \cdot g_t$$
   $$\theta_{t+1}^{\text{BF16}} = \operatorname{Cast}_{\text{BF16}}(\theta_{t+1}^{\text{FP32}})$$
   This master weight copy consumes $2\times$ parameter memory ($4\text{ bytes/weight}$) and imposes additional memory bandwidth traffic on every optimizer step.
2. **The Vanishing Update Trap (Round-to-Nearest Stalling)**:
   Why not update BF16 weights directly ($\theta_{t+1}^{\text{BF16}} = \theta_t^{\text{BF16}} - \eta g_t$)?
   The Bfloat16 format allocates only 7 bits to the mantissa, providing a machine epsilon / Unit in the Last Place:
   $$\text{ULP}(W) = 2^{\lfloor \log_2(|W|) \rfloor - 7}$$
   For a typical weight $W \approx 1.0$, $\text{ULP}(W) = 2^{-7} \approx 0.0078125$.
   During SGD or AdamW training, per-step weight updates $\Delta W = -\eta g_t$ typically fall in the range $10^{-5}$ to $10^{-4}$.
   Under standard Round-to-Nearest-Even (RTNE):
   $$|\Delta W| < \frac{1}{2} \text{ULP}(W) \implies \operatorname{RTNE}(W + \Delta W) = W$$
   Every single weight update rounds down to zero! The model stalls completely and never trains.

---

## 2. Stochastic Rounding (SR) Calculus & Unbiased Expectation Proof

Stochastic Rounding (SR, Barnes et al., 1951; Gupta et al., 2015) resolves the vanishing update problem by mapping real values probabilistically to adjacent discrete grid points:

### Mathematical Definition
Let $x \in \mathbb{R}$ fall between two representable floating point numbers $x_L = \lfloor x \rfloor_{\text{ULP}}$ and $x_R = x_L + \text{ULP}$:

$$\operatorname{SR}(x) = \begin{cases} x_R, & \text{with probability } P = \frac{x - x_L}{x_R - x_L} = \frac{x - x_L}{\text{ULP}} \\ x_L, & \text{with probability } 1 - P = \frac{x_R - x}{\text{ULP}} \end{cases}$$

### Proof of Unbiased Expectation
The expected value of the stochastically rounded result is identically equal to the exact continuous value:

$$\mathbb{E}[\operatorname{SR}(x)] = x_R \cdot \left(\frac{x - x_L}{x_R - x_L}\right) + x_L \cdot \left(\frac{x_R - x}{x_R - x_L}\right)$$

$$= \frac{x_R x - x_R x_L + x_L x_R - x_L x}{x_R - x_L} = \frac{x(x_R - x_L)}{x_R - x_L} = x \quad \blacksquare$$

### Cumulative Drift Invariant
Over $T$ optimization iterations, the accumulated weight change converges to the true analytical gradient integral:

$$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \operatorname{SR}(W_t + \Delta W_t) = W_0 + \sum_{t=1}^T \Delta W_t$$

Even when $\Delta W \approx 10^{-6} \ll \text{ULP}$, the update flips the least significant bit with probability $P \approx 10^{-4}$, accumulating the correct drift over training.

---

## 3. Master-Weight-Free Training Architecture

By replacing RTNE with Stochastic Rounding on the weight update step, the FP32 master copy is eliminated:

```
Standard Mixed Precision (Requires FP32 Master):
  [Grads (BF16)] ──► [Optimizer Step] ──► [Master Weights (FP32)] ──► Cast ──► [Weights (BF16)]
                                                ▲ (Consumes 4 bytes/param)

Master-Weight-Free BF16 Training:
  [Grads (BF16)] ──► [Optimizer Step] ──► [Stochastic Rounding (SR)] ──► [Weights (BF16)]
                                                ▲ (Consumes 2 bytes/param: 50% savings!)
```

---

## 4. Bitwise Floating Point PRNG Injection (Hardware Silicon Implementation)

In silicon accelerators (Graphcore IPU, Google TPU v4/v5, Cerebras WSE), Stochastic Rounding is implemented with zero transcendental arithmetic:
1. In IEEE FP32, a float has 23 mantissa bits; BF16 retains the top 7 mantissa bits.
2. The remaining 16 bits represent the fractional distance $\frac{x - x_L}{\text{ULP}} \in [0, 1)$ scaled by $2^{16} = 65536$.
3. A hardware Linear Feedback Shift Register (LFSR) or Xorshift PRNG generates a 16-bit pseudo-random unsigned integer $r \in [0, 65535]$.
4. The integer is added to the FP32 word:
   $$u_{\text{rounded}} = u_{\text{FP32}} + r$$
   Truncating $u_{\text{rounded}}$ to 16 bits implements exact Stochastic Rounding!

---

## 5. Kahan Compensated Summation Alternative

For hardware targets lacking fast PRNG engines, **Kahan Summation** offers an analytical alternative by tracking low-order residual rounding error:

$$y = \Delta W - c_t$$

$$t = W_t + y$$

$$c_{t+1} = (t - W_t) - y$$

$$W_{t+1} = t$$

The compensation variable $c_t$ recovers the lost fractional bits in the subsequent iteration without requiring stochastic noise.

---

## 6. Asymptotic Time & Memory Complexity

- **Time Complexity**: $\mathcal{O}(N)$ elementwise operations per optimizer step (identical asymptotic complexity to standard SGD/AdamW).
- **Memory Footprint**:
  - Standard BF16 Training: $2\text{ bytes (BF16)} + 4\text{ bytes (FP32 master)} = 6\text{ bytes/weight}$.
  - Master-Weight-Free BF16: $2\text{ bytes/weight}$ ($3\times$ memory reduction for model weights).

---

## 7. Numerical Invariants and Precision Contracts

- **Mean Parity Invariant**: $\mathbb{E}[\operatorname{SR}(x)] = x$ for all $x \in \mathbb{R}$.
- **Zero Invariance**: $\operatorname{SR}(0.0) = 0.0$ deterministically.
- **Transitive Parity**: All deterministic and stochastic assertions must agree to $< 10^{-5}$ across PyTorch, JAX, and Candle.

---

## 8. Hardware Dispatch & Micro-Architecture Realities

- **Google TPU (v4, v5e, v5p)**: Native hardware support for stochastic rounding in XLA via `xla_gpu_enable_stochastic_rounding`.
- **Graphcore IPU**: Native hardware instructions `STOCHASTIC_ROUND`.
- **CUDA / Tensor Cores**: Emulated via cuRAND noise injection or FP32 bitwise integer add before `__float2bfloat16`.
