# Automatic Mixed Precision (AMP), Autocast, and GradScaler

## 1. Mathematical Foundations & Mixed-Precision Lineage

Deep neural network training requires vast computational resources and memory bandwidth. While traditional training utilized 32-bit single-precision floating point (FP32), modern hardware architectures (NVIDIA Tensor Cores, Google TPUs, AMD Matrix Cores) offer $2\times$ to $8\times$ higher throughput and $50\%$ memory bandwidth reduction when executing operations in 16-bit half precision:
1. **Automatic Mixed Precision (AMP, Micikevicius et al. 2017)**: Discovered that deep networks do not require uniform 32-bit precision across all layers. Matrix multiplications and convolutions can safely execute in FP16 or BF16 without degrading final model accuracy, provided reductions, loss functions, and optimizer states remain in FP32.
2. **The Underflow Dilemma in FP16**: The IEEE 754 half-precision format (FP16) allocates only 5 bits to the exponent (bias 15), providing a minimum positive normalized number of $2^{-14} \approx 6.10 \times 10^{-5}$ and subnormal minimum of $2^{-24} \approx 5.96 \times 10^{-8}$. During backpropagation, over $80\%$ of gradient values in deep networks fall below $2^{-14}$, causing catastrophic underflow to zero.
3. **Dynamic Loss Scaling**: Solves gradient underflow by multiplying the scalar training loss by a dynamic scale factor $S$ prior to backpropagation, shifting small gradient values into the representable normal range of FP16.

---

## 2. Autocast Operator Dispatch Policies

`torch.amp.autocast` partitions deep learning operations into four distinct precision categories:
```
  Input Tensors (FP32 or Mixed)
              │
              ▼
┌────────────────────────────────────────────────────────┐
│ Autocast Dispatcher (Whitelist / Blacklist Policy)     │
└────────────────────────────────────────────────────────┘
     │                                      │
     ▼ (Whitelist: Matmul, Conv, Linear)     ▼ (Blacklist: Softmax, LayerNorm, Loss)
┌─────────────────────────┐            ┌─────────────────────────┐
│ Cast to FP16 / BF16     │            │ Elevate to FP32         │
│ (High-throughput GEMM)  │            │ (Numerical Stability)   │
└─────────────────────────┘            └─────────────────────────┘
```

1. **Whitelist (Cast to Half Precision)**: Matrix multiplications (`torch.matmul`, `torch.nn.Linear`), 2D/3D convolutions (`torch.nn.Conv2d`), and linear projections.
2. **Blacklist (Promote to FP32)**: Exponentials, logarithms, divisions, reductions, and normalizations (`torch.softmax`, `torch.layer_norm`, `torch.cross_entropy`), where intermediate values can explode or suffer severe cancellation in FP16.
3. **Fall-Through**: Elementwise arithmetic where output type matches the highest-precision input operand.

---

## 3. Dynamic Loss Scaling (`GradScaler`) Calculus

Let $\mathcal{L}(\theta)$ be the unscaled loss objective and $S \in \mathbb{R}^+$ be the active loss scale factor (typically initialized to $S_0 = 65536.0 = 2^{16}$):

### A. Scaled Forward and Backward Pass
The objective is scaled prior to autodiff:

$$\mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}(\theta)$$

By linearity of differentiation, backward propagation computes scaled gradients:

$$\mathbf{G}_{\text{scaled}} = \nabla_\theta \mathcal{L}_{\text{scaled}} = S \cdot \nabla_\theta \mathcal{L}(\theta)$$

### B. Finite Checking and Unscaling
Before the optimizer updates parameters, gradients are unscaled back to true magnitude:

$$\mathbf{G} = \frac{1}{S} \mathbf{G}_{\text{scaled}}$$

The scaler inspects all parameter gradients for non-finite values (`inf` or `nan`):

$$\text{found\_inf} = \bigvee_{i} \left( \operatorname{isnan}(\mathbf{G}_i) \lor \operatorname{isinf}(\mathbf{G}_i) \right)$$

### C. Dynamic Scale Update Rules
- **If `found_inf` is True**:
  - Skip the optimizer parameter update ($\theta_{t+1} = \theta_t$) to prevent corrupting weights with `NaN`.
  - Decrease the scale factor: $S_{\text{new}} = S \times \text{backoff\_factor}$ (e.g. $S \times 0.5$).
  - Reset consecutive good steps counter: $c = 0$.
- **If `found_inf` is False**:
  - Execute optimizer update: $\theta_{t+1} = \operatorname{OptimizerStep}(\theta_t, \mathbf{G})$.
  - Increment good steps counter: $c \leftarrow c + 1$.
  - If $c \ge \text{growth\_interval}$ (e.g. 2000 steps):
    $$S_{\text{new}} = S \times \text{growth\_factor} \quad (\text{e.g. } S \times 2.0), \quad c = 0$$

---

## 4. Gradient Clipping and Unscaling Ordering

When applying gradient norm clipping ($\|\mathbf{G}\|_2 \le \tau$), unscaling must occur **before** computing the Euclidean norm:

$$\|\mathbf{G}\|_2 = \sqrt{\sum_{i} \|\mathbf{G}_i\|_2^2}$$

$$\text{coeff} = \min\left(1.0, \; \frac{\tau}{\|\mathbf{G}\|_2 + \epsilon}\right)$$

$$\mathbf{G}_{\text{clipped}} = \text{coeff} \cdot \mathbf{G}$$

> [!CAUTION]
> If clipping is mistakenly performed on scaled gradients $\mathbf{G}_{\text{scaled}}$, the clipping coefficient evaluates to $\tau / (S \|\mathbf{G}\|_2)$, artificially suppressing all gradient updates by an extra factor of $1/S$, collapsing learning!

---

## 5. BF16 vs FP16 in AMP

- **FP16 AMP**: Requires `GradScaler` to prevent exponent underflow ($2^{-14}$). Sensitive to scale factor oscillations.
- **BF16 AMP**: Because BF16 shares FP32's 8-bit exponent field ($10^{\pm 38}$), gradient values never underflow normal range. `GradScaler` is **not required** for BF16; `torch.amp.autocast(dtype=torch.bfloat16)` can be used directly without a scaler.

---

## 6. Numerical Invariants and Precision Contracts

- **Transitive Unscaling Invariance**: When gradients are finite, unscaled gradients must match analytical gradients to within machine precision:
  $$\max |\mathbf{G} - \nabla_\theta \mathcal{L}| < 10^{-6}$$
- **Scale Stability Contract**: Scale transitions must strictly follow exact geometric powers of 2 ($S_0 \times 2^k$), avoiding floating-point drift in scale variables.
- **Transitive Parity**: All AMP forward and GradScaler step results must agree to $< 10^{-5}$ across PyTorch, JAX, and Candle.

---

## 7. PyTorch 2.x, JAX, and Candle Architectural Parity

| Feature | PyTorch 2.x | JAX | Rust Candle |
| :--- | :--- | :--- | :--- |
| **Autocast API** | `torch.amp.autocast("cuda")` | `jax.default_matmul_precision` / manual casts | Explicit dtype conversion in graph |
| **Loss Scaler** | `torch.amp.GradScaler()` | `optax.apply_loss_scale` | Custom `dynamic_grad_scaler_step` |
| **Inf / NaN Check** | Internal CUDA kernel reduction | `jnp.any(jnp.isinf(g))` / tree map | Vectorized `is_infinite()` scan |
| **Clip Gradients** | `scaler.unscale_(); clip_grad_norm_()` | `optax.clip_by_global_norm` | Explicit L2 norm scaling |

---

## 8. Production Diagnostics and Failure Modes

1. **Autocast Nesting Hazards**: Placing `optimizer.step()` or `loss.backward()` *inside* the `with torch.amp.autocast():` context causes optimizer states (momentum buffers) to be created in FP16, accelerating rounding stagnation.
2. **Loss Scale Collapsing to Zero**: If model architecture contains un-normalized residuals or unbounded attention logits, gradients repeatedly overflow, causing the scale to halve every step ($S \to 2^{-128} \approx 0$). Diagnosed by logging `scaler.get_scale()`.
3. **Omitting `scaler.unscale_(optimizer)`**: Calling `torch.nn.utils.clip_grad_norm_` without unscaling distorts clipping by factor $S$.
