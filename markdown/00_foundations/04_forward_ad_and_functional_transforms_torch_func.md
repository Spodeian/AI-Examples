# 04. Forward-Mode AD & Functional Transforms (`torch.func`)

## 1. Executive Architectural Brief & Core Invariants

Forward-Mode Automatic Differentiation evaluates Jacobian-vector products (JVPs) simultaneously alongside the primal evaluation via Dual Numbers algebra ($a + b\epsilon$ where $\epsilon^2 = 0$). While Reverse-Mode AD (backpropagation) is optimal for functions with scalar outputs and many inputs ($f: \mathbb{R}^N \to \mathbb{R}$ with complexity $\mathcal{O}(N)$ in reverse vs $\mathcal{O}(1)$ in backprop), Forward-Mode AD is dramatically faster and consumes exponentially less memory for functions with low input dimensions and high output dimensions ($f: \mathbb{R} \to \mathbb{R}^M$), as well as for higher-order curvature (Hessian-vector products via forward-over-reverse $JVP(\nabla f)$). `torch.func` (formerly `functorch`) brings composable, pure functional transforms (`vmap`, `grad`, `jvp`, `vjp`, `hessian`) natively to PyTorch.

```
+-----------------------------------------------------------------------------------+
|                        AUTODIFF PARADIGM TAXONOMY                                 |
|                                                                                   |
|  Wengert List (1964) --> Pearlmutter R-op (1994) --> JAX (2018) --> torch.func    |
|  [Dual Numbers]          [Forward-Over-Reverse]      [Composable]   [Batched AD]  |
|                                                               |                   |
|                                                               v                   |
|                                    +------------------------------------------+   |
|                                    | 04. Forward AD & Functional Transforms   |   |
|                                    +------------------------------------------+   |
|                                         /                  |                 \    |
|                                        v                   v                  v   |
|                              06. PINNs/PDEs       09. DQD/Jacobians   11. DP-SGD  |
+-----------------------------------------------------------------------------------+
```

### Core Invariant Table

| Metric / Invariant | Mathematical Expression | Target Bound / Invariant | Framework Mechanism |
| :--- | :--- | :--- | :--- |
| **JVP Directional Parity** | $\|J(x) v - \text{jvp}(f, x, v)\|_\infty$ | $< 10^{-5}$ | `torch.func.jvp`, `jax.jvp`, `candle` |
| **HVP Exactness** | $\|H(x) v - \text{jvp}(\nabla f, x, v)\|_\infty$ | $< 10^{-5}$ | Forward-over-reverse mixed AD |
| **Per-Sample Gradient Batching** | Shape: $[B, D_{\text{in}}, D_{\text{out}}]$ | Loop-free exact per-sample Jacobians | `torch.func.vmap`, `jax.vmap` |
| **Memory Invariance (Forward AD)** | Memory complexity $\mathcal{O}(1)$ activations | Zero tape retention during forward pass | Dual numbers forward propagation |
| **Transitive Parity** | $\|Y_{\text{torch}} - Y_{\text{golden}}\|_\infty$ | $< 10^{-5}$ | Cross-framework safetensors parity |

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> 1. **Do not use `jacrev` when $D_{\text{in}} \ll D_{\text{out}}$:** Evaluating reverse-mode Jacobians on tall vector-valued functions requires $D_{\text{out}}$ backward passes. Use `jacfwd` instead, which only requires $D_{\text{in}}$ forward passes!
> 2. **Do not mutate stateful buffers inside `vmap`:** In-place mutations on shared buffers (e.g., standard `nn.BatchNorm2d` running mean/variance) inside a vectorized `vmap` call trigger fatal batch dimension expansion errors. Use stateless functional equivalents.
> 3. **Do not materialize full Hessians for large models:** A full Hessian matrix for a model with $P = 10^6$ parameters requires $10^{12}$ elements ($\approx 4\text{TB}$ RAM). Always use matrix-free **Hessian-Vector Products (`HVP`)** via `jvp(grad(f))` which requires only $\mathcal{O}(P)$ memory!

---

## 2. Mathematical Formalism & First-Principles Derivation

### Dual Numbers & Forward-Mode Jacobian-Vector Products (JVP)
Extend real numbers $\mathbb{R}$ to the dual algebra $\mathbb{D} = \{a + b\epsilon \mid a, b \in \mathbb{R}, \epsilon^2 = 0\}$.

Evaluating the Taylor series of an analytical function $f$ at the dual argument $x + v\epsilon$:

$$f(x + v\epsilon) = f(x) + f'(x) v\epsilon + \frac{1}{2} f''(x) v^2 \epsilon^2 + \dots = f(x) + (J(x) v) \epsilon$$

The real component yields the primal output $f(x)$, while the dual component yields the exact directional derivative $J(x) v$ in a **single forward evaluation**, requiring zero storage of intermediate activations!

### Reverse AD vs Forward AD Asymptotic Complexity

| Dimension Regime | Function Mapping | Reverse-Mode AD (VJP / Backprop) | Forward-Mode AD (JVP / Dual Numbers) |
| :--- | :--- | :--- | :--- |
| **Scalar Loss, High-D Inputs** | $f: \mathbb{R}^N \to \mathbb{R}$ | **$\mathcal{O}(1)$ passes** (Optimal for ML training) | $\mathcal{O}(N)$ passes (Requires $N$ tangent sweeps) |
| **High-D Outputs, Low-D Inputs** | $f: \mathbb{R} \to \mathbb{R}^M$ | $\mathcal{O}(M)$ passes (Extremely slow) | **$\mathcal{O}(1)$ passes** (Optimal for trajectories/PINNs) |
| **Directional Curvature (HVP)** | $H v = \nabla^2 f(x) v$ | $\mathcal{O}(N)$ (Requires full Hessian or double backward) | **$\mathcal{O}(1)$ forward pass over gradient** |

### Matrix-Free Hessian-Vector Products (HVP) via Forward-Over-Reverse AD
Let $f: \mathbb{R}^N \to \mathbb{R}$ be a scalar objective. The Hessian is the Jacobian of the gradient: $H(x) = J(\nabla f(x))$.

Rather than materializing the $N \times N$ matrix, the directional curvature along vector $v$ is computed in a single forward pass over the reverse gradient:

$$H(x) v = \operatorname{jvp}(\nabla f, (x,), (v,))$$

---

## 3. Tensor Mechanics & Memory Layout Architecture

```
Forward AD Dual Memory Pipeline:
+--------------------------------------------------------------+
| Memory Flow: Primal and Tangent Pass in Parallel             |
|                                                              |
| Primal:   x  [B, Din]  ---( @ W + b )--->  y  [B, Dout]     |
| Tangent:  v  [B, Din]  ---(   @ W   )---> dy  [B, Dout]     |
|                                                              |
| Memory Footprint: O(1) extra activations stored on stack      |
+--------------------------------------------------------------+
```

### Stride-1 Reduction Alignment
In per-sample gradient calculations via `vmap`, tensors maintain batch-first contiguous ordering $[B, D_{\text{in}}, D_{\text{out}}]$. The parameter update reduction reduces along the batch dimension ($0$) while leaving the parameter matrix strides aligned with native contiguous memory blocks.

---

## 4. Cross-Framework Comparative Reference

| Operation | PyTorch 2.x Track (`pytorch2/`) | JAX Track (`jax/`) | Rust Candle Track (`candle/`) | Fused Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Forward JVP** | `func.jvp(fn, (x,), (v,))` | `jax.jvp(fn, (x,), (v,))` | `ForwardAdJvp::linear_jvp(...)` | `fused_forward_jvp_cpu(...)` |
| **Reverse VJP** | `func.vjp(fn, x)` | `jax.vjp(fn, x)` | Analytical VJP pass | Custom backward kernel |
| **Per-Sample Grads** | `func.vmap(func.grad(loss))` | `jax.vmap(jax.grad(loss))` | Batched tensor outer product | Vectorized batch outer product |
| **Hessian-Vector** | `func.jvp(func.grad(f), (x,), (v,))` | `jax.jvp(jax.grad(f), (x,), (v,))` | Linear operator $W W^\top v$ | Matrix-free projection |
| **Compilation Hook**| `torch.compile(fullgraph=True)` | `jax.jit(jvp)` | Native LLVM auto-vectorization | Cache-tiled SIMD loops |

---

## 5. Hardware & Compilation Dynamics

### Arithmetic Intensity & Roofline Characterization
- **Primal Forward:** $2 N D_{\text{in}} D_{\text{out}}$ FLOPs.
- **Tangent Forward (JVP):** $2 N D_{\text{in}} D_{\text{out}}$ FLOPs.
- **Combined Memory Traffic:** Reads $X, V, W, b$, writes $Y, dY$.
- **Roofline Metric:** Arithmetic intensity $I_{\text{op}} = \frac{4 N D_{\text{in}} D_{\text{out}}}{\text{Memory Bytes}} \approx 1.0\text{--}8.0\text{ FLOP/byte}$, demonstrating that fusing primal and tangent evaluations in a single pass maximizes memory cache locality (L1/L2 hits) by avoiding double DRAM reads of the weight matrix $W$.

---

## 6. Failure Modes, Anti-Patterns & Diagnostic Guide

| ID | Failure Mode | Root Cause | Observable Symptom | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **FAT-01** | Catastrophic Memory Exhaustion from Reverse AD | Calling `jacrev` on a vector function where $D_{\text{out}} \gg D_{\text{in}}$ | Out-Of-Memory (`OOM`) or extreme latency | Swap to `jacfwd` which scales with $D_{\text{in}}$ |
| **FAT-02** | Stateful Buffer Mutation Crash | Invoking `vmap` on layers mutating `running_mean` (e.g. BatchNorm) | `RuntimeError: Batching rule not implemented` | Use functional parameter calls or LayerNorm/RMSNorm |
| **FAT-03** | Hessian RAM Explosion | Calling `hessian(f)(x)` on high-dimensional weights | Allocation failure of $P \times P$ elements | Always use matrix-free **HVP** via `jvp(grad(f))` |

---

## 7. Quantitative Verification & Parity Assertions

```python
# 1. Forward AD Directional Derivative Parity against Golden Output
primals_out, jvp_out = func.jvp(linear_fn, (test_in,), (tangent_in,))
assert torch.norm(primals_out - test_out, p=float("inf")) < 1e-5
assert torch.norm(jvp_out - golden_tangent_out, p=float("inf")) < 1e-5

# 2. HVP vs Exact Full Hessian
h_full = func.hessian(scalar_loss)(x)
hvp = func.jvp(func.grad(scalar_loss), (x,), (v,))[1]
assert torch.norm(hvp - h_full @ v, p=float("inf")) < 1e-5
```

---

## 8. Relative Navigation & Synchronized Track Artifacts

- **Master Curriculum Index:** [README.md](../../README.md)
- **Goal-Oriented Index:** [INDEX_BY_GOAL.md](../../INDEX_BY_GOAL.md)
- **Interactive Notebook Explorer:** [04_forward_ad_and_functional_transforms_torch_func.ipynb](../../notebooks/00_foundations/04_forward_ad_and_functional_transforms_torch_func.ipynb)
- **Production PyTorch 2.x Module:** [04_forward_ad_and_functional_transforms_torch_func.py](../../pytorch2/00_foundations/04_forward_ad_and_functional_transforms_torch_func.py)
- **Functional JAX Module:** [04_forward_ad_and_functional_transforms_torch_func.py](../../jax/00_foundations/04_forward_ad_and_functional_transforms_torch_func.py)
- **Native Rust Candle Crate:** [04_forward_ad_and_functional_transforms_torch_func.rs](../../candle/00_foundations/04_forward_ad_and_functional_transforms_torch_func.rs)
- **Hardware CPU Kernel:** [04_forward_ad_and_functional_transforms_torch_func_cpu.py](../../kernels/00_foundations/04_forward_ad_and_functional_transforms_torch_func_cpu.py)
- **Golden Reference Checkpoint:** [04_forward_ad_and_functional_transforms_torch_func.safetensors](../../weights/00_foundations/04_forward_ad_and_functional_transforms_torch_func.safetensors)
- **Previous Topic:** [03. Autograd Mechanics & Complex Wirtinger](03_autograd_mechanics_complex_wirtinger_and_custom_functions.md)
- **Next Topic:** 05. Hooks & Saved Tensor Hooks (Upcoming in Category 00)
