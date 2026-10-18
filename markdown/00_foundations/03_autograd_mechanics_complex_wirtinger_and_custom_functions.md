# 03. Autograd Mechanics, Complex Wirtinger & Custom Functions

## 1. Executive Architectural Brief & Core Invariants

Automatic Differentiation (AD) is the computational engine of modern deep learning. Unlike symbolic differentiation (which suffers from expression swell) or numerical finite differences (which incurs $O(D)$ evaluations and truncation errors), reverse-mode AD computes exact vector-Jacobian products (VJPs) in a single backward pass with runtime proportional to the forward pass. Understanding tape internals, residual lifetime management, and complex Wirtinger calculus enables authoring memory-efficient custom operators and differentiating complex-valued physical models (e.g. signal processing, quantum states, Fourier neural operators).

```
+-----------------------------------------------------------------------------------+
|                        AUTOGRAD & WIRTINGER TAXONOMY                              |
|                                                                                   |
|  Wirtinger (1927)    -->  Linnainmaa (1970)      -->  Rumelhart et al. (1986)     |
|  [Complex Calculus]       [Reverse-Mode AD]           [Backpropagation / Tape]    |
|                                                               |                   |
|                                                               v                   |
|                                    +------------------------------------------+   |
|                                    | 03. Autograd Mechanics & Custom Functions|   |
|                                    +------------------------------------------+   |
|                                         /                  |                 \    |
|                                        v                   v                  v   |
|                              04. Forward AD/functorch  05. Hooks      06. FNO/PDEs|
+-----------------------------------------------------------------------------------+
```

### Core Invariant Table

| Metric / Invariant | Mathematical Expression | Target Bound / Invariant | Framework Mechanism |
| :--- | :--- | :--- | :--- |
| **VJP Exactness** | $\|J^\top v - \nabla_{\text{autograd}}\mathcal{L}\|_\infty$ | $< 10^{-5}$ | `torch.autograd.gradcheck`, finite diff |
| **Complex Conjugate Grad** | $\text{grad}_z = 2 \frac{\partial \mathcal{L}}{\partial z^*} = \frac{\partial \mathcal{L}}{\partial x} + i \frac{\partial \mathcal{L}}{\partial y}$ | Identical to component-wise real gradients | `torch.complex64`, `torch.complex128` |
| **Tape Memory Invariant** | Memory saved in forward $\le$ minimal required inputs | Zero unnecessary tensor retention | `setup_context` API, `ctx.save_for_backward` |
| **Finite Difference Convergence** | $\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon} - f'(x)$ | $\mathcal{O}(\epsilon^2)$ central truncation error | `gradcheck(eps=1e-6, atol=1e-4)` |
| **Accumulated Graph Freedom** | Peak RAM growth over training loop | $\Delta \text{RAM} = 0$ (no graph retention) | `loss.item()` detached scalar |

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> 1. **Do not mutate tensors saved for backward:** Modifying a tensor in-place after it has been registered in `ctx.save_for_backward()` triggers `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.
> 2. **Do not accumulate graph nodes across iterations:** Accumulating loss via `total_loss += loss` retains the entire computation graph across the epoch, causing catastrophic Out-Of-Memory (`OOM`) crashes. Always accumulate detached scalars via `total_loss += loss.item()`.
> 3. **Do not use standard complex derivatives for real-valued losses:** Real-valued loss functions over complex arguments are strictly non-holomorphic (they violate the Cauchy-Riemann equations). Differentiating them requires Wirtinger conjugate calculus.

---

## 2. Mathematical Formalism & First-Principles Derivation

### Reverse-Mode Automatic Differentiation & Vector-Jacobian Products
Let a differentiable composite function be $f: \mathbb{R}^N \to \mathbb{R}^M$ and let scalar objective be $\mathcal{L}: \mathbb{R}^M \to \mathbb{R}$.

The Jacobian matrix $J \in \mathbb{R}^{M \times N}$ is defined by elements:

$$J_{ij} = \frac{\partial f_i(x)}{\partial x_j}$$

By the chain rule, the gradient of $\mathcal{L}$ with respect to the input $x$ is:

$$\nabla_x \mathcal{L} = \left( \frac{\partial \mathcal{L}}{\partial f} \right) J = v^\top J$$

where $v = \nabla_f \mathcal{L} \in \mathbb{R}^M$ is the incoming adjoint vector (upstream gradient). Reverse-mode AD evaluates this matrix-vector product from left to right (transposed VJP) without ever materializing the full $M \times N$ Jacobian matrix.

### Analytical VJP for Affine Projection
For matrix affine transformation $Y = X W + b$ with $X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, and upstream adjoint $G = \frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{B \times D_{\text{out}}}$:

1. **Input Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial X} = G W^\top \in \mathbb{R}^{B \times D_{\text{in}}}$$
2. **Weight Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial W} = X^\top G \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$$
3. **Bias Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^B G_{i, :} = \mathbf{1}_B^\top G \in \mathbb{R}^{1 \times D_{\text{out}}}$$

### Wirtinger Calculus on the Complex Plane
Let $z = x + i y \in \mathbb{C}$ where $x, y \in \mathbb{R}$. The complex conjugate is $z^* = x - i y$.
We can write $x$ and $y$ in terms of $z$ and $z^*$:

$$x = \frac{z + z^*}{2}, \quad y = \frac{z - z^*}{2i}$$

Treating $z$ and $z^*$ as formal independent variables, the **Wirtinger differential operators** are defined as:

$$\frac{\partial}{\partial z} = \frac{1}{2} \left( \frac{\partial}{\partial x} - i \frac{\partial}{\partial y} \right), \quad \frac{\partial}{\partial z^*} = \frac{1}{2} \left( \frac{\partial}{\partial x} + i \frac{\partial}{\partial y} \right)$$

### Non-Holomorphic Real-Valued Losses & PyTorch Gradient Convention
A function $f: \mathbb{C} \to \mathbb{C}$ is holomorphic (complex-differentiable) if and only if it satisfies the Cauchy-Riemann equations:

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y} \quad \text{and} \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x} \iff \frac{\partial f}{\partial z^*} = 0$$

However, any non-trivial real-valued loss function $\mathcal{L}: \mathbb{C} \to \mathbb{R}$ (such as $\mathcal{L}(z) = \frac{1}{2} |z|^2 = \frac{1}{2}(x^2 + y^2)$) has $v(x, y) \equiv 0$. The Cauchy-Riemann equations require:

$$\frac{\partial u}{\partial x} = 0 \quad \text{and} \quad \frac{\partial u}{\partial y} = 0$$

which only holds for constant functions! Therefore, **no real-valued loss function on complex numbers is holomorphic**.

To optimize real-valued objectives over complex parameters via gradient descent, we consider the first-order differential:

$$d\mathcal{L} = \left\langle \frac{\partial \mathcal{L}}{\partial x}, dx \right\rangle + \left\langle \frac{\partial \mathcal{L}}{\partial y}, dy \right\rangle = 2 \operatorname{Re}\left( \left\langle \frac{\partial \mathcal{L}}{\partial z^*}, dz \right\rangle \right)$$

The direction of steepest descent is given by $-\frac{\partial \mathcal{L}}{\partial z^*}$.

**PyTorch Autograd Convention:**  
PyTorch defines the stored gradient `z.grad` as:

$$\text{z.grad} = 2 \cdot \frac{\partial \mathcal{L}}{\partial z^*} = \frac{\partial \mathcal{L}}{\partial x} + i \frac{\partial \mathcal{L}}{\partial y}$$

This convention ensures that the complex parameter update:

$$z \leftarrow z - \eta \cdot \text{z.grad} \iff \begin{cases} x \leftarrow x - \eta \frac{\partial \mathcal{L}}{\partial x} \\ y \leftarrow y - \eta \frac{\partial \mathcal{L}}{\partial y} \end{cases}$$

matches independent real gradient updates on the real and imaginary parts simultaneously.

---

## 3. Invertibility, Conditioning & Numerical Invariants

### Finite Differences & Gradcheck Error Bounds
`torch.autograd.gradcheck` verifies analytical gradients against symmetric two-point central finite differences:

$$f'_c(x) = \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

By Taylor expansion:
$$f(x + \epsilon) = f(x) + f'(x)\epsilon + \frac{1}{2}f''(x)\epsilon^2 + \frac{1}{6}f'''(x)\epsilon^3 + \mathcal{O}(\epsilon^4)$$
$$f(x - \epsilon) = f(x) - f'(x)\epsilon + \frac{1}{2}f''(x)\epsilon^2 - \frac{1}{6}f'''(x)\epsilon^3 + \mathcal{O}(\epsilon^4)$$
Subtracting yields the central difference error:
$$f'_c(x) - f'(x) = \frac{1}{6}f'''(x)\epsilon^2 + \mathcal{O}(\epsilon^4) = \mathcal{O}(\epsilon^2)$$

For float64 precision ($\epsilon_m \approx 10^{-16}$), optimal step size is $\epsilon \approx \epsilon_m^{1/3} \approx 10^{-6}$, yielding truncation error $\sim 10^{-12}$.

---

## 4. Cross-Framework Comparative Reference

| Operation | PyTorch 2.x Track (`pytorch2/`) | JAX / Functional Track (`jax/`) | Rust Candle Track (`candle/`) | Fused Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Custom Operator** | `torch.autograd.Function` + `setup_context` | `jax.custom_vjp` (`defvjp`) | Custom struct with backward VJP fn | Fused C/Triton function |
| **Residual Storage** | `ctx.save_for_backward(...)` | Returned tuple in `fwd()` | Tensor cloning in struct | Register accumulation |
| **Complex Grad** | `z.grad = dL/dx + i dL/dy` | `jax.grad` (Wirtinger conjugate) | Explicit real/imag split | Fused complex arithmetic |
| **Grad Verification** | `torch.autograd.gradcheck()` | `jax.test_util.check_grads()` | Analytical target assertion | Unit test assertion |
| **Double Backward** | Define custom `backward` with autograd | Auto-differentiable `bwd` | Recursive backward graph | Direct 2nd-order kernel |

---

## 5. Hardware & Compilation Dynamics

### Activation Lifetime & Tape Memory Footprint
During the forward pass of deep neural networks, intermediate activations required for computing backward VJPs are retained on the autograd tape.
- **Peak Memory:** Without activation checkpointing, memory consumption scales as $\mathcal{O}(L \cdot B \cdot D)$, where $L$ is network depth.
- **Minimal Residual Saving (`setup_context`):** Saving only tensors whose values appear in the Jacobian (e.g. saving $X$ for $\frac{\partial \mathcal{L}}{\partial W}$, but NOT saving intermediate bias additions) minimizes DRAM footprint.
- **AOTAutograd & torch.compile:** Under `torch.compile`, `AOTAutograd` traces forward and backward graphs ahead-of-time, executing min-cut partitioning to decide which activations are cheaper to recompute versus store in DRAM.

---

## 6. Failure Modes, Pathology Lab & Anti-Patterns

### Pathology Table

| ID | Failure Mode | Root Cause | Observable Symptom | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **FAT-01** | Saved Tensor In-Place Corruption | Modifying a tensor after `ctx.save_for_backward()` | `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` | Use out-of-place operations or call `.clone()` before saving |
| **FAT-02** | Autograd Graph Memory Leak | Accumulating loss via `total += loss` | Monotonically climbing VRAM until fatal CUDA Out-Of-Memory (`OOM`) | Accumulate detached scalars: `total += loss.item()` |
| **FAT-03** | Complex Conjugate Gradient Sign Error | Using $dL/dz$ instead of $dL/dz^*$ in custom backward | Complex parameters diverge or increase loss | Follow PyTorch convention: `z.grad = dL/dx + i*dL/dy` |
| **FAT-04** | Disconnected Graph Gradient None | Performing detached operation (e.g. `.numpy()`, `.item()`) in forward | `param.grad is None` or zeros throughout network | Ensure all intermediate ops remain within framework tensor dispatch |

### Concrete Misuse Snippet
```python
# ANTI-PATTERN 1: Memory leak via graph retention in training loop
total_loss = 0.0
for x, y in dataloader:
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    total_loss += loss  # CATASTROPHIC: Retains entire computation graph across all batches!

# CORRECT:
total_loss += loss.item()
```

---

## 7. Quantitative Evaluation & Verification Harness

### Verification Harness Specification
Every implementation of the Autograd & Wirtinger track must pass the following automated assertions:

```python
# 1. Finite-Difference Gradcheck
test_passed = torch.autograd.gradcheck(scaled_linear, (x, w, b, 1.5), eps=1e-6, atol=1e-4)
assert test_passed

# 2. Complex Wirtinger Invariant
z = x @ w + b
loss = 0.5 * torch.sum(z.real**2 + z.imag**2)
loss.backward()
# PyTorch convention: w.grad == x.mH @ (z)
torch.testing.assert_close(w.grad, x.mH @ z, atol=1e-5, rtol=1e-5)
```

---

## 8. Related Nodes & Architectural Lineage

### Lineage Graph
```
                        01. Linear Neuron & Invertible Transforms
                                          |
                                          v
                    02. Tensors, Views, Strides & Memory Layouts
                                          |
                                          v
              03. Autograd Mechanics, Complex Wirtinger & Custom Functions
                                          |
         +--------------------------------+--------------------------------+
         |                                                                 |
         v                                                                 v
04. Forward AD & Functional Transforms                     05. Saved Tensor Hooks
```

### Relative Navigation
- **Master Curriculum Index:** [README.md](../../README.md)
- **Goal-Oriented Index:** [INDEX_BY_GOAL.md](../../INDEX_BY_GOAL.md)
- **Interactive Notebook Explorer:** [03_autograd_mechanics_complex_wirtinger_and_custom_functions.ipynb](../../notebooks/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.ipynb)
- **PyTorch 2.x Production Module:** [03_autograd_mechanics_complex_wirtinger_and_custom_functions.py](../../pytorch2/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.py)
- **Functional JAX Track:** [03_autograd_mechanics_complex_wirtinger_and_custom_functions.py](../../jax/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.py)
- **Native Rust Candle Track:** [03_autograd_mechanics_complex_wirtinger_and_custom_functions.rs](../../candle/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.rs)
- **CPU Fused Kernel:** [03_autograd_mechanics_complex_wirtinger_and_custom_functions_cpu.py](../../kernels/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions_cpu.py)
- **Golden Reference Checkpoint:** [03_autograd_mechanics_complex_wirtinger_and_custom_functions.safetensors](../../weights/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.safetensors)
- **Previous Topic:** [02. Tensors, Views, Strides & Memory Layouts](02_tensors_views_strides_and_memory_layouts.md)
- **Next Topic:** 04. Forward AD and Functional Transforms (`torch.func`) *(Upcoming in Category 00)*
