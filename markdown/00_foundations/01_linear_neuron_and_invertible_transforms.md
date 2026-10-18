# 01. Linear Neuron and Invertible Coordinate Transforms

## 1. Executive Architectural Brief & Core Invariants

The linear neuron is the foundational parametric atom of deep learning, realizing an affine mapping between Euclidean vector spaces. When combined with coordinate transformations whose Jacobian determinant is strictly non-zero, it forms the bedrock for normalized gradient propagation, invertible flows, and conditioned regression.

```
+-----------------------------------------------------------------------------------+
|                        TOPIC TAXONOMY & LINEAGE                                   |
|                                                                                   |
|  McCulloch-Pitts (1943)  -->  Rosenblatt (1958)  -->  Widrow-Hoff ADALINE (1960)  |
|  [Binary Threshold]           [Perceptron Step]       [Continuous Affine + MSE]   |
|                                                               |                   |
|                                                               v                   |
|                                    +------------------------------------------+   |
|                                    | 01. Linear Neuron & Invertible Transforms|   |
|                                    +------------------------------------------+   |
|                                         /                  |                 \    |
|                                        v                   v                  v   |
|                              02. Tensors & Memory   03. Autograd & Tape   05. GLMs|
+-----------------------------------------------------------------------------------+
```

### Core Invariant Table

| Metric / Invariant | Mathematical Expression | Target Bound / Invariant | Framework Mechanism |
| :--- | :--- | :--- | :--- |
| **Inverse Reconstruction** | $\|x - f^{-1}(f(x))\|_\infty$ | $< 10^{-5}$ | `torch.linalg.solve`, `candle::broadcast_div` |
| **Jacobian Log-Determinant** | $\log \|\det(J_f)\|$ | Finite ($\ne -\infty, \ne +\infty$) | `torch.linalg.slogdet`, `jnp.linalg.slogdet` |
| **Spectral Condition Number**| $\kappa(W) = \frac{\sigma_{\max}(W)}{\sigma_{\min}(W)}$ | $\kappa(W) < 10^3$ (well-conditioned) | `torch.linalg.svdvals` |
| **Loss Convexity** | $\nabla^2_{\theta} \mathcal{L}(\theta) = \frac{2}{N} \tilde{X}^\top \tilde{X}$ | Positive semi-definite ($\succeq 0$) | Global minimum guaranteed |
| **Normal Equation Parity** | $\|\theta_{\text{GD}}^* - \theta_{\text{OLS}}^*\|_2$ | $< 10^{-3}$ | Iterative vs Closed-Form convergence |

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> 1. **Intrinsic Non-Linear Manifolds:** Unadorned linear neurons cannot partition non-linearly separable topologies (e.g. XOR, multi-modal class manifolds) without polynomial or kernel feature maps.
> 2. **Rank-Deficient Regime ($D \gg N$):** When feature dimension $D$ exceeds sample count $N$, the Gram matrix $X^\top X$ is strictly rank-deficient, rendering OLS normal equations non-invertible and gradient descent underdetermined.
> 3. **Non-Stationary / Adversarial Outliers:** Under MSE loss, outlier influence scales quadratically $\mathcal{O}(e^2)$, degrading parameter estimates unless robust estimators (Huber loss, Student-t likelihood) are substituted.

---

## 2. Mathematical Formalism & First-Principles Derivation

### Continuous & Discrete Formulation
Let input vector $x \in \mathbb{R}^{D_{\text{in}}}$ and target $y \in \mathbb{R}^{D_{\text{out}}}$. The affine transformation is defined as:

$$f(x; W, b) = W^\top x + b$$

In batched discrete matrix notation with batch dimension $N$, design matrix $X \in \mathbb{R}^{N \times D_{\text{in}}}$, weight matrix $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, and row bias vector $b^\top \in \mathbb{R}^{1 \times D_{\text{out}}}$:

$$\hat{Y} = X W + \mathbf{1}_N b^\top$$

Augmenting the design matrix $\tilde{X} = [X, \mathbf{1}_N] \in \mathbb{R}^{N \times (D_{\text{in}} + 1)}$ and parameter matrix $\Theta = [W^\top, b]^\top \in \mathbb{R}^{(D_{\text{in}} + 1) \times D_{\text{out}}}$:

$$\hat{Y} = \tilde{X} \Theta$$

The empirical Mean Squared Error (MSE) objective is:

$$\mathcal{L}(\Theta) = \frac{1}{N} \|Y - \tilde{X} \Theta\|_F^2 = \frac{1}{N} \operatorname{tr}\left( (Y - \tilde{X} \Theta)^\top (Y - \tilde{X} \Theta) \right)$$

### Step-by-Step Analytical Derivation of the OLS Normal Equations

1. **Matrix Expansion:**
   $$\mathcal{L}(\Theta) = \frac{1}{N} \operatorname{tr}\left( Y^\top Y - Y^\top \tilde{X} \Theta - \Theta^\top \tilde{X}^\top Y + \Theta^\top \tilde{X}^\top \tilde{X} \Theta \right)$$

2. **First Variation with respect to $\Theta$:**
   $$\nabla_{\Theta} \mathcal{L}(\Theta) = \frac{1}{N} \left( -2 \tilde{X}^\top Y + 2 \tilde{X}^\top \tilde{X} \Theta \right) = \frac{2}{N} \tilde{X}^\top (\tilde{X} \Theta - Y)$$

3. **Stationary Condition:**
   $$\nabla_{\Theta} \mathcal{L}(\Theta^*) = \mathbf{0} \implies \tilde{X}^\top \tilde{X} \Theta^* = \tilde{X}^\top Y$$

4. **Closed-Form Solution:**
   Assuming $\tilde{X}^\top \tilde{X}$ is non-singular:
   $$\Theta^* = (\tilde{X}^\top \tilde{X})^{-1} \tilde{X}^\top Y$$

### Geometric Intuition & Curvature
The Hessian of $\mathcal{L}(\Theta)$ is constant across the entire parameter space:

$$\mathbf{H} = \nabla^2_{\Theta} \mathcal{L}(\Theta) = \frac{2}{N} \tilde{X}^\top \tilde{X}$$

Because $\tilde{X}^\top \tilde{X}$ is symmetric positive semi-definite, the loss surface is a quadratic paraboloid with a unique global minimum. The condition number of $\mathbf{H}$, $\kappa(\mathbf{H}) = \lambda_{\max}(\mathbf{H}) / \lambda_{\min}(\mathbf{H})$, governs the eccentricity of the error contours. When $\kappa(\mathbf{H}) \gg 1$, gradient descent exhibits severe oscillatory behavior along high-curvature directions while stalling along low-curvature valleys.

---

## 3. Invertibility, Conditioning & Numerical Invariants

### Exact Forward and Inverse Maps
When $D_{\text{in}} = D_{\text{out}} = D$ and $\det(W) \ne 0$, the affine mapping is a diffeomorphism $f: \mathbb{R}^D \to \mathbb{R}^D$:

$$y = x W + b \iff x = (y - b) W^{-1}$$

To avoid explicit matrix inversion ($W^{-1}$) which introduces numerical error and numerical instability, the exact inverse is computed via linear system solve:

$$W^\top x^\top = (y - b)^\top \implies x^\top = \operatorname{solve}(W^\top, (y - b)^\top)$$

### Jacobian Matrix & Log-Determinant
The Jacobian of the forward affine map is:

$$J_f(x) = \frac{\partial f(x)}{\partial x} = W^\top$$

The change of variable rule for probability densities $p_Y(y) = p_X(x) |\det(J_f)|^{-1}$ requires the log-absolute determinant of the Jacobian:

$$\log |\det(J_f)| = \log |\det(W)| = \sum_{i=1}^D \log |\lambda_i(W)| = \sum_{i=1}^D \log \sigma_i(W)$$

### Foundational Coordinate Transforms

1. **Affine Min-Max Scaling ($[-1, 1]$):**
   $$t = 2 \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1 \iff x = \frac{t + 1}{2} (x_{\max} - x_{\min}) + x_{\min}$$
   $$\left|\det\left(\frac{\partial t}{\partial x}\right)\right| = \prod_{d=1}^D \frac{2}{x_{\max, d} - x_{\min, d}}$$

2. **Standard Z-Score Standardization:**
   $$t = \frac{x - \mu}{\sigma} \iff x = t \cdot \sigma + \mu$$
   $$\log \left|\det\left(\frac{\partial t}{\partial x}\right)\right| = -\sum_{d=1}^D \log \sigma_d$$

3. **Parameterized Tanh Projection ($(-1, 1)$):**
   $$t = \tanh(s(x - c)) \iff x = \frac{\operatorname{arctanh}(\operatorname{clip}(t, -1+\epsilon, 1-\epsilon))}{s} + c$$
   $$\frac{d t}{d x} = s (1 - t^2)$$

### Floating-Point Sensitivity & Stability

| Transform | Stable Numerical Implementation | Unstable / Catastrophic Form | Failure Threshold |
| :--- | :--- | :--- | :--- |
| **Affine Inversion** | `torch.linalg.solve(W.T, (y-b).T).T` | `(y - b) @ torch.inverse(W)` | $\kappa(W) > 10^7$ |
| **Tanh Inverse** | `torch.arctanh(torch.clamp(t, -1+eps, 1-eps))` | `torch.arctanh(t)` | $\|t\| \to 1.0$ (produces `inf`) |
| **Variance Normalization** | `(x - mu) / torch.clamp_min(std, eps)` | `(x - mu) / std` | Constant feature (`std == 0`) |

---

## 4. Cross-Framework Comparative Reference

| Operation | PyTorch 2.x Track (`pytorch2/`) | JAX / Functional Track (`jax/`) | Rust Candle Track (`candle/`) | Fused Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Forward Pass** | `x @ self.weight + self.bias` | `jnp.matmul(x, p.weight) + p.bias` | `x.matmul(&w)?.broadcast_add(&b)?` | `fused_affine_zscore_forward(...)` |
| **Exact Inverse** | `torch.linalg.solve(self.weight.T, (y - self.bias).T).T` | `jnp.linalg.solve(p.weight.T, (y - p.bias).T).T` | `centered.matmul(&inv_w)?` | `fused_affine_zscore_inverse(...)` |
| **Condition Number** | `s = torch.linalg.svdvals(W); s[0]/s[-1]` | `s = jnp.linalg.svd(W, compute_uv=False); s[0]/s[-1]` | Analytical scalar ratio or external SVD | Vectorized SVD metric |
| **Compilation Hook**| `torch.compile(fullgraph=True)` | `jax.jit(forward)` | Native LLVM `rustc --release` | Streaming fused cache loop |
| **Serialization** | `safetensors.torch.save_file` | `safetensors.numpy.save_file` | `safetensors::serialize` | Universal `.safetensors` |

---

## 5. Hardware & Compilation Dynamics

### Arithmetic Intensity & Roofline Placement
The matrix-vector product $y = x W + b$ for batch size $N$, input dimension $D_{\text{in}}$, and output dimension $D_{\text{out}}$ requires:
- **Floating Point Operations:** $2 N D_{\text{in}} D_{\text{out}} + N D_{\text{out}}$ FLOPs.
- **Memory Traffic:** Reading $X$ ($4 N D_{\text{in}}$ bytes), reading $W$ ($4 D_{\text{in}} D_{\text{out}}$ bytes), reading $b$ ($4 D_{\text{out}}$ bytes), and writing $Y$ ($4 N D_{\text{out}}$ bytes).
- **Arithmetic Intensity:**
  $$I = \frac{\text{FLOPs}}{\text{Bytes}} \approx \frac{2 N D_{\text{in}} D_{\text{out}}}{4(N D_{\text{in}} + D_{\text{in}} D_{\text{out}} + N D_{\text{out}})}$$

For small batch sizes ($N=1$), $I \approx \frac{2 D_{\text{out}}}{4(1 + D_{\text{out}})} \approx 0.5$ FLOP/byte, placing the operation deeply in the **memory bandwidth bound** regime of the Roofline model. For large batch sizes ($N \gg D_{\text{in}}$), $I \approx \frac{D_{\text{out}}}{2}$, becoming **compute bound**.

```
Memory Bandwidth Bound                     Compute Bound
      \                                         /
       \   Peak Bandwidth Slope                /   Peak Tensor Compute
        \       /                             /
  FLOP/s \     /    +------------------------+
          \   /     | Large Batches (N=1024) |
           \ /      +------------------------+
            X
           / \      +------------------------+
          /   \     | Micro Batches (N=1)    |
         /     \    +------------------------+
        +-------+----------------------------+
       0.1      1            10             100   Arithmetic Intensity (FLOP/byte)
```

### Compiler Fusion Benefits (`torch.compile` & XLA)
In an unfused execution chain `y = tanh((x - mu) / sigma @ W + b)`:
1. Intermediate tensor `x_norm` written to and read from DRAM.
2. Intermediate tensor `proj` written to and read from DRAM.
3. Intermediate tensor `biased` written to and read from DRAM.
4. Final activation `y` written to DRAM.

Under `torch.compile(mode="reduce-overhead")` (Inductor) and JAX XLA, the point-wise normalization, bias addition, and tanh activation are fused into a single kernel with the GEMM epilogue, eliminating 3 intermediate DRAM read/write roundtrips and achieving up to $2.5\times$ memory speedup.

---

## 6. Failure Modes, Pathology Lab & Anti-Patterns

### Pathology Table

| ID | Failure Mode | Root Cause | Observable Symptom | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **FAT-01** | Gradient Explosion from Unscaled Coordinates | Input feature scale $X \sim [10^3, 10^4]$ with learning rate $\eta \ge 10^{-3}$ | Gradients explode to $>10^7$ or `NaN` within 2 iterations | Mandatory Z-Score or Min-Max normalization before affine projection |
| **FAT-02** | Singular Gram Matrix in OLS | Collinear features or $D > N$ | `LinAlgError: Singular matrix` during $(X^\top X)^{-1}$ | Moore-Penrose pseudoinverse or Ridge regularization $(X^\top X + \lambda I)^{-1}$ |
| **FAT-03** | Saturation in Inverse Tanh | Forward values $t = \pm 1.0$ due to large inputs | `arctanh(1.0) = inf`, corrupting backward gradients | Clamping $t$ to $[-1+\epsilon, 1-\epsilon]$ prior to inverse evaluation |
| **FAT-04** | Vanishing Coordinates from Zero Variance | Constant feature across dataset ($x_i = c \implies \sigma_i = 0$) | Division by zero producing `NaN` in normalization | Floor clamping denominator: $\max(\sigma, \epsilon)$ |

### Concrete Misuse Snippet
```python
# ANTI-PATTERN: Unscaled coordinates fed into standard linear layer
x_unscaled = torch.randn(64, 1) * 5000.0
w = torch.tensor([[0.5]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

pred = x_unscaled @ w + b
loss = torch.mean((pred - 2.0 * x_unscaled) ** 2)
loss.backward()

# Observable Pathology:
# w.grad is on the order of 10^7 -> immediate step divergence!
print(f"Exploding Gradient: {w.grad.item():.2e}")
```

---

## 7. Quantitative Evaluation & Verification Harness

### Verification Harness Specification
Every implementation of the Linear Neuron and Invertible Transforms track must satisfy the following automated assertions:

```python
# 1. Round-Trip Invertibility Invariant
y = model.forward(x)
x_recon = model.inverse(y)
assert torch.norm(x - x_recon, p=float("inf")) < 1e-5

# 2. Transitive Cross-Track Parity
# Torch vs JAX vs Candle outputs on golden test input
weights, test_in, golden_out, meta = core.parity.load_reference_checkpoint(
    "weights/00_foundations/01_linear_neuron_and_invertible_transforms.safetensors"
)
y_torch = model(test_in)
assert torch.norm(y_torch - golden_out, p=float("inf")) < 1e-5

# 3. Analytic vs Autograd Gradient Parity
# dL/dW = (2/N) * X^T (X W + b - Y)
grad_w_analytic = (2.0 / len(x)) * (x.T @ (y - y_true))
torch.testing.assert_close(model.weight.grad, grad_w_analytic, atol=1e-5, rtol=1e-5)
```

---

## 8. Related Nodes & Architectural Lineage

### Lineage Graph
```
                     01. Linear Neuron & Invertible Transforms
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
02. Tensors & Memory Layouts                               03. Computational Graphs
         |                                                         |
         +----------------------------+----------------------------+
                                      |
                                      v
                        05. Generalized Linear Models
                                      |
                                      v
                        06. Multi-Layer Perceptrons
```

### Relative Navigation
- **Master Curriculum Index:** [README.md](../../README.md)
- **Goal-Oriented Index:** [INDEX_BY_GOAL.md](../../INDEX_BY_GOAL.md)
- **Interactive Notebook Explorer:** [01_linear_neuron_and_invertible_transforms.ipynb](../../notebooks/00_foundations/01_linear_neuron_and_invertible_transforms.ipynb)
- **PyTorch 2.x Production Module:** [01_linear_neuron_and_invertible_transforms.py](../../pytorch2/00_foundations/01_linear_neuron_and_invertible_transforms.py)
- **Functional JAX Module:** [01_linear_neuron_and_invertible_transforms.py](../../jax/00_foundations/01_linear_neuron_and_invertible_transforms.py)
- **Native Rust Candle Crate:** [01_linear_neuron_and_invertible_transforms.rs](../../candle/00_foundations/01_linear_neuron_and_invertible_transforms.rs)
- **CPU Fused Kernel:** [01_linear_neuron_and_invertible_transforms_cpu.py](../../kernels/00_foundations/01_linear_neuron_and_invertible_transforms_cpu.py)
- **Golden Reference Checkpoint:** [01_linear_neuron_and_invertible_transforms.safetensors](../../weights/00_foundations/01_linear_neuron_and_invertible_transforms.safetensors)
- **Next Topic:** [02. Tensors, Views, Strides & Memory Layouts](02_tensors_views_strides_and_memory_layouts.md)
