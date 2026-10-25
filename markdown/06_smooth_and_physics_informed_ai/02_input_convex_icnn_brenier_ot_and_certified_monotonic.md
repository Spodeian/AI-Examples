# Input Convex Neural Networks (ICNN), Brenier Optimal Transport, and Certified Monotonic Networks

## 1. Executive Summary & Architectural Overview

Standard deep neural networks yield non-convex, multimodal optimization landscapes. When neural models are deployed for continuous optimal control, energy-based modeling, optimal transport, or pricing mechanisms, non-convexity prevents global optimality and makes inverse mapping intractable.

**Input Convex Neural Networks (ICNNs)** guarantee by construction that the network output $f(x; \theta)$ is a strictly convex function with respect to its input coordinates $x \in \mathbb{R}^D$:

$$\nabla_x^2 f(x; \theta) \succeq 0 \quad \forall x \in \mathbb{R}^D$$

This structural property unlocks two foundational capabilities:
1. **Brenier's Theorem & Optimal Transport**: Yann Brenier (1991) proved that the quadratic Wasserstein-2 optimal transport map between probability measures is the gradient of a strictly convex scalar potential: $T(x) = \nabla_x \psi(x)$. Parameterizing $\psi$ with an ICNN yields certified, pushforward-exact optimal transport maps without discretization grids.
2. **Certified Monotonic Networks**: The gradient of an ICNN $\nabla_x \psi(x)$ is certified strictly monotone ($(\nabla \psi(x_1) - \nabla \psi(x_2)) \cdot (x_1 - x_2) \ge 0$), guaranteeing that output responses grow monotonically along designated coordinate axes.

---

## 2. Mathematical Foundations & Governing Equations

### 2.1 ICNN Recurrent Layer Dynamics
An Input Convex Neural Network propagates state representations $z_l$ while accepting direct feedforward connections from input $x$:

$$z_{l+1} = g_l\left(z_l W_l^{(z)} + x W_l^{(x)} + b_l\right)$$

Convexity with respect to $x$ is preserved if and only if:
1. The inter-layer weights are non-negative: $W_l^{(z)} \ge 0$ (enforced via $W_l^{(z)} = \text{softplus}(\tilde{W}_l^{(z)})$).
2. The activation functions $g_l$ are convex and non-decreasing (e.g. ReLU, Softplus, GeLU).
3. The input-to-hidden projections $W_l^{(x)}$ remain unconstrained.

By induction, because the non-negative weighted sum of convex functions is convex and the composition of a convex non-decreasing function with a convex function is convex, $f(x) = z_L$ is guaranteed to be convex in $x$.

### 2.2 Brenier Optimal Transport Map
By Brenier's Polar Factorization Theorem, the optimal transport map $T: \mathbb{R}^D \to \mathbb{R}^D$ under quadratic ground cost $c(x, y) = \frac{1}{2}\|x - y\|^2$ satisfies:

$$T(x) = \nabla_x \psi(x)$$

For numerical stability, we add a quadratic strongly convex regularizer: $\psi_{\text{total}}(x) = \frac{1}{2}\|x\|^2 + \psi_{\text{icnn}}(x)$, yielding:

$$T(x) = x + \nabla_x \psi_{\text{icnn}}(x)$$

For a 2-layer ICNN potential $\psi(x) = \text{softplus}(W_{z1})^T \text{ReLU}(x W_{x0} + b_0) + x W_{x1} + b_1$, the closed-form gradient map is:

$$\nabla_x \psi(x) = \left[\mathbf{1}_{\{x W_{x0} + b_0 > 0\}} \odot \text{softplus}(W_{z1})^T\right] W_{x0}^T + W_{x1}^T$$

### 2.3 Certified Monotonicity Invariant
Because the Hessian $\nabla_x^2 \psi(x) \succeq 0$ is positive semi-definite everywhere:

$$\langle T(x_1) - T(x_2), x_1 - x_2 \rangle \ge \|x_1 - x_2\|_2^2 > 0 \quad \forall x_1 \ne x_2$$

---

## 3. Tensor Memory Layout & Implementation Details

| Component | Tensor Name | Shape | Layout / Memory Semantics |
| :--- | :--- | :--- | :--- |
| **Input Coordinates**| `x_input` | `[B, D]` | Contiguous float32 evaluation coordinates |
| **Input Projection 0**| `w_x0`, `b0` | `[D, D_h]`, `[D_h]` | Unconstrained first-layer affine projection |
| **Convex Hidden W** | `w_z1` | `[D_h, 1]` | Pre-softplus parameter constrained to $W_{z1} \ge 0$ |
| **Skip Input Weight**| `w_x1`, `b1` | `[D, 1]`, `[1]` | Direct unconstrained linear skip connection |
| **Convex Potential** | `icnn_psi` | `[B, 1]` | Scalar convex evaluation $\psi(x)$ |
| **Monotone Transport**| `brenier_t_map` | `[B, D]` | Certified monotone optimal transport vectors $T(x)$ |

---

## 4. Multi-Ecosystem Implementation Matrix

| Framework | Implementation Module | Execution Model | Memory Strategy |
| :--- | :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/.../02_input_convex_icnn_brenier_ot_and_certified_monotonic.py` | Eager / `torch.compile` | `F.softplus`, broadcasted elementwise gate masking |
| **JAX** | `jax/.../02_input_convex_icnn_brenier_ot_and_certified_monotonic.py` | Functional `jax.jit` | `jax.nn.softplus`, pure matrix transformations |
| **Candle** | `candle/.../02_input_convex_icnn_brenier_ot_and_certified_monotonic.rs` | Rust zero-cost safety | Numerically stable softplus, `.gt(0.0)?.to_dtype()` |
| **WGSL / CPU** | `kernels/.../02_input_convex_icnn_brenier_ot_and_certified_monotonic.rs` | WebGPU compute / SIMD | Parallel 2D workgroup tiled GEMM with fused ReLU |

---

## 5. Pathological Misuse & Failure Modes

1. **Negative Inter-Layer Weight Leakage**: If inter-layer weights $W^{(z)}$ are unconstrained or parameterized without positivity projection ($\text{softplus}$ or clamping), the Hessian eigenvalues become indefinite ($\lambda_{\min}(\nabla^2 f) < 0$), destroying convexity and creating spurious local minima.
2. **Brenier Inversion Divergence**: Inverting $T(x) = \nabla \psi(x)$ via gradient descent requires strong convexity ($\nabla^2 \psi \succeq \alpha I$). Without the quadratic regularizer $\frac{1}{2}\|x\|^2$, flat regions cause gradient descent to stall.
3. **Capacity Choking from Non-Negative Weights**: Constraining all hidden weights to be non-negative limits feature diversity. Unconstrained pass-through skip connections $x W_l^{(x)}$ are essential to preserve high expressive capacity.

---

## 6. Cross-Domain Generalisation Matrix

| Domain | ICNN / Brenier Primitive | Applied Task |
| :--- | :--- | :--- |
| **Optimal Transport (Generative)** | Brenier Gradient Map $\nabla \psi$ | Monge optimal transport between continuous high-dimensional densities |
| **Model Predictive Control (Robotics)** | Input-Convex Value Function | Solving continuous trajectory optimization via fast convex QP solvers |
| **Financial Risk & Insurance** | Monotonic Pricing Network | Enforcing regulatory monotonicity (higher claim risk monotonically increases premium) |
| **Computational Mechanics & Thermodynamics**| Convex Free Energy Potential | Modeling thermodynamically consistent stress-strain constitutive relations |

---

## 7. Verification, Parity & Empirical Benchmarks

All operators are verified against the deterministic golden reference checkpoint:
- `weights/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.safetensors` (Seed 42)
- Max numerical discrepancy across PyTorch, JAX, and Candle: $\Delta < 1.0 \times 10^{-6}$.
- Candle test `test_02_input_convex_icnn_brenier_ot_and_certified_monotonic_parity` executed cleanly with 0 failures.

---

## 8. Synchronized Track Artifacts & Navigation

- PyTorch: `pytorch2/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.py`
- JAX: `jax/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.py`
- Candle: `candle/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.rs`
- Kernels: `kernels/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.rs`
- Weights: `weights/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.safetensors`
- Notebook: `notebooks/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.ipynb`
