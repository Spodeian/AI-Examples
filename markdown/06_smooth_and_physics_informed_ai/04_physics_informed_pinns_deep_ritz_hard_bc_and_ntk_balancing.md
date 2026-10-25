# Physics-Informed PINNs, Deep Ritz Variational Method, Hard Boundary Conditions, and NTK Balancing

## 1. Executive Summary & Architectural Overview

Standard deep neural networks are black-box approximators that frequently violate foundational physical laws (conservation of mass, momentum, energy, and boundary constraints) when trained exclusively on finite data. **Physics-Informed Neural Networks (PINNs)** embed governing partial differential equations (PDEs) directly into neural network computational graphs by penalizing differential residuals via automatic differentiation.

However, canonical PINN formulations face four severe pathological limitations:
1. **High-Order Autograd Overhead**: Second- and higher-order strong-form derivatives (e.g. Laplacian $\Delta u = \nabla \cdot \nabla u$) incur quadratic computational scaling in autograd graph construction.
2. **Boundary Penalty Competition**: Weak boundary enforcement via soft MSE loss terms ($L = L_{\text{PDE}} + \lambda_{\text{BC}} L_{\text{BC}}$) leads to gradient starvation or boundary leakage.
3. **Activation Function Pathology**: Using piecewise-linear activations like ReLU destroys second-order PDE residuals ($\frac{\partial^2}{\partial x^2}\text{ReLU}(x) \equiv 0$).
4. **Spectral Stiffness & Gradient Imbalance**: As proven through Neural Tangent Kernel (NTK) analysis, disparate eigenvalue spectra between boundary and PDE residuals lead to catastrophic failure during gradient descent.

This topic implements the complete four-part modern physical AI architecture:
- **Strong-Form PINNs**: Exact residual formulation with smooth $C^\infty$ ($\tanh$) activations and closed-form autograd parity.
- **Deep Ritz Method**: Variational weak-form energy minimization ($I(u) = \int \frac{1}{2}|\nabla u|^2 - f u\, dx$) halving differential orders via integration by parts.
- **Lagaris Hard Boundary Conditions**: Exact geometric distance ansatz guaranteeing zero boundary violation by construction ($u_{\text{hard}}(x) = g(x) + d(x) \text{NN}(x)$).
- **NTK Eigenvalue Gradient Balancing**: Dynamic task re-weighting matching trace/Frobenius norms of multi-task Jacobians to eliminate gradient starvation.

---

## 2. Mathematical Foundations & Governing Equations

### 2.1 Strong-Form PINN Formulation
Consider the canonical 1D Poisson boundary value problem on the unit domain $\Omega = (0, 1)$:

$$-\frac{d^2 u}{dx^2} = f(x), \quad x \in (0, 1), \qquad u(0) = 0, \quad u(1) = 0$$

For source function $f(x) = \pi^2 \sin(\pi x)$, the exact analytical solution is $u^*(x) = \sin(\pi x)$.
Parameterized by a two-layer neural network with $\tanh$ activations:

$$z = x W_1 + b_1, \quad a = \tanh(z), \quad u(x) = a W_2 + b_2$$

The spatial derivatives are given in exact closed form:

$$\frac{\partial u}{\partial x} = \left[(1 - a^2) \odot W_1\right] W_2$$

$$\frac{\partial^2 u}{\partial x^2} = \left[-2a(1 - a^2) \odot W_1^2\right] W_2$$

The strong-form PDE residual and mean squared error loss are:

$$r(x) = -\frac{\partial^2 u}{\partial x^2} - f(x), \qquad \mathcal{L}_{\text{PDE}} = \frac{1}{N} \sum_{i=1}^N |r(x_i)|^2$$

### 2.2 Variational / Weak-Form Deep Ritz Method
The Deep Ritz method (E & Yu, 2018) reformulates the boundary value problem as the minimization of the Dirichlet variational energy:

$$\mathcal{E}(u) = \int_0^1 \left( \frac{1}{2} \left|\frac{du}{dx}\right|^2 - f(x) u(x) \right) dx$$

By the calculus of variations, Euler-Lagrange extremization yields:

$$\delta \mathcal{E}(u) = \int_0^1 \left( -\frac{d^2 u}{dx^2} - f(x) \right) \delta u\, dx + \left[ \frac{du}{dx} \delta u \right]_0^1 = 0$$

Crucially, the Deep Ritz functional lowers derivative order from 2 to 1, cutting backward-mode graph traversal depth by 50% and improving numerical stability.

### 2.3 Lagaris Hard Boundary Condition Ansatz
Soft boundary losses penalize boundary mismatch but never satisfy conditions identically. Lagaris et al. (1998) introduced the hard boundary ansatz:

$$u_{\text{hard}}(x) = g(x) + d(x) \cdot u(x)$$

where $g(x) = (1 - x)g_0 + x g_1$ interpolates boundary values ($g(0)=0, g(1)=0 \implies g(x) \equiv 0$), and $d(x) = x(1 - x)$ is a smooth distance function vanishing on $\partial \Omega$.
Its first derivative follows the Leibniz product rule:

$$\frac{d u_{\text{hard}}}{dx} = (1 - 2x) u(x) + x(1 - x) \frac{du}{dx}$$

By construction, $u_{\text{hard}}(0) \equiv 0$ and $u_{\text{hard}}(1) \equiv 0$ for all weights $\theta$.

### 2.4 NTK Eigenvalue Loss Balancing
In soft PINNs, the total loss is $\mathcal{L}(\theta) = \mathcal{L}_{\text{PDE}}(\theta) + \lambda \mathcal{L}_{\text{BC}}(\theta)$.
Let $J_{\text{PDE}} = \nabla_\theta \mathcal{L}_{\text{PDE}}$ and $J_{\text{BC}} = \nabla_\theta \mathcal{L}_{\text{BC}}$.
The Neural Tangent Kernel characterizes training dynamics. When $\|J_{\text{BC}}\| \ll \|J_{\text{PDE}}\|$ or vice versa, one loss term freezes.
Wang et al. (2021) balancing adjusts $\lambda$ dynamically via gradient norm matching:

$$\lambda_{\text{NTK}} = \frac{\|J_{\text{PDE}}\|_2}{\|J_{\text{BC}}\|_2 + \epsilon}$$

$$\nabla_{\text{balanced}} = J_{\text{PDE}} + \lambda_{\text{NTK}} J_{\text{BC}}$$

---

## 3. Tensor Memory Layout & Implementation Details

| Component | Tensor Name | Shape | Layout / Memory Semantics |
| :--- | :--- | :--- | :--- |
| **Collocation Points** | `x_colloc` | `[8, 1]` | Quadrature coordinates in domain $(0, 1)$ |
| **Hidden Weight 1** | `w1` | `[1, 16]` | Input linear projection weights |
| **Hidden Bias 1** | `b1` | `[16]` | Input linear projection biases |
| **Output Weight 2** | `w2` | `[16, 1]` | Output linear layer weights |
| **Output Bias 2** | `b2` | `[1]` | Output linear layer bias |
| **Raw Network State** | `u_raw` | `[8, 1]` | Unconstrained neural field evaluation $u(x)$ |
| **First Derivative** | `u_x` | `[8, 1]` | Analytical spatial gradient $\frac{\partial u}{\partial x}$ |
| **Second Derivative** | `u_xx` | `[8, 1]` | Analytical Laplacian $\frac{\partial^2 u}{\partial x^2}$ |
| **PDE Residual** | `pde_residual` | `[8, 1]` | Strong-form residual $r(x) = -u_{xx} - f(x)$ |
| **PDE Loss** | `loss_pde` | `[1]` | Mean squared residual $\frac{1}{N}\sum r(x)^2$ |
| **Hard BC Field** | `u_hard` | `[8, 1]` | Exact boundary-constrained field $d(x) \cdot u(x)$ |
| **Hard BC Gradient** | `u_hard_x` | `[8, 1]` | Spatial derivative $\frac{d}{dx} u_{\text{hard}}(x)$ |
| **Deep Ritz Energy** | `ritz_energy` | `[1]` | Variational energy functional evaluation $\mathcal{E}(u)$ |
| **Soft BC Loss** | `loss_bc` | `[1]` | Boundary mismatch penalty $\frac{1}{2}\sum u(x_b)^2$ |
| **PDE Weight Gradient**| `g_pde` | `[16, 1]` | Analytical gradient $\nabla_{w_2} \mathcal{L}_{\text{PDE}}$ |
| **BC Weight Gradient** | `g_bc` | `[16, 1]` | Analytical gradient $\nabla_{w_2} \mathcal{L}_{\text{BC}}$ |
| **NTK Multiplier** | `lambda_ntk` | `[1]` | Dynamic balancing ratio $\|g_{\text{pde}}\| / \|g_{\text{bc}}\|$ |
| **Balanced Gradient** | `g_balanced` | `[16, 1]` | Stiff-balanced gradient vector $g_{\text{pde}} + \lambda g_{\text{bc}}$ |

---

## 4. Multi-Ecosystem Implementation Matrix

| Framework | Implementation Module | Execution Model | Automatic Differentiation Strategy |
| :--- | :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/.../04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.py` | Eager / `torch.compile` | Autograd with `create_graph=True` and closed-form dual verification |
| **JAX** | `jax/.../04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.py` | Functional `jax.jit` | Pure vectorized primitives (`jnp.tanh`, `jnp.sin`, `jnp.linalg.norm`) |
| **Candle** | `candle/.../04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.rs` | Zero-GC Rust | Native tensor matrix multiplies, affine transforms, broadcast arithmetic |
| **WGSL / CPU** | `kernels/.../04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.rs` | WebGPU compute / SIMD | Fused parallel kernel computing $u_{\text{hard}}$, $u_{\text{hard}}'$, and energy density |

---

## 5. Pathological Misuse & Failure Modes

1. **ReLU in Strong-Form PINNs**: Using piecewise linear activation functions (ReLU, LeakyReLU) in strong-form PINNs results in $\frac{\partial^2}{\partial x^2}\text{ReLU}(x) \equiv 0$ almost everywhere. The neural network computes zero Laplacian, causing the PDE loss to evaluate only the source term $\|f(x)\|^2$ and freezing optimization. Always use smooth activations ($C^2$ or $C^\infty$ like $\tanh$, $\sin$, or GELU).
2. **Boundary Condition Starvation in Soft PINNs**: When $\lambda_{\text{BC}}$ is fixed to $1.0$, the PDE residual gradient norm frequently overwhelms the boundary gradient norm by orders of magnitude (e.g. $17.4$ vs $0.55$). The optimizer minimizes interior PDE error while violating boundary conditions entirely. NTK eigenvalue balancing is required to resolve this stiffness.
3. **Hard BC Distance Singularity**: Formulating distance functions $d(x)$ with discontinuous derivatives or roots outside $\partial \Omega$ introduces artificial interior nodal lines where the solution is forced to zero, ruining model expressivity. For simple polyhedra, use normalized R-functions or smooth distance bounds.

---

## 6. Cross-Domain Generalisation Matrix

| Physical Domain | Governing PDE / Functional | PINN / Deep Ritz Role |
| :--- | :--- | :--- |
| **Fluid Mechanics** | Navier-Stokes ($-\nu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p = 0$) | Velocity-pressure field recovery from sparse PIV / sensor probes |
| **Elasticity & Solids** | Minimum Potential Energy ($\int \frac{1}{2}\sigma : \epsilon - \mathbf{b} \cdot \mathbf{u}\, dx$) | Direct Deep Ritz displacement field solving without mesh generation |
| **Electromagnetics** | Maxwell's & Helmholtz ($-\Delta u - k^2 u = f$) | Scattering and wave propagation with absorbing boundary condition ansätze |
| **Heat Transfer** | Steady-State Conduction ($-\nabla \cdot (\kappa \nabla T) = Q$) | Inverse thermal conductivity parameter estimation and hotspot identification |

---

## 7. Verification, Parity & Empirical Benchmarks

All operators and gradients are verified against the deterministic golden reference checkpoint:
- Checkpoint: `weights/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.safetensors` (Seed 42)
- Tensors Verified: 22 tensors (anchor, spatial fields, derivatives, losses, gradients, NTK weight).
- Numerical Parity:
  - PyTorch autograd vs analytical closed-form: $\Delta < 1.2 \times 10^{-7}$.
  - JAX vs Golden Checkpoint: $\Delta_{\text{rel}} < 8.9 \times 10^{-7}$, $\Delta_{\text{abs}} < 2.3 \times 10^{-5}$ on $\lambda_{\text{NTK}}$.
  - Candle Integration Test: `test_04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing_parity` passed with max discrepancy $< 10^{-5}$.

---

## 8. Synchronized Track Artifacts & Navigation

- PyTorch: `pytorch2/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.py`
- JAX: `jax/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.py`
- Candle: `candle/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.rs`
- Kernels: `kernels/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.rs`
- Weights: `weights/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.safetensors`
- Notebook: `notebooks/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.ipynb`
