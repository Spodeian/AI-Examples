# Second-Order and Curvature Optimisers: L-BFGS, Sophia, AdaHessian, and Levenberg-Marquardt

## 1. Executive Summary & Intuitive Mental Model

First-order gradient methods (SGD, AdamW) navigate the optimization landscape using only local tangent planes (the gradient vector $g = \nabla f(\theta)$). When objective landscapes exhibit ill-conditioned anisotropic valleys, high condition numbers $\kappa = \lambda_{\max}(H) / \lambda_{\min}(H) \gg 1$, or non-convex saddle structures, first-order methods oscillate transversely across steep ravines while creeping agonizingly slowly along gentle descent directions.

Second-order optimization exploits curvature information encoded in the Hessian matrix $H = \nabla^2 f(\theta)$. In the idealized quadratic regime, the exact Newton-Raphson update $\Delta \theta = - H^{-1} g$ leaps directly to the global minimum in a single step regardless of anisotropic conditioning. However, in modern deep learning with $d \sim 10^7 \text{--} 10^{11}$ parameters:
1. Materializing the full Hessian $H \in \mathbb{R}^{d \times d}$ requires $O(d^2)$ memory (terabytes to petabytes).
2. Inverting or factorizing $H$ requires $O(d^3)$ compute FLOPs.
3. Stochastic mini-batch noise and indefinite saddle geometries ($\lambda_i < 0$) cause pure Newton steps to accelerate directly toward local *maxima* or saddle ridges.

This treatise formalizes four modern paradigms of second-order and curvature optimization that circumvent these limits:
- **L-BFGS (Limited-Memory BFGS)**: The gold standard quasi-Newton method approximating the inverse Hessian $H^{-1}$ via a sliding history buffer of $m$ displacement and gradient-difference vectors using the celebrated $O(m \cdot d)$ two-loop recursion.
- **Sophia-G (Second-Order Clipped Stochastic Optimization)**: Estimates the diagonal Hessian using Gauss-Newton-Bartlett outer products ($g \odot g$) or Hutchinson vector products ($z \odot (Hz)$), scaling coordinates by curvature while applying element-wise clipping $\text{clamp}(m_t / \max(h_t, \epsilon), -\rho, \rho)$ to guarantee bounded steps even in non-convex regions.
- **AdaHessian**: Replaces Adam's gradient-variance second moment with an adaptive EMA of the spatial Hessian diagonal ($v_t = \beta_2 v_{t-1} + (1-\beta_2) \bar{D}_t^2$), providing coordinate-adaptive curvature damping.
- **Gauss-Newton & Levenberg-Marquardt (LM)**: Drops difficult second-derivative tensor terms in non-linear least squares by approximating $H \approx J^T J$, augmenting it with a dynamic trust-region damping term $(J^T J + \lambda I)$ that smoothly interpolates between aggressive Gauss-Newton and robust gradient descent.

```
+---------------------------------------------------------------------------------------------------+
|                                 SECOND-ORDER & CURVATURE TAXONOMY                                 |
+------------------------------------+----------------------------------+---------------------------+
| Paradigm                           | Curvature Representation         | Computational Complexity  |
+------------------------------------+----------------------------------+---------------------------+
| Pure Newton-Raphson                | Dense full Hessian H in R^{dxd}  | O(d^2) memory, O(d^3) ops |
| L-BFGS (Nocedal 1980)              | Implicit low-rank history (m<<d) | O(m*d) memory, O(m*d) ops |
| Sophia-G (Liu et al. 2023)         | Diagonal GNB Hessian + Clipping  | O(d) memory, O(d) ops     |
| AdaHessian (Yao et al. 2021)       | EMA of Diagonal Hessian D_t^2    | O(d) memory, O(d) ops     |
| Levenberg-Marquardt (Marquardt 63) | Damped Gauss-Newton (J^T J + λI) | O(n*d) memory, O(d^3) ops |
+------------------------------------+----------------------------------+---------------------------+
```

---

## 2. Mathematical Foundations & First-Principles Derivations

### 2.1 The Newton-Raphson Ideal and the Secant Condition
Consider a twice-differentiable loss function $f: \mathbb{R}^d \to \mathbb{R}$. Expanding $f(\theta)$ as a second-order Taylor series around $\theta_k$:
$$f(\theta_k + \Delta) \approx f(\theta_k) + \nabla f(\theta_k)^T \Delta + \frac{1}{2} \Delta^T \nabla^2 f(\theta_k) \Delta$$
Setting the gradient of this quadratic model with respect to $\Delta$ to zero yields the classical Newton step:
$$\nabla_\Delta \left[ f(\theta_k) + g_k^T \Delta + \frac{1}{2} \Delta^T H_k \Delta \right] = g_k + H_k \Delta = 0 \implies \Delta_k = - H_k^{-1} g_k$$

Quasi-Newton methods replace the explicit Hessian $H_k$ with an approximation $B_k \approx \nabla^2 f(\theta_k)$ or its inverse $C_k \approx H_k^{-1}$. Letting $s_k = \theta_{k+1} - \theta_k$ denote the parameter displacement and $y_k = g_{k+1} - g_k$ denote the gradient change, any valid curvature approximation must satisfy the **Secant Equation**:
$$B_{k+1} s_k = y_k \quad \iff \quad C_{k+1} y_k = s_k$$
This requires the strict **Curvature Condition**:
$$s_k^T y_k > 0$$
which holds whenever the objective is strictly locally convex along the displacement direction.

### 2.2 L-BFGS and the Two-Loop Recursion
The BFGS update formula maintains positive-definiteness via symmetric rank-2 updates. In limited-memory BFGS (L-BFGS), only the last $m$ pairs $\{s_i, y_i\}_{i=k-m}^{k-1}$ are stored. The inverse Hessian-vector product $r = C_k g_k$ is computed on the fly in $O(m \cdot d)$ operations without ever materializing $C_k$ via **Nocedal's Two-Loop Recursion**:

Let $\rho_i = \frac{1}{y_i^T s_i}$.
1. **First Loop (Backward Recursion)**:
   Initialize $q = g_k$.
   For $i = k-1, k-2, \dots, k-m$:
   $$\alpha_i = \rho_i s_i^T q$$
   $$q \leftarrow q - \alpha_i y_i$$
2. **Initial Scaling**:
   Scale the initial base matrix $C_k^{(0)} = \gamma_k I$, where:
   $$\gamma_k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}$$
   $$r = \gamma_k q$$
3. **Second Loop (Forward Recursion)**:
   For $i = k-m, k-m+1, \dots, k-1$:
   $$\beta_i = \rho_i y_i^T r$$
   $$r \leftarrow r + s_i (\alpha_i - \beta_i)$$
The final search direction is $d_k = -r$, and the parameter updates via $\theta_{k+1} = \theta_k + \eta d_k$.

### 2.3 Sophia-G: Second-Order Clipped Stochastic Optimization
Proposed by *Liu et al. (2023)*, Sophia addresses the fundamental challenges of mini-batch stochastic noise and non-convex negative curvature:
1. **Diagonal Hessian Estimation**: Every $k$ steps, a stochastic diagonal estimate $\hat{h}_t \in \mathbb{R}^d$ is computed via the Gauss-Newton-Bartlett (GNB) estimator $\hat{h}_t = g_{\text{sample}} \odot g_{\text{sample}}$ or Hutchinson estimator $\hat{h}_t = z \odot (H z)$ with Rademacher vector $z \sim \{-1, +1\}^d$.
2. **Momentum and Curvature Tracking**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   $$h_t = \beta_2 h_{t-1} + (1 - \beta_2) \hat{h}_t$$
3. **Element-wise Clipped Preconditioning**:
   $$\Delta \theta_t = - \eta \cdot \text{clamp}\left( \frac{m_t}{\max(h_t, \epsilon)}, -\rho, \rho \right)$$
   $$\theta_{t+1} = \theta_t (1 - \eta \lambda) + \Delta \theta_t$$
Because the update is clipped element-wise to $[-\rho, \rho]$, worst-case step sizes are strictly bounded regardless of near-zero or negative curvature, eliminating catastrophic divergence in deep networks.

### 2.4 AdaHessian: Curvature-Scaled Adaptive Learning Rates
Proposed by *Yao et al. (2021)*, AdaHessian replaces the second moment of gradients with an EMA of the diagonal Hessian:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) D_t^2$$
where $D_t \approx \text{diag}(\nabla^2 f(\theta_t))$. Applying bias corrections:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 2.5 Levenberg-Marquardt and Damped Gauss-Newton
For non-linear least squares objectives $f(\theta) = \frac{1}{2} \|R(\theta)\|_2^2$ where $R: \mathbb{R}^d \to \mathbb{R}^n$:
$$\nabla f(\theta) = J(\theta)^T R(\theta)$$
$$\nabla^2 f(\theta) = J(\theta)^T J(\theta) + \sum_{i=1}^n R_i(\theta) \nabla^2 R_i(\theta)$$
Neglecting the residual second-derivative sum gives the Gauss-Newton approximation $H_{GN} = J^T J \succeq 0$. The Levenberg-Marquardt algorithm augments this with trust-region damping $\lambda > 0$:
$$(J^T J + \lambda I) \Delta \theta = - J^T R$$
When $\lambda \to 0$, LM recovers the fast quadratic convergence of Gauss-Newton; when $\lambda \to \infty$, LM gracefully degenerates to conservative gradient descent with step size $1/\lambda$.

---

## 3. Tensor Mechanics & Memory Layout Architecture

```
TENSOR FLOW: L-BFGS TWO-LOOP & DIAGONAL SECOND-ORDER HARNESS
============================================================

1. L-BFGS History Buffer:
   S_hist: [m, d] (contiguous float32 row-major)
   Y_hist: [m, d] (contiguous float32 row-major)
   Memory: 2 * m * d * sizeof(float32) bytes (e.g. 2 * 4 * 8 * 4 = 256 bytes)

2. Sophia-G State:
   theta:         [d] (parameter vector)
   momentum:      [d] (EMA of gradient)
   hess_diag:     [d] (EMA of diagonal Hessian)
   Memory: 3 * d * sizeof(float32) bytes (identical footprint to AdamW!)

3. Levenberg-Marquardt State:
   Jacobian J:    [n, d]
   Residual R:    [n]
   Normal Eq A:   [d, d] -> J^T J + lambda * I (Symmetric Positive Definite)
   Cholesky L:    [d, d] -> Lower triangular factor L L^T = A
```

---

## 4. Multi-Ecosystem Comparative Implementation Matrix

| Metric / Dimension | PyTorch 2.x (`pytorch2/`) | JAX / Functional (`jax/`) | Rust Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **L-BFGS Engine** | Vectorized PyTorch loops | Pure `numpy` recursion | Native Rust f64 iterators | CPU SIMD vector loops |
| **Sophia Clipping** | `torch.clamp` | `np.clip` | Pure Rust `.clamp(-rho, rho)` | WGSL compute shader `clamp()` |
| **Linear Solve (LM)** | `torch.linalg.solve` | `np.linalg.solve` | In-place Cholesky ($L L^T$) | Native Cholesky $L L^T$ in f64 |
| **Memory Allocations** | Transient graph buffers | Zero-allocation array views | Static stack/heap vecs | Single uniform/storage buffer |
| **Deterministic Parity** | Seed 42 golden reference | $< 10^{-6}$ max abs diff | $< 10^{-6}$ max abs diff | $< 10^{-6}$ max abs diff |

---

## 5. Pathological Failure Modes, Anti-Patterns & Diagnostic Guide

### Failure Mode 1: Unclipped Newton Step Jumping to a Saddle-Point Maximum
- **Mechanism**: In non-convex optimization, the Hessian $H$ frequently possesses negative eigenvalues near saddle points or ridges. In a 2D saddle $f(x, y) = x^2 - y^2$, the Hessian is $H = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}$. A pure Newton step computes $\Delta = - H^{-1} g = - \begin{bmatrix} 1/2 & 0 \\ 0 & -1/2 \end{bmatrix} \begin{bmatrix} g_x \\ g_y \end{bmatrix} = \begin{bmatrix} -0.5 g_x \\ +0.5 g_y \end{bmatrix}$. The step along $y$ moves *with* the gradient directly toward the local maximum!
- **Diagnostic Signal**: Sudden loss explosion or divergence toward $\pm \infty$ when passing through saddle points.
- **Cure**: Levenberg-Marquardt damping $(H + \lambda I)$ with $\lambda > |\lambda_{\min}|$ to restore positive definiteness, or Sophia's element-wise clipping with non-negative diagonal estimates $\max(h_t, \epsilon)$.

### Failure Mode 2: Stochastic Mini-Batch L-BFGS Violating the Curvature Condition
- **Mechanism**: Standard L-BFGS assumes $s_k^T y_k > 0$. When evaluating gradients on small, noisy mini-batches $\mathcal{B}_k \neq \mathcal{B}_{k+1}$, noise causes $g(\theta_{k+1}; \mathcal{B}_{k+1}) - g(\theta_k; \mathcal{B}_k)$ to point opposite to $s_k$, resulting in $s_k^T y_k \le 0$. The resulting BFGS update becomes indefinite or singular, generating uphill directions ($d_k^T g_k > 0$).
- **Diagnostic Signal**: `rho_k <= 0`, NaN search directions, or immediate loss increases after quasi-Newton steps.
- **Cure**: Powell damping, skipping history updates when $s_k^T y_k \le \epsilon \|s_k\| \|y_k\|$, or switching to Sophia-G/AdaHessian for stochastic mini-batch settings.

---

## 6. Cross-Domain Generalisation & Mathematical Isomorphisms

| Discipline / Domain | Isomorphic Problem | Curvature Mathematical Role |
| :--- | :--- | :--- |
| **Robotics & SLAM** | Bundle Adjustment & Pose Graph | Levenberg-Marquardt solves large sparse non-linear least squares $(J^T J + \lambda I) \Delta = -J^T r$ |
| **Scientific Computing** | Molecular Dynamics & DFT Energy Minimization | L-BFGS finds ground-state potential energy minima in zero-noise conservative force fields |
| **Deep Sequence Models (LLMs)** | Pretraining Attention & FFN Layers | Sophia-G provides $2\times$ faster pretraining by dampening high-curvature outlier features |
| **Computer Vision (3DGS)** | Gaussian Splatting Photometric Optimization | Damped Gauss-Newton aligns covariance matrices without exploding on transparent primitives |
| **Finance / Econometrics** | GARCH Volatility Surface Calibration | Quasi-Newton methods fit implied volatility surfaces under strict positive-definite constraints |

---

## 7. Verification, Parity & Benchmarking Protocol

All implementations verify 5 quantitative invariants against seed-42 golden reference tensors:
1. `inv1_lbfgs_theta`: Parameter update under L-BFGS two-loop search direction ($\|Y_{\text{torch}} - Y_{\text{golden}}\|_\infty < 10^{-5}$).
2. `inv2_sophia_theta`: Parameter update under Sophia-G clipped preconditioning ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).
3. `inv3_adahessian_theta`: Parameter update under AdaHessian curvature EMA ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).
4. `inv4_lm_theta`: Parameter update under Levenberg-Marquardt damped solve ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).
5. `inv5_anchor`: Deterministic linear projection baseline ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).

---

## 8. Navigation & Synchronized Track Artifacts

- **Jupyter Notebook Track**: [`notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.ipynb)
- **PyTorch 2.x Track**: [`pytorch2/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.py)
- **JAX Functional Track**: [`jax/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.py)
- **Rust Candle Track**: [`candle/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.rs)
- **Hardware Kernel Track**: [`kernels/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.rs)
- **Reference Checkpoints**: [`weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.safetensors)
