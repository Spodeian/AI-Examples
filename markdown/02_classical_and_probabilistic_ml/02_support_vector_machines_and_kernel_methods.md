# 02. Support Vector Machines and Kernel Methods

## 1. Executive Summary
Support Vector Machines (SVMs) and reproducing kernel Hilbert space (RKHS) methods formalize optimal margin classification, nonlinear function regression, and high-dimensional anomaly detection. Grounded in Mercer's theorem, kernel methods compute inner products $\langle \phi(x), \phi(x') \rangle_{\mathcal{H}} = k(x, x')$ in implicit (often infinite-dimensional) feature spaces without materializing coordinates explicitly. This topic unifies primal soft-margin formulation, dual quadratic programming, Kernel Ridge Regression (KRR), One-Class SVM / Support Vector Data Description (SVDD), and low-rank acceleration via Nyström landmark approximations across PyTorch 2.x, JAX, native Rust Candle, and WebGPU compute shaders.

---

## 2. Mathematical Foundations

### 2.1 The Primal and Dual Soft-Margin Formulation
For a binary classification task with training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ where $\mathbf{x}_i \in \mathbb{R}^D$ and $y_i \in \{-1, +1\}$, the soft-margin primal objective balances margin maximization with slack variable penalties:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
Introducing Lagrange multipliers $\alpha_i \ge 0$ and $\beta_i \ge 0$, the Lagrangian is:
$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\beta}) = \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \xi_i - \sum_{i=1}^N \alpha_i \left[ y_i (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) - 1 + \xi_i \right] - \sum_{i=1}^N \beta_i \xi_i$$
Stationarity conditions $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0} \implies \mathbf{w} = \sum_{i=1}^N \alpha_i y_i \phi(\mathbf{x}_i)$ and $\frac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$ yield the Wolfe Dual:
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \quad \sum_{i=1}^N \alpha_i y_i = 0$$

### 2.2 Mercer Kernels Zoo
A symmetric bivariate function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a valid Mercer kernel if its Gram matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$ is positive semi-definite ($\mathbf{v}^\top \mathbf{K} \mathbf{v} \ge 0, \forall \mathbf{v} \in \mathbb{R}^N$):
1. **Radial Basis Function (Gaussian RBF):**
   $$k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \|\mathbf{x} - \mathbf{x}'\|_2^2 \right) = \exp\left( -\frac{\|\mathbf{x} - \mathbf{x}'\|_2^2}{2\sigma^2} \right)$$
2. **Homogeneous / Inhomogeneous Polynomial Kernel:**
   $$k_{\text{Poly}}(\mathbf{x}, \mathbf{x}') = \left( \frac{\mathbf{x}^\top \mathbf{x}'}{D} + c \right)^d$$
3. **Matérn Class Kernel ($\nu = 3/2, 5/2$):**
   $$k_{\text{Matérn}, 3/2}(r) = \left(1 + \frac{\sqrt{3}r}{\ell}\right) \exp\left(-\frac{\sqrt{3}r}{\ell}\right), \quad r = \|\mathbf{x} - \mathbf{x}'\|_2$$

### 2.3 Kernel Ridge Regression (KRR)
For continuous regression targets $\mathbf{y} \in \mathbb{R}^N$, minimizing regularized squared error in RKHS $\mathcal{H}$ yields:
$$\min_{f \in \mathcal{H}} \frac{1}{2} \sum_{i=1}^N (y_i - f(\mathbf{x}_i))^2 + \frac{\alpha}{2} \|f\|_{\mathcal{H}}^2 \implies \boldsymbol{\alpha}^* = (\mathbf{K} + \alpha \mathbf{I}_N)^{-1} \mathbf{y}$$
Predictions at novel query point $\mathbf{x}_*$ evaluate via:
$$\hat{f}(\mathbf{x}_*) = \sum_{i=1}^N \alpha_i^* k(\mathbf{x}_i, \mathbf{x}_*) = \mathbf{k}_*^\top (\mathbf{K} + \alpha \mathbf{I}_N)^{-1} \mathbf{y}$$

### 2.4 Nyström Low-Rank Approximation
When $N \gg 10^4$, computing and inverting full $\mathbf{K} \in \mathbb{R}^{N \times N}$ requires $\mathcal{O}(N^2)$ memory and $\mathcal{O}(N^3)$ operations. Choosing $m \ll N$ landmark points partitions the kernel matrix:
$$\mathbf{K} \approx \begin{bmatrix} \mathbf{K}_{mm} & \mathbf{K}_{nm}^\top \\ \mathbf{K}_{nm} & \mathbf{K}_{nn} \end{bmatrix} \implies \tilde{\mathbf{K}} = \mathbf{K}_{nm} \mathbf{K}_{mm}^{-1} \mathbf{K}_{nm}^\top$$
This reduces inversion complexity to $\mathcal{O}(m^3 + N m^2)$ with $\mathcal{O}(N m)$ storage.

---

## 3. Tensor Mechanics & Compute Graphs

```
[Input X: N x D] ──┬──> [Landmarks X_m: m x D] ──> [K_mm = exp(-gamma ||x_m - x_m||^2)] ──> [Cholesky / Invert: m x m]
                   │                                                                                    │
                   └──> [Cross Gram K_nm: N x m] ───────────────────────────────────────────────────────┴──> [Nystrom Gram: N x N]
```

### 3.1 Pairwise Squared Distance GEMM Expansion
Computing pairwise Euclidean distances naïvely requires $\mathcal{O}(N^2 D)$ elementwise operations. Expanding the inner binomial identity vectorizes the operation entirely into Tensor Core matrix multiplications:
$$\|\mathbf{x}_i - \mathbf{x}_j\|_2^2 = \|\mathbf{x}_i\|_2^2 + \|\mathbf{x}_j\|_2^2 - 2 \mathbf{x}_i^\top \mathbf{x}_j$$
```python
x1_sq = torch.sum(X1 ** 2, dim=1, keepdim=True)       # [N, 1]
x2_sq = torch.sum(X2 ** 2, dim=1, keepdim=True).T     # [1, M]
dist_sq = torch.clamp(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), min=0.0)
K = torch.exp(-gamma * dist_sq)
```

---

## 4. Multi-Ecosystem Implementation Parity
The implementation is verified across four execution platforms:
- **PyTorch 2.x (`pytorch2/`):** Dynamic tensor dispatch with `torch.linalg.solve` and subgradient primal soft-margin SVM.
- **JAX (`jax/`):** Pure functional JIT-compiled kernels using `jax.numpy.linalg.solve` and parallel distance reductions.
- **Native Rust Candle (`candle/`):** Pure Rust implementation using in-place Cholesky decomposition ($L L^\top = A$) for machine-precision dual coefficient solving.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shaders parallelizing $16 \times 16$ 2D workgroups over pairwise distance tiles.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Quadratic VRAM OOM** | Allocating full $N \times N$ Gram matrix for $N \ge 50\,000$ | Memory allocation abort | Nyström landmark approximation or Random Fourier Features ($m \ll N$) |
| **Negative Distance NaN** | Floating-point cancellation in $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ expansion | Negative epsilon before $\sqrt{\cdot}$ yielding `NaN` | Explicit clamp: `dist_sq.clamp(min=0.0)` |
| **Gram Matrix Singularity** | Duplicated rows or high feature collinearity | `torch.linalg.solve` numerical failure | Tikhonov jitter: $\mathbf{K} + \epsilon \mathbf{I}$ with $\epsilon \ge 10^{-5}$ |
| **RBF Saturation / Delta Collapse** | Unnormalized features or unscaled $\gamma$ parameter | Gram matrix collapses to identity $\mathbf{I}$ or all-ones | Standardize inputs ($Z$-score) and set default $\gamma = 1/D$ |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Transformer Attention as Kernel Smoothing:** Self-attention computes $A = \operatorname{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$. This is mathematically equivalent to Nadaraya-Watson kernel regression with an exponential dot-product kernel.
2. **Deep Gaussian Processes:** Kernel Ridge Regression dual solution $\hat{f}(x_*) = \mathbf{k}_*^\top (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y}$ corresponds exactly to the posterior mean of a Gaussian Process with likelihood $\mathcal{N}(0, \sigma^2)$.
3. **Implicit Bias in Neural Networks:** Unregularized gradient descent on separable data converges in direction to the maximum-margin hard-margin SVM hyperplane (Soudry et al., 2018).

---

## 7. Verification & Parity Results
All numerical routines satisfy the strict multi-track parity contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-7}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-5}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path25"`: **PASS**
- `cargo test -p ai-candle --test 02_classical_and_probabilistic_ml`: **PASS (2/2 tests)**
- `cargo test -p ai-kernel --test 02_classical_and_probabilistic_ml`: **PASS (2/2 tests)**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [01_generalized_linear_models_and_proximal_lasso.md](01_generalized_linear_models_and_proximal_lasso.md)
- **Current Topic:** 02_support_vector_machines_and_kernel_methods.md
- **Next Topic:** [03_knn_prototype_and_locally_weighted_regression.md](03_knn_prototype_and_locally_weighted_regression.md)
- **Category Index:** [Master Curriculum Index](../../README.md)
