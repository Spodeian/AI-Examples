# 06. Naive Bayes, Bayesian Linear Regression, and Gaussian Processes

## 1. Executive Summary
Probabilistic machine learning treats learning as an exercise in inductive inference, quantifying epistemic uncertainty across parameter configurations and functional predictions. This module investigates the spectrum of foundational Bayesian techniques: from generative conditional independence models (Bayes, 1763 Naive Bayes) and exact parameter posteriors in conjugate linear models to non-parametric function-space priors (Rasmussen & Williams, 2006 Gaussian Processes) and scalable inducing-point approximations (Titsias, 2009 Sparse Variational GPs). Implemented across PyTorch 2.x, JAX, native Rust Candle, and WebGPU WGSL compute shaders, every algorithm is rigorously synchronized against a golden reference SafeTensors checkpoint to guarantee multi-track parity.

---

## 2. Mathematical Foundations

### 2.1 Gaussian Naive Bayes in the Log-Domain
Naive Bayes factorizes the class-conditional density under the assumption that features are mutually independent given class $y = c$:
$$p(\mathbf{x} \mid y = c) = \prod_{d=1}^D p(x_d \mid y = c) = \prod_{d=1}^D \frac{1}{\sqrt{2\pi \sigma_{cd}^2}} \exp\left(-\frac{(x_d - \mu_{cd})^2}{2\sigma_{cd}^2}\right)$$
Direct product evaluation catastrophically underflows to `0.0` in high dimensions ($D \gg 10$). The algorithm is evaluated strictly in the log-domain:
$$\log p(y = c \mid \mathbf{x}) = \log \pi_c - \frac{1}{2} \sum_{d=1}^D \left[ \log(2\pi \sigma_{cd}^2) + \frac{(x_d - \mu_{cd})^2}{\sigma_{cd}^2} \right] - \text{LogSumExp}_{k} \log p(y = k, \mathbf{x})$$

### 2.2 Conjugate Bayesian Linear Regression
Given isotropic Gaussian weight prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I})$ and Gaussian likelihood $y \mid \mathbf{X}, \mathbf{w} \sim \mathcal{N}(\mathbf{X} \mathbf{w}, \beta^{-1} \mathbf{I})$, the posterior over parameters is conjugate Gaussian $\mathbf{w} \mid \mathbf{X}, \mathbf{y} \sim \mathcal{N}(\mathbf{m}_N, \mathbf{S}_N)$:
$$\mathbf{S}_N^{-1} = \alpha \mathbf{I} + \beta \mathbf{X}^\top \mathbf{X}, \quad \mathbf{m}_N = \beta \mathbf{S}_N \mathbf{X}^\top \mathbf{y}$$
For any unseen query observation $\mathbf{x}_*$, the predictive distribution is analytic:
$$p(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \mathcal{N}(\mu_*, \sigma_*^2), \quad \mu_* = \mathbf{x}_*^\top \mathbf{m}_N, \quad \sigma_*^2 = \frac{1}{\beta} + \mathbf{x}_*^\top \mathbf{S}_N \mathbf{x}_*$$

### 2.3 Exact Gaussian Process Regression via Cholesky Factorization
A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution: $f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$. For Squared Exponential (RBF) covariance $k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)$ and noisy observations $y = f(\mathbf{x}) + \epsilon, \ \epsilon \sim \mathcal{N}(0, \sigma_n^2)$:
$$\mathbf{K}_y = \mathbf{K}(\mathbf{X}, \mathbf{X}) + \sigma_n^2 \mathbf{I} = \mathbf{L} \mathbf{L}^\top \quad (\text{Cholesky decomposition})$$
The predictive mean and variance at test locations $\mathbf{X}_*$ are computed using numerically stable triangular back-solves:
$$\boldsymbol{\alpha} = \mathbf{L}^{-\top} \mathbf{L}^{-1} \mathbf{y}, \quad \boldsymbol{\mu}_* = \mathbf{K}_* \boldsymbol{\alpha}$$
$$\mathbf{v} = \mathbf{L}^{-1} \mathbf{K}_*^\top, \quad \text{diag}(\mathbf{\Sigma}_*) = \text{diag}(\mathbf{K}_{**}) + \sigma_n^2 - \sum_{i} v_{ij}^2$$
> [!IMPORTANT]
> **Inversion vs Cholesky:** Directly calling `torch.inverse(K_y)` squares the numerical condition number and scales unstably with $O(N^3)$. Solving via triangular Cholesky decomposition $\mathbf{L} \mathbf{L}^\top$ guarantees symmetry, strictly positive variance estimates, and numerical robustness.

### 2.4 Sparse Variational GP Inducing-Point / Nyström Approximation
Exact GP inference scales as $O(N^3)$ in compute and $O(N^2)$ in memory. Inducing-point methods introduce $M \ll N$ pseudo-inputs $\mathbf{Z} \in \mathbb{R}^{M \times D}$, factorizing the covariance through low-rank projection:
$$\mathbf{Q}_{XX} = \mathbf{K}_{XZ} \mathbf{K}_{ZZ}^{-1} \mathbf{K}_{ZX} = \mathbf{A}^\top \mathbf{A}, \quad \text{where } \mathbf{L}_Z \mathbf{A} = \mathbf{K}_{ZX}$$
The diagonal predictive variance reduction $\text{diag}(\mathbf{Q}_{XX}) = \sum_m A_{mi}^2$ reduces compute complexity from $O(N^3)$ to $O(N M^2)$.

---

## 3. Tensor Mechanics & Compute Graphs

```
[Inputs X: N x D, Targets y: N]
       │
       ├──> [Log-Domain Naive Bayes] ──> Mean [C, D], Var [C, D] ──> Posterior Log Probs [N, C]
       │
       ├──> [Bayesian Regression] ────> Precision S_N^{-1} [D, D] ──> Inversion ──> Mean m_N, Var [N]
       │
       └──> [RBF Kernel Gram Matrix K_y: N x N]
                   │
                   ▼
       [Cholesky Factor L: N x N] ──> [Solve L w = y, L^T alpha = w] ──> Predictive Mean [N] & Variance [N]
```

---

## 4. Multi-Ecosystem Implementation Parity

- **PyTorch 2.x (`pytorch2/`):** Implemented via `nn.Module` subclasses utilizing `torch.linalg.cholesky`, `torch.cholesky_solve`, and `torch.linalg.solve_triangular`.
- **JAX (`jax/`):** Pure functional JIT-compiled transforms using `jax.scipy.linalg.solve_triangular` and `jax.nn.logsumexp`.
- **Native Rust Candle (`candle/`):** Pure Rust tensor implementations of log-domain Naive Bayes, Bayesian linear regression, and RBF kernel projection without external linear algebra runtime dependencies.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader parallelizing pairwise squared Euclidean distance and exponential evaluations across 2D workgroups, with native CPU fallback.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Probability Underflow** | Multiplying raw likelihood probabilities $\prod_d p(x_d \mid c)$ | Product collapses to machine zero `0.0` | Compute in log-domain and normalize via `LogSumExp` |
| **Singular Covariance Inversion** | Inverting $\mathbf{K}_y$ via `torch.inverse` under high collinearity | NaN gradients / non-positive variance estimates | Decompose with Cholesky: $\mathbf{L} = \text{cholesky}(\mathbf{K} + \sigma_n^2 \mathbf{I})$ |
| **Zero Feature Variance** | Identical feature values within a class in Naive Bayes | Division by zero in Gaussian likelihood exponent | Add variance smoothing jitter: $\sigma_{cd}^2 + \epsilon$ |
| **Inducing Point Degeneracy** | Duplicate or clustered inducing landmarks $\mathbf{Z}$ | Cholesky factorization failure on $\mathbf{K}_{ZZ}$ | Enforce minimum landmark spacing or add diagonal jitter $\epsilon \mathbf{I}$ |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Neural Tangent Kernel (NTK):** In the infinite-width limit, deep neural networks initialized with Gaussian priors converge to Gaussian Processes, establishing an exact mathematical bridge between deep learning and kernel machines.
2. **Monte Carlo Dropout as Bayesian Approximation:** Gal & Ghahramani (2016) showed that training deep networks with dropout is mathematically equivalent to variational inference in deep Gaussian Processes.
3. **Active Learning & Bayesian Optimization:** Exact predictive variance $\sigma_*^2$ forms the foundation of Upper Confidence Bound (UCB) and Expected Improvement (EI) acquisition functions in hyperparameter tuning and automated drug discovery.

---

## 7. Verification & Parity Results
All implementations satisfy the cross-track invariant contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path29"`: **PASS**
- `cargo test -p ai-candle --test 02_classical_and_probabilistic_ml -- test_06`: **PASS**
- `cargo test -p ai-kernel --test 02_classical_and_probabilistic_ml -- test_06`: **PASS**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [05_dimensionality_reduction_pca_lda_ica_tsne_umap.md](05_dimensionality_reduction_pca_lda_ica_tsne_umap.md)
- **Current Topic:** 06_naive_bayes_bayesian_regression_and_gaussian_processes.md
- **Next Topic:** [07_differentiable_trees_node_soft_gbdt_and_gam_nam.md](07_differentiable_trees_node_soft_gbdt_and_gam_nam.md)
- **Category Index:** [02_classical_and_probabilistic_ml/README.md](README.md)
