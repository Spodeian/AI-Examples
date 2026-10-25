# Randomized Numerical Linear Algebra (RandNLA), Spectral Autograd & Hutch++

**Category:** 00 Foundations  
**Topic Index:** 11 / 11 in Category 00 (Topic 11 of 263 Total)  
**Status:** Production Ready  
**Parity Verification:** Transitive Parity Across PyTorch 2.x, JAX, Candle, CPU Kernels ($\|Y_A - Y_B\|_\infty < 10^{-5}$)

---

## 1. Executive Summary & Theoretical Lineage

High-dimensional deep learning architectures, spectral normalizers, Hessian-free optimizers, and neural ODEs routinely require spectral operations—such as computing singular values, Schatten norms, trace estimations, and matrix factorizations—on large weight matrices $\mathbf{W} \in \mathbb{R}^{m \times n}$. Exact deterministic algorithms require $\mathcal{O}(m n \min(m, n))$ floating-point operations for SVD or $\mathcal{O}(n^3)$ for explicit matrix inversion and trace operations, rendering them computationally prohibitive.

This module formalizes and unifies three cornerstone algorithms from **Randomized Numerical Linear Algebra (RandNLA)**:
1. **Randomized SVD (rSVD) and Range Finders** (*Halko, Martinsson, & Tropp, 2011*): Compresses matrices into low-dimensional random sketches $\mathbf{Y} = \mathbf{A} \mathbf{\Omega}$, computes an orthonormal basis $\mathbf{Q} \in \mathbb{R}^{m \times (k+p)}$ via Modified Gram-Schmidt QR, and projects $\mathbf{A}$ into a small $(k+p) \times n$ subspace where standard SVD evaluates in $\mathcal{O}(k^2 n)$ time.
2. **Hutch++ Stochastic Trace Estimation** (*Meyer, Musco, Musco, & Woodruff, 2021*): Dramatically improves classical Hutchinson trace estimation (*Hutchinson, 1989*) from $\mathcal{O}(1/\epsilon^2)$ matrix-vector queries down to the optimal $\mathcal{O}(1/\epsilon)$ queries by combining low-rank subspace sketching with residual Monte Carlo queries.
3. **Lorentzian-Broadened Spectral Autograd** (*Giles, 2008; Seeger et al., 2017*): Resolves the catastrophic $1 / (\sigma_i - \sigma_j) \to \text{NaN}$ gradient explosion occurring in standard matrix backpropagation when singular values collide, replacing poles with smooth Cauchy-Lorentz broadening kernels.

```
+-----------------------------------------------------------------------------------------+
|                  RANDOMIZED NUMERICAL LINEAR ALGEBRA & SPECTRAL OPERATORS               |
|                                                                                         |
|  [ Matrix A (n x n) ]                                                                   |
|          |                                                                              |
|          +---> S Sketch (n x l) ----> Y = A S ---> MGS-QR ---> Basis Q (n x l)          |
|          |                                                          |                   |
|          |                                                          v                   |
|          |                                              tr(Q^T A Q) [Exact Low-Rank]    |
|          |                                                          +                   |
|          +---> G Rademacher (n x m) ---> Res = (I - Q Q^T) A G ---> (1/m) sum g_i^T r_i  |
|                                                                     v                   |
|                                                     Hutch++ Trace Estimate              |
|                                                     O(1/eps) Query Complexity           |
|                                                                                         |
|  [ Singular Values (sigma_i, sigma_j) ]                                                 |
|          |                                                                              |
|          +---> Collision (sigma_i == sigma_j)                                           |
|                   Standard Autograd:  F_ij = 1 / (sigma_j - sigma_i)   --> NaN / INF    |
|                   Lorentzian Cauchy:  F_ij = Delta / (Delta^2 + eps^2) --> 0.0 (STABLE) |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Architecture & Operator Invariants

### 2.1 Modified Gram-Schmidt (MGS) QR & Range Finding
Given sketch $\mathbf{Y} = \mathbf{A} \mathbf{\Omega} \in \mathbb{R}^{m \times l}$ with $l = k + p$:
$$\mathbf{v}_j = \mathbf{y}_j - \sum_{i=0}^{j-1} \langle \mathbf{q}_i, \mathbf{v}_j \rangle \mathbf{q}_i, \quad r_{jj} = \|\mathbf{v}_j\|_2, \quad \mathbf{q}_j = \frac{\mathbf{v}_j}{r_{jj}}$$
**Invariants:**
- Orthogonality: $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}_l$.
- Strictly positive diagonal: $r_{jj} > 0$, ensuring canonical deterministic phase across all target runtimes.

### 2.2 Hutch++ Stochastic Trace Estimator
For symmetric positive semi-definite $\mathbf{A} \in \mathbb{R}^{n \times n}$:
$$\hat{\operatorname{tr}}_{\text{Hutch++}}(\mathbf{A}) = \operatorname{tr}(\mathbf{Q}^\top \mathbf{A} \mathbf{Q}) + \frac{1}{m} \sum_{i=1}^m \mathbf{g}_i^\top (\mathbf{A} - \mathbf{Q}\mathbf{Q}^\top \mathbf{A}) \mathbf{g}_i$$
where $\mathbf{Q} = \operatorname{qr}(\mathbf{A}\mathbf{S})[0]$ for sketch $\mathbf{S} \in \mathbb{R}^{n \times l}$ and $\mathbf{g}_i \sim \operatorname{Rademacher}(\pm 1)$.
**Invariants:**
- Unbiased expectation: $\mathbb{E}[\hat{\operatorname{tr}}_{\text{Hutch++}}(\mathbf{A})] = \operatorname{tr}(\mathbf{A})$.
- Query reduction: Achieves $(1 \pm \epsilon)$ relative accuracy with $\mathcal{O}(1/\epsilon)$ matrix-vector queries, whereas standard Hutchinson requires $\mathcal{O}(1/\epsilon^2)$.

### 2.3 Lorentzian-Broadened Spectral Autograd
Standard SVD backpropagation involves the off-diagonal coupling matrix:
$$F_{ij} = \frac{1}{\sigma_j^2 - \sigma_i^2} \quad (i \neq j)$$
When eigenvalues or singular values degenerate ($\sigma_i \to \sigma_j$), $F_{ij}$ explodes to $\pm \infty$, producing `NaN` gradients. Under Lorentzian broadening with parameter $\epsilon > 0$:
$$\tilde{F}_{ij}^{(\epsilon)} = \begin{cases} \frac{\sigma_j - \sigma_i}{(\sigma_j - \sigma_i)^2 + \epsilon^2} & i \neq j \\ 0 & i = j \end{cases}$$
**Invariants:**
- Away from degeneracy ($|\sigma_j - \sigma_i| \gg \epsilon$): $\tilde{F}_{ij}^{(\epsilon)} \to \frac{1}{\sigma_j - \sigma_i}$.
- At collision ($\sigma_j = \sigma_i$): $\tilde{F}_{ij}^{(\epsilon)} = 0$, guaranteeing finite, non-exploding backpropagation.

---

## 3. Cross-Framework Implementation Blueprint

The implementation spans all systems tiers without external LAPACK variance:

1. **PyTorch 2.x:** [`pytorch2/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py`](../../pytorch2/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py)
   - `modified_gram_schmidt_qr`: Vectorized Gram-Schmidt QR.
   - `randomized_range_finder`: Randomized subspace projector.
   - `randomized_svd`: Halko-Martinsson-Tropp randomized SVD.
   - `hutch_plus_plus_trace`: Low-rank sketch + residual Monte Carlo trace estimator.
   - `lorentzian_broadened_f_matrix`: Broadened spectral autograd coupling matrix.
   - `SpectralLinear`: Production neural module with power iteration and Hutch++ Frobenius estimators.

2. **Functional JAX:** [`jax/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py`](../../jax/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py)
   - Stateless functional transforms with transparent CPU NumPy acceleration.

3. **Native Rust Candle:** [`candle/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.rs`](../../candle/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.rs)
   - Zero-allocation Rust implementation integrated into `ai-candle` with comprehensive unit and parity test suites.

4. **Hardware CPU Kernel:** [`kernels/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp_cpu.py`](../../kernels/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp_cpu.py)
   - Vectorized NumPy kernels implementing MGS-QR, Hutch++, and Lorentzian broadening.

---

## 4. Quantitative Verification & Parity Matrix

| Operation / Property | Reference Checkpoint | PyTorch 2.x Error | JAX Error | Candle Error | CPU Kernel Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MGS-QR Basis Matrix $\mathbf{Q}$** | Golden $\mathbf{Q}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **MGS-QR Upper Triangular $\mathbf{R}$** | Golden $\mathbf{R}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **Hutch++ Trace Estimate** | Golden Estimate | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **Power Iteration Spectral Norm** | Golden $\sigma_{\max}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **Lorentzian Coupling Matrix $\mathbf{F}$** | Golden $\mathbf{F}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **Forward Linear Prediction** | Golden Output | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **Gradient Stability on Degeneracy** | **NaN-Free** | **Finite** | **Finite** | **Finite** | **Finite** |

---

## 5. Cross-Domain Generalisation & Mathematical Isomorphisms

- **Physics & Dynamical Systems:** Shares structural operator invariance and conservation properties across continuous-to-discrete state mappings.
- **Signal Processing & Information Theory:** Acts as a bandwidth-preserving filter over high-dimensional representation manifolds.
- **Scientific Computing:** Enables hardware-aware vectorized execution with bounded numerical conditioning.

---

## 6. Curriculum Navigation & Synchronized Artifacts

- **Goal-Oriented Index:** [INDEX_BY_GOAL.md](../../INDEX_BY_GOAL.md)
- **Interactive Notebook Explorer:** [11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.ipynb](../../notebooks/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.ipynb)
- **Production PyTorch 2.x Module:** [11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py](../../pytorch2/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py)
- **Functional JAX Module:** [11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py](../../jax/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py)
- **Native Rust Candle Crate:** [11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.rs](../../candle/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.rs)
- **Hardware CPU Kernel:** [11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp_cpu.py](../../kernels/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp_cpu.py)
- **Golden Reference Checkpoint:** [11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.safetensors](../../weights/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.safetensors)
- **Previous Topic:** [10. Floating-Point Arithmetic, Conditioning & Numerical Stability](10_floating_point_arithmetic_conditioning_and_numerical_stability.md)
- **Next Topic:** [Category 01: 01. Generative vs. Discriminative Classifiers](../01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.md)


