# AI Analysis: Loss Landscape Geometry, Hessian Lanczos SLQ, and Empirical NTK

## 1. Mathematical Foundations & Lineage

The optimization geometry of neural networks governs training dynamics, speed of convergence, and generalization capability:
1. **Filter-Normalized Loss Landscape Slicing (Li et al. 2018)**: Demonstrates that visualizing neural loss surfaces along random directions without normalization yields spurious sharpness artifacts because deep networks with Batch Normalization exhibit parameter scale invariance ($W \to \alpha W$). Filter-wise normalization scales perturbing vectors by filter Frobenius norms, restoring scale invariance.
2. **Hessian Spectral Estimation via Lanczos Quadrature (Ghorbani et al. 2019; Yao et al. 2020 / PyHessian)**: The full Hessian $H = \nabla_\theta^2 L \in \mathbb{R}^{P \times P}$ is intractable to instantiate for modern networks ($P > 10^7$). Stochastic Lanczos Quadrature (SLQ) uses Hessian-Vector Products (HVP) to compute a compact $m \times m$ tridiagonal matrix whose Ritz spectrum closely approximates the true Hessian spectral density $\rho(\lambda)$.
3. **Empirical Neural Tangent Kernel (eNTK — Jacot, Gabriel, & Hongler 2018; Lee et al. 2019)**: In the infinite-width limit, overparameterized neural networks trained with gradient descent converge along linear trajectories governed by a static kernel $\Theta_\infty$. In finite-width models, tracking the empirical NTK Gram matrix $\Theta = J J^T$ identifies whether a model operates in the lazy "kernel regime" or the adaptive "feature learning / rich regime".

---

## 2. Filter-Normalized Loss Landscape Visualization

Given model parameters $\theta$ and random Gaussian direction vectors $\delta, \eta \sim \mathcal{N}(0, I)$, naive 1D or 2D slices $\theta(\alpha, \beta) = \theta + \alpha \delta + \beta \eta$ produce arbitrary apparent sharpness depending on the arbitrary norm of $\theta$.

Filter-wise normalization rescales each filter $j$ in layer $l$:
$$\delta_{l, j} \leftarrow \delta_{l, j} \cdot \frac{\| \theta_{l, j} \|_F}{\| \delta_{l, j} \|_F + \epsilon}, \quad \eta_{l, j} \leftarrow \eta_{l, j} \cdot \frac{\| \theta_{l, j} \|_F}{\| \eta_{l, j} \|_F + \epsilon}$$
The loss surface coordinate is evaluated as:
$$\theta(\alpha, \beta) = \theta + \alpha \delta + \beta \eta$$
This normalization ensures that multiplying a layer's weights by a scalar $c$ (compensated by $1/c$ elsewhere) does not alter the visualized curvature.

---

## 3. Stochastic Lanczos Quadrature (SLQ) & Hessian Tridiagonalization

Let $H \in \mathbb{R}^{P \times P}$ be the loss Hessian. Given an initial normalized probe vector $q_1 = v_0 / \|v_0\|_2$, the Lanczos algorithm constructs an orthonormal Krylov basis $\{q_1, \dots, q_m\}$ and a symmetric tridiagonal matrix $T_m$:
$$T_m = \begin{pmatrix} \alpha_1 & \beta_1 & 0 & \dots \\ \beta_1 & \alpha_2 & \beta_2 & \dots \\ 0 & \beta_2 & \alpha_3 & \dots \\ \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

At step $j$:
1. Compute matrix-vector product $w_j = H q_j$.
2. Orthogonalize against previous basis: $w_j \leftarrow w_j - \beta_{j-1} q_{j-1}$ (for $j > 1$).
3. Diagonal entry: $\alpha_j = \langle q_j, w_j \rangle$.
4. Residual: $w_j \leftarrow w_j - \alpha_j q_j$.
5. Sub-diagonal entry: $\beta_j = \| w_j \|_2$.
6. Next basis vector: $q_{j+1} = w_j / \beta_j$.

The eigenvalues of $T_m$ (Ritz values) approximate the top eigenvalues of $H$, and Gaussian quadrature on $T_m$ estimates the spectral density $\rho(\lambda) = \frac{1}{P} \sum_{i=1}^P \delta(\lambda - \lambda_i)$.

---

## 4. Empirical Neural Tangent Kernel (eNTK)

For a dataset of $N$ evaluation instances $\{x_i\}_{i=1}^N$ with scalar model predictions $f(x_i; \theta)$, the per-example gradient (Jacobian) is:
$$J_i = \nabla_\theta f(x_i; \theta) \in \mathbb{R}^{1 \times P}$$
The empirical NTK Gram matrix $\Theta \in \mathbb{R}^{N \times N}$ is the inner product of parameter gradients:
$$\Theta_{ij} = \langle J_i, J_j \rangle = J_i J_j^T$$
In matrix form:
$$\Theta = J J^T \in \mathbb{R}^{N \times N}$$
where $J \in \mathbb{R}^{N \times P}$. The condition number and trace of $\Theta$ govern gradient descent convergence rates under quadratic loss: $\frac{d f(X)}{dt} \approx -\Theta (f(X) - y)$.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`filter_normalize_direction`** | Per-filter norm ratio + broadcast mul | Array norm reduction + broadcast | 2D vector iteration + row scaling | Filter norm scaling shader |
| **`loss_surface_coord`** | Linear combination $\theta + \alpha d_1 + \beta d_2$ | Pure array arithmetic | `affine` scaling + tensor addition | Vectorized coordinate slicer |
| **`lanczos_tridiagonal`** | Krylov vector recurrence | Sequential vector iteration | Vectorized matrix-vector + projection | Lanczos HVP step kernel |
| **`empirical_ntk`** | $J J^T$ matrix multiplication | `jacobians @ jacobians.T` | Tensor `matmul` with transpose | `EMPIRICAL_NTK_WGSL` |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Unnormalized Direction Illusion**:
   - *Symptom*: A model with BatchNorm appears to have a razor-sharp loss landscape, yet trains smoothly with high learning rates.
   - *Cause*: Unnormalized random directions sample disproportionately large relative perturbations in small-weight layers.
   - *Fix*: Always apply filter-normalization scaling direction vectors by $\|\theta_{l, j}\|$.
2. **Loss of Orthogonality in Lanczos**:
   - *Symptom*: Ghost eigenvalues appear in the estimated Hessian spectrum after $m > 20$ iterations.
   - *Cause*: Finite floating-point precision causes roundoff errors that destroy mutual orthogonality among Krylov vectors $q_j$.
   - *Fix*: Keep $m \le 15$ for fast spectral bounds, or apply selective partial reorthogonalization.
3. **NTK Rank Degeneracy on Duplicate Inputs**:
   - *Symptom*: Empirical NTK matrix has zero eigenvalues, causing unstable kernel ridge regression.
   - *Cause*: Duplicate or linearly dependent samples produce identical gradient directions.
   - *Fix*: Add a small Tikhonov ridge regularization $\Theta \leftarrow \Theta + \lambda I_N$ ($\lambda \approx 10^{-4}$).

---

## 7. Real-World Engineering Applications

1. **Architecture Loss Flatness Comparison**: Quantifying why residual connections (ResNet) flatten the loss landscape compared to plain feedforward networks.
2. **Edge-of-Stability Diagnostics**: Tracking top Hessian eigenvalue $\lambda_{\max}(H)$ over training to detect when gradient descent approaches the stability threshold $2/\eta$.
3. **Feature Learning vs. Lazy Regime Auditing**: Measuring the relative change $\|\Theta(t) - \Theta(0)\|_F / \|\Theta(0)\|_F$ during fine-tuning to diagnose whether weights adapt or remain in the kernel regime.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `ls_d1_norm`, `ls_d2_norm`: Filter-normalized perturbation direction vectors.
3. `ls_slice_ref`: 2D loss surface parameter coordinate slice.
4. `lanczos_alpha_ref`, `lanczos_beta_ref`: Tridiagonal Lanczos diagonal and sub-diagonal vectors.
5. `ntk_gram_ref`: Empirical NTK Gram matrix $J J^T$.
