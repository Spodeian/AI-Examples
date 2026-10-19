# 03. k-Nearest Neighbors, Prototype Learning, and Locally Weighted Regression

## 1. Executive Summary
Nonparametric instance-based methods formulate predictions directly from local neighborhoods in sample space without committing to rigid global parametric functional forms. Grounded in the asymptotic Bayes-risk bounds of Cover and Hart (1967), $k$-Nearest Neighbors ($k$-NN) and locally weighted kernel regression (Nadaraya, 1964; Watson, 1964; Cleveland, 1979 LOESS) adapt dynamically to complex manifolds. This topic unifies vectorized Tensor Core distance matrix evaluations ($\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2 \mathbf{x}^\top \mathbf{y}$), distance-weighted neighborhood voting, Nearest Centroid & Learning Vector Quantization (LVQ), and Nadaraya-Watson kernel regression across PyTorch 2.x, JAX, native Rust Candle, and WebGPU WGSL compute shaders.

---

## 2. Mathematical Foundations

### 2.1 The Asymptotic Bayes Risk of 1-NN
Let $\mathcal{X} \subset \mathbb{R}^D$ and $Y \in \{1, \dots, C\}$ with prior class distributions $p(y)$ and class-conditional densities $p(\mathbf{x}|y)$. The Bayes-optimal decision rule achieves minimum misclassification error $R^*$:
$$R^* = 1 - \int_{\mathcal{X}} \max_{c \in \{1,\dots,C\}} p(c|\mathbf{x}) p(\mathbf{x}) d\mathbf{x}$$
Cover and Hart (1967) proved that as sample size $N \to \infty$, the asymptotic error of the 1-nearest neighbor classifier $R_{1\text{-NN}}$ satisfies tight bounds relative to the theoretical Bayes risk:
$$R^* \le R_{1\text{-NN}} \le 2 R^* (1 - R^*) \le 2 R^*$$
For $k$-NN with $k \to \infty$ and $k/N \to 0$, $R_{k\text{-NN}} \to R^*$, achieving non-parametric Bayes optimality.

### 2.2 Distance Metric & GEMM Expansion
Computing pairwise Euclidean distances naïvely between $N_1$ training samples and $N_2$ query vectors requires looping over all coordinate dimensions. Expanding the binomial Euclidean norm vectorizes the entire computation into matrix multiplication:
$$\|\mathbf{x}_i - \mathbf{x}_j\|_2^2 = \|\mathbf{x}_i\|_2^2 + \|\mathbf{x}_j\|_2^2 - 2 \mathbf{x}_i^\top \mathbf{x}_j$$
$$\mathbf{D}^2 = \mathbf{1}_{N_2}^\top \otimes \operatorname{diag}(\mathbf{X}_1 \mathbf{X}_1^\top) + \operatorname{diag}(\mathbf{X}_2 \mathbf{X}_2^\top)^\top \otimes \mathbf{1}_{N_1} - 2 \mathbf{X}_1 \mathbf{X}_2^\top$$
> [!CAUTION]
> **Catastrophic Floating-Point Cancellation:** In finite-precision IEEE 754 float32, identical vectors $\mathbf{x}_i = \mathbf{x}_j$ can evaluate to small negative values (e.g., $-10^{-7}$) due to rounding errors during subtraction. Calling `sqrt(dist_sq)` without `clamp(min=0.0)` produces silent `NaN` values that propagate throughout backpropagation and inference.

### 2.3 Distance-Weighted k-NN Regression
Given query point $\mathbf{x}_*$ and its $k$ nearest neighbors $\{\mathbf{x}_{(i)}, y_{(i)}\}_{i=1}^k$ with sorted Euclidean distances $d_{(i)} = \|\mathbf{x}_* - \mathbf{x}_{(i)}\|_2$:
$$\hat{y}(\mathbf{x}_*) = \frac{\sum_{i=1}^k w_{(i)} y_{(i)}}{\sum_{i=1}^k w_{(i)}}, \quad w_{(i)} = \frac{1}{d_{(i)} + \epsilon}$$

### 2.4 Nadaraya-Watson Kernel Regression
Instead of hard top-$k$ cutoffs, Nadaraya-Watson regression applies smooth radial kernel weighting across the entire training dataset:
$$\hat{m}(\mathbf{x}_*) = \frac{\sum_{i=1}^N K_h(\mathbf{x}_* - \mathbf{x}_i) y_i}{\sum_{i=1}^N K_h(\mathbf{x}_* - \mathbf{x}_i)}, \quad K_h(\mathbf{u}) = \exp\left(-\frac{\|\mathbf{u}\|_2^2}{2 h^2}\right)$$
As bandwidth $h \to 0$, $\hat{m}(\mathbf{x})$ interpolates training points; as $h \to \infty$, $\hat{m}(\mathbf{x})$ converges to the global sample mean $\bar{y}$.

---

## 3. Tensor Mechanics & Compute Graphs

```
[Train X: N1 x D] ──┬──> [Row Norms: ||x1||^2: N1 x 1] ──┐
                    │                                    ▼
                    ├──> [GEMM: -2 X1 X2^T: N1 x N2] ──> [Pairwise D^2: N1 x N2] ──> [Clamp(0.0)] ──> [sqrt: D]
                    │                                    ▲
[Query X: N2 x D] ──┴──> [Col Norms: ||x2||^2: 1 x N2] ──┘                                               │
                                                                                                          ▼
                                                                                              [Top-k Neighbor Gather]
                                                                                                          │
                                                                                                          ▼
                                                                                              [Weighted Average Target]
```

---

## 4. Multi-Ecosystem Implementation Parity
The implementation is verified across four execution environments:
- **PyTorch 2.x (`pytorch2/`):** Batch tensor operations using `torch.topk` with fused clamping.
- **JAX (`jax/`):** Pure functional JIT-compiled kernels with vectorized pairwise matrix reductions.
- **Native Rust Candle (`candle/`):** Memory-safe pure Rust implementation with in-place top-$k$ partial sorting.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader executing parallel $16 \times 16$ tile reductions with memory-safe clamping.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Square Root NaN Collapse** | IEEE 754 cancellation yields small negative values in $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ | `NaN` in distance matrix and gradients | Always apply `.clamp(min=0.0)` before calling `sqrt()` |
| **Curse of Dimensionality** | As $D \to \infty$, distance ratio $\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$ | All neighbors become equidistant | Dimensionality reduction (PCA/UMAP) or metric learning |
| **Division by Zero at Exact Match** | Distance weight $1/d_i \to \infty$ when query equals training point | Division by zero `Inf`/`NaN` | Add epsilon stabilizer: $w_i = 1 / (d_i + 10^{-5})$ |
| **Bandwidth Underflow** | Small bandwidth $h \ll \min_{i \ne j} \|\mathbf{x}_i - \mathbf{x}_j\|$ in Nadaraya-Watson | Kernel sum $\sum K_h \to 0 \implies 0/0 = \text{NaN}$ | Clamp denominator to $\max(\sum K_h, 10^{-8})$ |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Transformer Attention as Soft Nearest Neighbors:** Scaled dot-product attention $\operatorname{softmax}(Q K^\top / \sqrt{d}) V$ is an asymmetric Nadaraya-Watson kernel smoother where queries retrieve value vectors weighted by key similarity.
2. **Episodic Memory & Retrieval Augmented Generation (RAG):** Modern vector databases (HNSW, ScaNN) execute approximate $k$-NN search over dense embeddings to ground generative models.
3. **Local Linear Embedding & Manifold Learning:** Manifold learning techniques (Isomap, LLE) construct adjacency graphs directly from $k$-NN graphs.

---

## 7. Verification & Parity Results
All numerical routines satisfy the strict multi-track parity contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-6}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path26"`: **PASS**
- `cargo test -p ai-candle --test 02_classical_and_probabilistic_ml`: **PASS (3/3 tests)**
- `cargo test -p ai-kernel --test 02_classical_and_probabilistic_ml`: **PASS (3/3 tests)**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [02_support_vector_machines_and_kernel_methods.md](02_support_vector_machines_and_kernel_methods.md)
- **Current Topic:** 03_knn_prototype_and_locally_weighted_regression.md
- **Next Topic:** [04_clustering_kmeans_spectral_and_gmm_em.md](04_clustering_kmeans_spectral_and_gmm_em.md)
- **Category Index:** [02_classical_and_probabilistic_ml/README.md](README.md)
