# 05. Dimensionality Reduction: PCA, LDA, ICA, t-SNE, and UMAP

## 1. Executive Summary
High-dimensional representation spaces pose profound computational and statistical hurdles, known collectively as the *Curse of Dimensionality*. Dimensionality reduction discovers compact, low-dimensional coordinate systems that preserve maximal variance, optimize class discriminability, separate statistically independent non-Gaussian sources, or preserve neighborhood topological manifolds. This topic unifies linear projection methods (Pearson, 1901 PCA; Fisher, 1936 LDA; Hyvärinen, 1999 FastICA) with non-linear kernel extensions (Schölkopf et al., 1998 Kernel PCA) and modern manifold learning techniques (van der Maaten & Hinton, 2008 $t$-SNE; McInnes et al., 2018 UMAP) across PyTorch 2.x, JAX, native Rust Candle, and WebGPU WGSL compute shaders.

---

## 2. Mathematical Foundations

### 2.1 Principal Component Analysis (PCA) via SVD
Given centered observations $\tilde{\mathbf{X}} = \mathbf{X} - \frac{1}{N}\mathbf{1}\mathbf{1}^\top \mathbf{X} \in \mathbb{R}^{N \times D}$, the sample covariance matrix is $\mathbf{\Sigma} = \frac{1}{N-1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}$. PCA seeks orthogonal projection directions $\mathbf{v} \in \mathbb{R}^D$ maximizing variance:
$$\max_{\|\mathbf{v}\|_2 = 1} \mathbf{v}^\top \mathbf{\Sigma} \mathbf{v} \iff \mathbf{\Sigma} \mathbf{v} = \lambda \mathbf{v}$$
Directly computing Singular Value Decomposition (SVD) on the data matrix $\tilde{\mathbf{X}} = \mathbf{U} \mathbf{S} \mathbf{V}^\top$ circumvents squaring the condition number ($\kappa(\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}) = \kappa(\tilde{\mathbf{X}})^2$):
$$\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{V}_{:, 1:K} = \mathbf{U}_{:, 1:K} \mathbf{S}_{1:K}, \quad \lambda_k = \frac{S_k^2}{N - 1}$$
> [!WARNING]
> **Omission of Mean-Centering:** Performing SVD on uncentered data forces the first principal component to align with the global empirical mean $\bar{\mathbf{x}}$, completely contaminating the eigenspectrum with spurious offset bias.

### 2.2 Fisher's Linear Discriminant Analysis (LDA)
For labeled classification data, Fisher's criterion maximizes between-class variance $\mathbf{S}_B$ relative to within-class variance $\mathbf{S}_W$:
$$\mathcal{J}(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}, \quad \mathbf{S}_W = \sum_{c=1}^C \sum_{i \in \mathcal{C}_c} (\mathbf{x}_i - \boldsymbol{\mu}_c)(\mathbf{x}_i - \boldsymbol{\mu}_c)^\top$$
For binary classification ($C=2$), the optimal projection direction is given in closed form:
$$\mathbf{w}^* = (\mathbf{S}_W + \epsilon \mathbf{I})^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$$

### 2.3 Kernel PCA with Double-Centering
Kernel PCA maps data into an RKHS $\phi(\mathbf{x})$ and solves the eigensystem of the centered Gram matrix $\tilde{\mathbf{K}}$:
$$\tilde{\mathbf{K}} = \left(\mathbf{I} - \frac{1}{N}\mathbf{1}\mathbf{1}^\top\right) \mathbf{K} \left(\mathbf{I} - \frac{1}{N}\mathbf{1}\mathbf{1}^\top\right) = \mathbf{K} - \frac{1}{N}\mathbf{1}\mathbf{1}^\top\mathbf{K} - \frac{1}{N}\mathbf{K}\mathbf{1}\mathbf{1}^\top + \frac{1}{N^2}\mathbf{1}\mathbf{1}^\top\mathbf{K}\mathbf{1}\mathbf{1}^\top$$
$$\tilde{\mathbf{K}} \boldsymbol{\alpha}_k = \lambda_k \boldsymbol{\alpha}_k \implies \hat{\boldsymbol{\alpha}}_k = \frac{\boldsymbol{\alpha}_k}{\sqrt{\lambda_k}}, \quad z_k(\mathbf{x}_*) = \sum_{i=1}^N \hat{\alpha}_{ki} \tilde{k}(\mathbf{x}_i, \mathbf{x}_*)$$

### 2.4 t-SNE & UMAP Non-Linear Manifold Projection
- **t-SNE:** Symmetrized high-dimensional affinities $p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \ne i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$ are matched to low-dimensional Student-$t$ distribution probabilities $q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$ by minimizing Kullback-Leibler divergence $\mathcal{D}_{\text{KL}}(P \| Q)$. The heavy Cauchy tails solve the "crowding problem".
- **UMAP:** Formulates fuzzy simplicial sets and minimizes binary cross-entropy, preserving global continuum structure substantially better than $t$-SNE.

---

## 3. Tensor Mechanics & Compute Graphs

```
[Raw Input X: N x D] ──> [Column Mean: 1 x D] ──> [Centered Matrix: N x D] ──> [Thin SVD: U, S, V^T]
                                                                                       │
                                                                                       ▼
                                                                           [V_top = V^T[0:K, :]^T]
                                                                                       │
                                                                                       ▼
                                                                         [Projection: Z = X_tilde @ V_top]
```

---

## 4. Multi-Ecosystem Implementation Parity
The implementation is verified across four execution environments:
- **PyTorch 2.x (`pytorch2/`):** Batch tensor operations using `torch.linalg.svd` and `torch.linalg.eigh`.
- **JAX (`jax/`):** Functional JIT-compiled transforms with static component arguments and vector dot products.
- **Native Rust Candle (`candle/`):** Pure Rust implementation using in-place Cholesky decomposition for LDA and SVD projection matrices.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader parallelizing 2D workgroups over rows and dimensions for fused mean centering.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Offset Domination in PCA** | Performing SVD without mean-centering ($\bar{\mathbf{x}} \ne \mathbf{0}$) | 1st component points directly at dataset centroid | Subtract column means $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{x}}$ |
| **Within-Class Singularity in LDA** | Sample size $N < D$ makes $\mathbf{S}_W$ rank-deficient | Inversion failure / numerical explosion | Add Tikhonov ridge regularizer: $\mathbf{S}_W + \epsilon \mathbf{I}$ |
| **Double-Centering Drift in KPCA** | Uncentered Gram matrix produces un-normalized RKHS vectors | Principal components confound mean shift with variance | Apply explicit double-centering: $\mathbf{H} \mathbf{K} \mathbf{H}$ |
| **Crowding Collapse in Manifold Maps** | Gaussian low-dimensional kernel in high dimension embedding | All points condense into single uninformative blob | Use Student-$t$ heavy tail (Cauchy distribution in $t$-SNE) |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Low-Rank Adaptation (LoRA):** Linear low-rank weight updates $\Delta W = B A$ ($B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$) mirror PCA low-rank factorizations of downstream weight delta manifolds.
2. **Latent Space Visualization:** Modern diffusion models (Stable Diffusion, Imagen) and VAEs visualize latent geometry and manifold trajectories via PCA and UMAP projections.
3. **Contrastive Feature Compression:** Representation learning heads (e.g., in SimCLR, CLIP) use linear bottleneck layers trained with InfoNCE loss that naturally emulate supervised Fisher LDA objectives.

---

## 7. Verification & Parity Results
All numerical routines satisfy the strict multi-track parity contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path28"`: **PASS**
- `cargo test -p ai-candle --test 02_classical_and_probabilistic_ml`: **PASS (5/5 tests)**
- `cargo test -p ai-kernel --test 02_classical_and_probabilistic_ml`: **PASS (5/5 tests)**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [04_clustering_kmeans_spectral_and_gmm_em.md](04_clustering_kmeans_spectral_and_gmm_em.md)
- **Current Topic:** 05_dimensionality_reduction_pca_lda_ica_tsne_umap.md
- **Next Topic:** [06_naive_bayes_bayesian_regression_and_gaussian_processes.md](06_naive_bayes_bayesian_regression_and_gaussian_processes.md)
- **Category Index:** [02_classical_and_probabilistic_ml/README.md](README.md)
