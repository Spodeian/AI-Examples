# 04. Clustering: k-Means, Spectral Clustering, and GMM Expectation-Maximization

## 1. Executive Summary
Unsupervised clustering groups data points into coherent partitions without manual ground-truth labels. While $k$-Means (Lloyd, 1957; Arthur & Vassilvitskii, 2007 $k$-means++) identifies convex isotropic Voronoi cells, Gaussian Mixture Models (Dempster, Laird & Rubin, 1977 EM) model soft probabilistic cluster memberships and ellipsoidal covariances. For complex non-convex manifolds (such as concentric circles or intertwining spirals), Spectral Clustering (Ng, Jordan & Weiss, 2001) relaxes discrete graph partitioning into continuous eigenvector embeddings of the normalized graph Laplacian. This topic unifies vectorized Voronoi assignment, numerically stable log-domain GMM EM with diagonal jitter, and symmetric normalized graph Laplacians ($L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$) across PyTorch 2.x, JAX, native Rust Candle, and WebGPU WGSL compute shaders.

---

## 2. Mathematical Foundations

### 2.1 Lloyd's k-Means & Voronoi Partitions
Given dataset $\mathcal{X} = \{\mathbf{x}_i\}_{i=1}^N \subset \mathbb{R}^D$, $k$-Means minimizes total within-cluster sum of squares (inertia):
$$\mathcal{J}(\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K) = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2, \quad r_{ik} = \mathbb{I}\left(k = \arg\min_c \|\mathbf{x}_i - \boldsymbol{\mu}_c\|_2^2\right)$$
Lloyd's algorithm alternates between two steps:
1. **Assignment Step:** Compute distances $d_{ik} = \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$ and assign each sample to its nearest centroid.
2. **Update Step:** $\boldsymbol{\mu}_k = \frac{\sum_{i=1}^N r_{ik} \mathbf{x}_i}{\sum_{i=1}^N r_{ik}}$.

### 2.2 Gaussian Mixture Models (GMM) and Expectation-Maximization
A GMM models the data distribution as a convex combination of $K$ Gaussian densities:
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \quad \sum_{k=1}^K \pi_k = 1, \quad \pi_k \ge 0$$
Under diagonal covariance $\boldsymbol{\Sigma}_k = \operatorname{diag}(\boldsymbol{\sigma}_k^2)$, EM iterates:
- **E-Step (Posterior Responsibility in Log Domain):**
  $$\log \gamma_{ik} = \log \pi_k - \frac{1}{2} \left[ D \log(2\pi) + \sum_{j=1}^D \log \sigma_{kj}^2 + \sum_{j=1}^D \frac{(x_{ij} - \mu_{kj})^2}{\sigma_{kj}^2} \right]$$
  $$\gamma_{ik} = \frac{\exp(\log \gamma_{ik})}{\sum_{c=1}^K \exp(\log \gamma_{ic})} = \operatorname{softmax}_k(\log \boldsymbol{\gamma}_i)$$
- **M-Step (Parameter Updates with Diagonal Jitter):**
  $$N_k = \sum_{i=1}^N \gamma_{ik}, \quad \pi_k = \frac{N_k}{N}, \quad \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} \mathbf{x}_i$$
  $$\sigma_{kj}^2 = \max\left( \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_{ij} - \mu_{kj})^2, \; \epsilon_{\text{jitter}} \right)$$

### 2.3 Spectral Clustering & Graph Laplacians
To cluster non-linearly separable points, construct an undirected affinity graph with adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ where $A_{ij} = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|_2^2)$ for $i \ne j$ and $A_{ii} = 0$.
The degree matrix $\mathbf{D} = \operatorname{diag}(\mathbf{A} \mathbf{1})$ yields the Symmetric Normalized Graph Laplacian:
$$\mathbf{L}_{\text{sym}} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$
The bottom $K$ eigenvectors of $\mathbf{L}_{\text{sym}}$ embed arbitrary manifold shapes into an orthogonal $K$-dimensional space where standard $k$-Means successfully isolates disconnected clusters.

---

## 3. Tensor Mechanics & Compute Graphs

```
[Input X: N x D] ──> [Pairwise Dist Sq: N x N] ──> [Affinity A: exp(-gamma D^2)] ──> [Degree Matrix D = sum(A, dim=1)]
                                                                                               │
                                                                                               ▼
                                                                     [L_sym = I - D^{-1/2} A D^{-1/2}]
                                                                                               │
                                                                                               ▼
                                                                                   [Eigendecomposition]
```

---

## 4. Multi-Ecosystem Implementation Parity
The implementation is verified across four execution environments:
- **PyTorch 2.x (`pytorch2/`):** Batch tensor operations using `torch.logsumexp` and vectorized broadcasting.
- **JAX (`jax/`):** Functional JIT-compiled Lloyd iterations and log-domain GMM E-steps.
- **Native Rust Candle (`candle/`):** Zero-copy Candle tensors with fused log-domain softmax responsibilities.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader parallelizing 1D workgroups over sample points for nearest centroid finding.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **GMM Covariance Collapse** | Single point dominates component $k$ ($\boldsymbol{\mu}_k \to \mathbf{x}_i, \sigma_k \to 0$) | $\mathcal{L} \to +\infty$ and variance underflow `NaN` | Diagonal covariance jitter: $\sigma^2 \leftarrow \max(\sigma^2, 10^{-4})$ |
| **Non-Convex Ring Failure** | $k$-Means minimizes isotropic spherical variance | Cuts concentric rings into wedge slices | Use Spectral Clustering with Graph Laplacian eigenvectors |
| **Log-Sum-Exp Underflow** | Large negative log-likelihoods in high dimensions | Underflow to zero before denominator sum | Stable log-sum-exp: $\log \sum \exp(x_i - x_{\max}) + x_{\max}$ |
| **Isolated Graph Nodes** | Small affinity bandwidth $\gamma \gg 1$ yields isolated points | Degree $D_{ii} = 0 \implies D_{ii}^{-1/2} = \text{Inf}$ | Degree clamping: $\max(D_{ii}, 10^{-6})$ |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Vector Quantization in Generative Models:** VQ-VAE and discrete tokenizers (SoundStream, EnCodec) use online EMA $k$-Means codebooks to quantize continuous representations.
2. **Graph Neural Networks (GNNs):** The symmetric normalized Laplacian $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$ forms the exact mathematical foundation of GCN spectral convolution filters (Kipf & Welling, 2016).
3. **Mixture-of-Experts (MoE) Gating:** Soft GMM responsibilities $\gamma_{ik} = \operatorname{softmax}_k(\log p(\mathbf{x}|k))$ mirror top-$k$ gating routers in sparse Transformer architectures.

---

## 7. Verification & Parity Results
All numerical routines satisfy the strict multi-track parity contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-6}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path27"`: **PASS**
- `cargo test -p ai-candle --test 02_classical_and_probabilistic_ml`: **PASS (4/4 tests)**
- `cargo test -p ai-kernel --test 02_classical_and_probabilistic_ml`: **PASS (4/4 tests)**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [03_knn_prototype_and_locally_weighted_regression.md](03_knn_prototype_and_locally_weighted_regression.md)
- **Current Topic:** 04_clustering_kmeans_spectral_and_gmm_em.md
- **Next Topic:** [05_dimensionality_reduction_pca_lda_ica_tsne_umap.md](05_dimensionality_reduction_pca_lda_ica_tsne_umap.md)
- **Category Index:** [02_classical_and_probabilistic_ml/README.md](README.md)
