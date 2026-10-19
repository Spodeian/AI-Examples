# 08. Matrix Factorization: ALS, NMF, BPR, and Factorization Machines

## 1. Executive Summary
Recommender systems and collaborative filtering models operate over high-dimensional, highly sparse user-item interaction spaces. Traditional dense regression fails catastrophically when missing interactions are conflated with true negative feedback. This module investigates low-rank latent representation learning across four landmark architectures: Alternating Least Squares (Koren et al., 2009 ALS), Non-Negative Matrix Factorization (Lee & Seung, 1999 NMF), Bayesian Personalized Ranking (Rendle et al., 2009 BPR), and Factorization Machines (Rendle, 2010 FM). We emphasize Rendle's algebraic $O(KD)$ trick, which calculates full second-order pairwise polynomial feature interactions in linear time, fully implemented across PyTorch 2.x, JAX, native Rust Candle, and WebGPU WGSL compute shaders.

---

## 2. Mathematical Foundations

### 2.1 Low-Rank Matrix Factorization via Alternating Least Squares (ALS)
Given an observed rating matrix $\mathbf{R} \in \mathbb{R}^{U \times I}$, low-rank matrix factorization approximates $\mathbf{R} \approx \mathbf{P} \mathbf{Q}^\top$, with $\mathbf{P} \in \mathbb{R}^{U \times K}$ and $\mathbf{Q} \in \mathbb{R}^{I \times K}$ ($K \ll \min(U, I)$). Because the joint optimization is non-convex, ALS holds one factor fixed while solving the ridge regression problem for the other in closed form:
$$\mathbf{p}_u = \left(\sum_{i \in \mathcal{I}_u} \mathbf{q}_i \mathbf{q}_i^\top + \lambda \mathbf{I}\right)^{-1} \sum_{i \in \mathcal{I}_u} r_{ui} \mathbf{q}_i, \quad \mathbf{q}_i = \left(\sum_{u \in \mathcal{U}_i} \mathbf{p}_u \mathbf{p}_u^\top + \lambda \mathbf{I}\right)^{-1} \sum_{u \in \mathcal{U}_i} r_{ui} \mathbf{p}_u$$

### 2.2 Non-Negative Matrix Factorization (NMF) Multiplicative Updates
When representations must be strictly additive and parts-based ($\mathbf{V}_{ui} \ge 0$), NMF optimizes $\min_{\mathbf{W} \ge 0, \mathbf{H} \ge 0} \|\mathbf{V} - \mathbf{W}\mathbf{H}\|_F^2$. Lee & Seung's diagonal rescaling updates guarantee non-negativity and non-increasing Frobenius norm:
$$\mathbf{H} \leftarrow \mathbf{H} \odot \frac{\mathbf{W}^\top \mathbf{V}}{\mathbf{W}^\top \mathbf{W} \mathbf{H} + \epsilon}, \quad \mathbf{W} \leftarrow \mathbf{W} \odot \frac{\mathbf{V} \mathbf{H}^\top}{\mathbf{W} \mathbf{H} \mathbf{H}^\top + \epsilon}$$

### 2.3 Bayesian Personalized Ranking (BPR)
Implicit feedback datasets (clicks, views) contain only positive observations; unobserved entries are a mixture of negative preferences and unexposed items. BPR optimizes pairwise ranking for triplets $(u, i, j)$ where user $u$ preferred item $i$ over unobserved item $j$:
$$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j) \in \mathcal{D}_S} \ln \sigma(\hat{x}_{ui} - \hat{x}_{uj}) + \lambda \|\Theta\|^2, \quad \hat{x}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$$

### 2.4 Factorization Machines (FM) and the O(KD) Algebraic Identity
2-way polynomial feature interactions naively require $O(D^2)$ time:
$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^D w_i x_i + \sum_{i=1}^D \sum_{j=i+1}^D \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$
By reformulating symmetric bilinear interactions, Rendle showed that pairwise inner products can be computed in linear time $O(K D)$:
$$\sum_{i=1}^D \sum_{j=i+1}^D \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^K \left[ \left(\sum_{i=1}^D v_{i,f} x_i\right)^2 - \sum_{i=1}^D v_{i,f}^2 x_i^2 \right]$$
This identity enables factorization machines to operate on massive web-scale sparse features (e.g. $D = 10^7$, $K = 16$).

---

## 3. Tensor Mechanics & Compute Graphs

```
[Input Observations X: N x D]
       │
       ├──> [Linear: w0 + X @ w] ──────────────────────────┐
       │                                                   ▼
       └──> [Latent Factor Projections: X @ V [N, K]] ──> [Sum_f: 0.5 * ((X@V)^2 - (X^2 @ V^2))] ──> [FM Out: N]
```

---

## 4. Multi-Ecosystem Implementation Parity

- **PyTorch 2.x (`pytorch2/`):** PyTorch tensor module using batch matrix multiplication for $O(KD)$ interactions and `torch.linalg.solve` for ALS.
- **JAX (`jax/`):** Functional JIT-compiled matrix operations with analytical gradient tracking.
- **Native Rust Candle (`candle/`):** Pure Rust Candle tensor implementation of Rendle's algebraic identity and BPR loss.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader parallelizing 2-way interaction sums over 1D workgroups of size 64.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Implicit Negative Collapse** | Treating unobserved implicit feedback as zero-rating negatives | Recommendation list collapses into majority-class bias | Optimize pairwise BPR ranking loss rather than point MSE |
| **$O(D^2)$ Memory Exhaustion** | Materializing full $D \times D$ pairwise interaction tensor | GPU out-of-memory error on $D > 10^4$ | Apply Rendle's $O(KD)$ algebraic summation identity |
| **NMF Division by Zero** | Zero entries in factor matrices during multiplicative steps | NaN values in $\mathbf{W}$ or $\mathbf{H}$ | Add machine precision epsilon jitter $\epsilon = 10^{-7}$ |
| **Collinear Rank Collapse in ALS** | Insufficient user ratings causing singular Gram matrix $\mathbf{P}^\top \mathbf{P}$ | Inversion failure / numerical explosion | Add Tikhonov regularizer $\lambda \mathbf{I}$ with $\lambda \ge 0.05$ |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Deep & Cross Networks (DCN):** Modern click-through rate (CTR) prediction architectures in Google and Meta search engines extend FM bilinear products into multi-layer explicit polynomial cross-networks.
2. **LoRA as Matrix Factorization:** Low-Rank Adaptation for large language models factorizes parameter updates as $\Delta W = B A$, matching the singular value structure of ALS.
3. **Implicit Matrix Factorization in Biology:** Single-cell RNA sequencing data uses zero-inflated NMF to discover latent cell types across ultra-sparse gene expression matrices.

---

## 7. Verification & Parity Results
All implementations satisfy the cross-track invariant contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path31"`: **PASS**
- `cargo test -p ai-candle --test 02_classical_and_probabilistic_ml -- test_08`: **PASS**
- `cargo test -p ai-kernel --test 02_classical_and_probabilistic_ml -- test_08`: **PASS**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [07_differentiable_trees_node_soft_gbdt_and_gam_nam.md](07_differentiable_trees_node_soft_gbdt_and_gam_nam.md)
- **Current Topic:** 08_matrix_factorization_als_nmf_bpr_and_fm.md
- **Next Topic:** [09_hidden_markov_models_kalman_ukf_and_particle_filters.md](09_hidden_markov_models_kalman_ukf_and_particle_filters.md)
- **Category Index:** [02_classical_and_probabilistic_ml/README.md](README.md)
