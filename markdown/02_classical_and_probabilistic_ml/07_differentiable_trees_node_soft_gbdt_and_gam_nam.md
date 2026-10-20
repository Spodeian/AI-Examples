# 07. Differentiable Trees, NODE, Soft-GBDT, and GAM/NAM

## 1. Executive Summary
Decision trees and gradient boosted ensembles (CART, Breiman et al. 1984; GBDT, Friedman 2001) dominate tabular machine learning competitions and industrial predictive pipelines. However, traditional discrete splitting decisions create non-differentiable step-function boundaries that prevent joint end-to-end backpropagation with deep representation learning architectures. This module explores the frontier of differentiable tree architectures: soft decision trees (Frosst & Hinton, 2017), Neural Oblivious Decision Ensembles (NODE, Popov et al., 2019), Neural Additive Models (NAM, Agarwal et al., 2021), and the Pool-Adjacent-Violators (PAV) monotonic regression algorithm. We demonstrate how bitwise tensorized oblivious routing unites tabular tree inductive biases with continuous gradient-based optimization across PyTorch 2.x, JAX, native Rust Candle, and WebGPU WGSL compute shaders.

---

## 2. Mathematical Foundations

### 2.1 Soft Decision Trees with Continuous Routing
Whereas classical CART trees route samples down binary paths deterministically ($\mathbb{I}[x_j \le \theta]$), soft decision trees relax routing into continuous Bernoulli split probabilities parameterized by sigmoid gates:
$$p(\text{right} \mid \mathbf{x}) = \sigma\left(\frac{\mathbf{w}^\top \mathbf{x} + b}{\tau}\right)$$
Here, $\tau > 0$ denotes routing temperature. The probability of reaching leaf node $m$ along path $\mathcal{P}_m$ is the joint product of directional probabilities:
$$\mu_m(\mathbf{x}) = \prod_{k \in \mathcal{P}_m^{\text{right}}} p(k \mid \mathbf{x}) \prod_{j \in \mathcal{P}_m^{\text{left}}} (1 - p(j \mid \mathbf{x}))$$
The tree output is the probability-weighted sum of leaf response parameters: $\hat{y}(\mathbf{x}) = \sum_{m=1}^M \mu_m(\mathbf{x}) \mathbf{v}_m$.

### 2.2 Neural Oblivious Decision Ensembles (NODE)
An oblivious decision tree (CatBoost, Popov et al. 2019) restricts all split nodes at depth $l \in \{1, \dots, d\}$ to share the exact same decision feature and threshold. For a tree of depth $d$, there are $2^d$ leaves. The gating probability at depth level $l$ is:
$$p_l(\mathbf{x}) = \sigma\left(\sum_{j=1}^D W_{lj} x_j + b_l\right)$$
Because all splits at level $l$ are identical, the routing probability vector over all $2^d$ leaves is the tensor outer product (Kronecker product) of level-wise binary choices:
$$\boldsymbol{\pi}(\mathbf{x}) = \bigotimes_{l=1}^d \begin{bmatrix} 1 - p_l(\mathbf{x}) \\ p_l(\mathbf{x}) \end{bmatrix} \in \mathbb{R}^{2^d}$$
For depth $d = 2$, this tensorizes as:
$$\boldsymbol{\pi}(\mathbf{x}) = \begin{bmatrix} (1 - p_1)(1 - p_2) \\ (1 - p_1) p_2 \\ p_1 (1 - p_2) \\ p_1 p_2 \end{bmatrix}$$
The tree output is simply $\hat{y}(\mathbf{x}) = \boldsymbol{\pi}(\mathbf{x})^\top \mathbf{v}_{\text{leaves}}$. An ensemble of $T$ trees sums their respective outputs: $\hat{y}_{\text{NODE}}(\mathbf{x}) = \sum_{t=1}^T \hat{y}_t(\mathbf{x})$.

### 2.3 Neural Additive Models (NAM)
Generalized Additive Models (Hastie & Tibshirani, 1990) constrain predictive functions to an additive sum of univariate nonlinear components: $g(\mathbb{E}[y]) = \beta_0 + \sum_{j=1}^D f_j(x_j)$. Neural Additive Models parameterize each shape function $f_j$ by an independent deep neural network:
$$\hat{y}(\mathbf{x}) = \beta_0 + \sum_{j=1}^D f_j(x_j; \theta_j)$$
Because feature interactions are structurally prohibited, NAMs remain 100% transparent and interpretable: plotting $f_j(x_j)$ against $x_j$ reveals the exact causal influence of each individual covariate.

### 2.4 Pool-Adjacent-Violators (PAV) Isotonic Regression
Given ordered observations $x_{(1)} \le x_{(2)} \le \dots \le x_{(N)}$ and targets $y_i$, isotonic regression finds the optimal non-decreasing fit $\hat{\mathbf{y}}$ minimizing quadratic loss:
$$\min_{\hat{y}_1 \le \hat{y}_2 \le \dots \le \hat{y}_N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
The PAV algorithm processes points sequentially. Whenever a monotonicity violation occurs ($\hat{y}_i > \hat{y}_{i+1}$), it pools the adjacent violators into a single contiguous block replaced by their weighted arithmetic mean, iterating backwards until global monotonicity is restored.

---

## 3. Tensor Mechanics & Compute Graphs

```
[Input Observations X: N x D]
       │
       ├──> [NODE Gating: W_split [T, depth, D], b_split [T, depth]]
       │         │
       │         ▼
       │    [Gating Probs: p_l in (0, 1) [N, T, depth]]
       │         │
       │         ▼
       │    [Kronecker Leaf Probs: pi(X) [N, T, 2^d]] ──> [Leaf Values: V [T, 2^d]] ──> [NODE Out: N]
       │
       ├──> [NAM Modular Ensembles: f_1(x_1), f_2(x_2), ..., f_D(x_D)] ───────────────> [NAM Out: N]
       │
       └──> [PAV Monotonic Sweep: Adjacent Pooling Blocks] ────────────────────────────> [Isotonic Fit: N]
```

---

## 4. Multi-Ecosystem Implementation Parity

- **PyTorch 2.x (`pytorch2/`):** PyTorch `nn.Module` classes using outer product leaf probability accumulation and modular `nn.ModuleList` subnets for NAM.
- **JAX (`jax/`):** Pure functional JIT-compiled implementation leveraging `jax.lax.fori_loop` over trees and feature channels.
- **Native Rust Candle (`candle/`):** Pure Rust tensor implementation using Candle's `Tensor::matmul` and `candle_nn::ops::sigmoid`.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader parallelizing oblivious gating and leaf probability dot products over 1D workgroups of size 64.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Vanishing Tree Gradients** | Temperature $\tau \to 0$ (steep sigmoid slope) early in training | Sigmoid saturation; zero gradient through tree splits | Anneal inverse temperature gradually from $\tau = 1.0 \to 0.1$ |
| **Ensemble Overfitting** | Training soft GBDT without shrinkage rate ($\nu = 1.0$) | Massive train-test generalization gap | Apply shrinkage factor $\nu \in [0.05, 0.2]$ per boosting round |
| **Violator Memory Explosion** | Naive recursive implementation of PAV | Stack overflow on $N > 10^5$ | Use iterative contiguous block merging representation |
| **Feature Interaction Blindness in NAM** | Strong pairwise/triplet interactions present in data | Suboptimal test $R^2$ or accuracy | Extend NAM with shallow interaction pairs ($f_{jk}(x_j, x_k)$) |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Mixture of Experts (MoE) Routing:** The continuous routing mechanism $\boldsymbol{\pi}(\mathbf{x})$ in soft trees is mathematically identical to top-$k$ gating networks in Switch Transformers and Mixtral.
2. **Clinical Decision Support:** NAMs provide certified interpretability for high-stakes healthcare risk modeling (e.g., pneumonia mortality prediction) where black-box deep nets cannot be deployed safely.
3. **Hardware Acceleration:** Oblivious trees evaluate all splits concurrently via bitwise index addressing, making them exceptionally fast on SIMD hardware and FPGA accelerators.

---

## 7. Verification & Parity Results
All numerical routines satisfy the strict cross-track invariant contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path30"`: **PASS**
- `cargo test -p ai-candle --test 02_classical_and_probabilistic_ml -- test_07`: **PASS**
- `cargo test -p ai-kernel --test 02_classical_and_probabilistic_ml -- test_07`: **PASS**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [06_naive_bayes_bayesian_regression_and_gaussian_processes.md](06_naive_bayes_bayesian_regression_and_gaussian_processes.md)
- **Current Topic:** 07_differentiable_trees_node_soft_gbdt_and_gam_nam.md
- **Next Topic:** [08_matrix_factorization_als_nmf_bpr_and_fm.md](08_matrix_factorization_als_nmf_bpr_and_fm.md)
- **Category Index:** [Master Curriculum Index](../../README.md)
