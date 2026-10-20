# 11. Probabilistic Graphical Models: Factor Graphs, Belief Propagation, and CAVI

## 1. Executive Summary
Probabilistic Graphical Models (PGMs) provide a unified language for structured reasoning across high-dimensional discrete and continuous probability distributions. By decomposing joint distributions into products of localized factor potentials over graphs (Kschischang et al., 2001), exact marginal and MAP inference can be executed via dynamic programming message passing (Pearl, 1982 Belief Propagation). When graphs contain cycles, Loopy Belief Propagation provides rapid approximate marginals, but requires message damping $\alpha \in (0, 1)$ to prevent limit-cycle oscillations. For continuous intractable posteriors, Coordinate Ascent Mean-Field Variational Inference (Blei et al., 2017 CAVI) iteratively optimizes the Evidence Lower Bound (ELBO). This module implements Sum-Product and Max-Product tree inference, Damped Loopy BP, and CAVI across PyTorch 2.x, JAX, native Rust Candle, and hardware kernels.

---

## 2. Mathematical Foundations

### 2.1 Factor Graph Representation & The Sum-Product Algorithm
A factor graph consists of variable nodes $i \in \mathcal{V}$ and factor nodes $a \in \mathcal{F}$, defining the joint probability:
$$p(\mathbf{x}) = \frac{1}{Z} \prod_{a \in \mathcal{F}} f_a(\mathbf{x}_{\text{ne}(a)})$$

On tree-structured factor graphs, the Sum-Product algorithm computes exact marginal distributions in two passes (leaves-to-root, root-to-leaves):
1. **Variable-to-Factor Message:**
   $$\mu_{i \to a}(x_i) = \prod_{b \in \text{ne}(i) \setminus \{a\}} \mu_{b \to i}(x_i)$$
2. **Factor-to-Variable Message:**
   $$\mu_{a \to i}(x_i) = \sum_{\mathbf{x}_{\text{ne}(a) \setminus \{i\}}} f_a(\mathbf{x}_{\text{ne}(a)}) \prod_{j \in \text{ne}(a) \setminus \{i\}} \mu_{j \to a}(x_j)$$
3. **Marginal Belief:**
   $$b(x_i) = \frac{1}{Z_i} \prod_{a \in \text{ne}(i)} \mu_{a \to i}(x_i)$$

### 2.2 Max-Product Algorithm for MAP Inference
To find the global maximum a posteriori state $\mathbf{x}^* = \arg\max_\mathbf{x} p(\mathbf{x})$, the summation in the factor-to-variable message is replaced by the maximum operator:
$$\mu_{a \to i}^{\max}(x_i) = \max_{\mathbf{x}_{\text{ne}(a) \setminus \{i\}}} \left[ f_a(\mathbf{x}_{\text{ne}(a)}) \prod_{j \in \text{ne}(a) \setminus \{i\}} \mu_{j \to a}^{\max}(x_j) \right]$$

### 2.3 Damped Loopy Belief Propagation on Cyclic Graphs
On graphs with cycles, message passing is iterated until convergence. Because raw message updates can enter sustained oscillations on short cycles, convex damping with momentum factor $\alpha \in (0, 1)$ is applied:
$$\mu_{a \to i}^{(t)}(x_i) = (1 - \alpha) \mu_{a \to i}^{(t-1)}(x_i) + \alpha \mu_{a \to i}^{\text{raw}}(x_i)$$

### 2.4 Coordinate Ascent Mean-Field Variational Inference (CAVI)
When exact graph inference is intractable, Variational Inference approximates the posterior $p(\mathbf{z} | \mathbf{x})$ using a fully factorized family $q(\mathbf{z}) = \prod_{j=1}^m q_j(z_j)$. CAVI optimizes each factor $q_j$ while holding all others fixed:
$$\ln q_j^*(z_j) = \mathbb{E}_{-j} [\ln p(\mathbf{x}, \mathbf{z})] + \text{const}$$
The optimization monotonically increases the Evidence Lower Bound (ELBO):
$$\text{ELBO}(q) = \mathbb{E}_q [\ln p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_q [\ln q(\mathbf{z})] \le \ln p(\mathbf{x})$$

---

## 3. Tensor Mechanics & Compute Graphs

```
[Unary & Edge Potentials] ──> [Sum-Product Tree Pass: Leaves -> Root -> Leaves]
                                   │
                                   ├──> [Exact Variable Marginals b(x_i)]
                                   │
[Cyclic Factor Graph] ───────> [Damped Message Updates: (1-alpha)*m_old + alpha*m_new] ──> [Loopy Marginals]
                                   │
[Observations & Latents] ────> [CAVI Coordinate Updates: q_j* ~ exp(E_{-j}[log p])] ────> [Monotonic ELBO Ascent]
```

---

## 4. Multi-Ecosystem Implementation Parity

- **PyTorch 2.x (`pytorch2/`):** Tensorized tree message passing and vectorized CAVI responsibility updates.
- **JAX (`jax/`):** Pure functional `jax.jit` tree marginal solver and differentiable ELBO evaluation.
- **Native Rust Candle (`candle/`):** Zero-allocation matrix-vector message propagation and exact combinatorial MAP state search.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader parallelizing CAVI softmax responsibility updates across 64 threads per workgroup.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Loopy BP Limit Cycles** | Strong cyclic feedback loops without damping | Message values oscillate indefinitely; failure to converge | Apply damping $\mu^{(t)} = (1-\alpha)\mu^{(t-1)} + \alpha \mu^{\text{raw}}$ with $\alpha \le 0.5$ |
| **Message Numerical Underflow** | Multiplying unnormalized message products on deep trees | Values underflow to 0.0 | Normalize messages at every step or operate in log-semiring |
| **Mean-Field Variance Underestimation** | Enforcing factorized $q(z_1, z_2) = q_1(z_1)q_2(z_2)$ | Posterior covariance completely ignored; overconfident predictions | Use structured variational families or normalizing flows |
| **ELBO Monotonicity Failure** | Bugs in coordinate derivation or non-closed-form expectations | ELBO decreases between iterations | Validate analytical expectations $\mathbb{E}_{-j}[\ln p]$ against numerical integration |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Telecommunications & Error Correcting Codes:** Low-Density Parity-Check (LDPC) and Turbo codes decode gigabit 5G wireless transmissions using Loopy BP on bipartite factor graphs.
2. **Computer Vision & Image Segmentation:** Markov Random Fields (MRFs) with graph cut or Loopy BP inference perform stereo disparity estimation and semantic boundary labeling.
3. **Computational Structural Biology:** Protein side-chain conformation prediction operates on pairwise interaction graphs solved via tree-reweighted BP.

---

## 7. Verification & Parity Results
All implementations satisfy the cross-track invariant contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path34"`: **PASS**
- `cargo nextest run -p ai-candle -E 'test(test_11)'`: **PASS**
- `cargo nextest run -p ai-kernel -E 'test(test_11)'`: **PASS**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [10_survival_analysis_and_contextual_bandits.md](10_survival_analysis_and_contextual_bandits.md)
- **Current Topic:** 11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.md
- **Next Topic:** [12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.md](12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.md)
- **Category Index:** [Master Curriculum Index](../../README.md)
