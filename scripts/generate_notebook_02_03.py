import nbformat as nbf
from pathlib import Path

cells = []

# Cell 1: Markdown
c1 = """# 03. k-Nearest Neighbors, Prototype Learning & Locally Weighted Regression

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Fix & Hodges (1951) *Nonparametric discrimination: Consistency properties*; Cover & Hart (1967) *Nearest neighbor pattern classification*; Nadaraya (1964) / Watson (1964) *Smooth regression analysis*; Cleveland (1979) *Robust locally weighted regression (LOESS)*.
- **Purpose:** Implements memory-efficient non-parametric classification, distance-weighted k-NN regression, and Nadaraya-Watson kernel regression using Tensor Core GEMM distance expansion.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use raw Euclidean k-NN when feature dimensionality $D \\gg 50$ without dimensionality reduction (Curse of Dimensionality renders pairwise distance ratios indistinguishable). Do not call un-clamped `sqrt(dist_sq)` on floating-point distance expansions, as cancellation introduces $-10^{-7} \\implies \\text{NaN}$."""
cells.append(nbf.v4.new_markdown_cell(c1))

# Cell 2: Markdown
c2 = """## Mathematical Foundations & Governing Equations

### 1. Vectorized Pairwise Euclidean Distance Metric
Given datasets $\\mathbf{X}_1 \\in \\mathbb{R}^{N_1 \\times D}$ and $\\mathbf{X}_2 \\in \\mathbb{R}^{N_2 \\times D}$, expanding the binomial norm yields:
$$\\|\\mathbf{x}_i - \\mathbf{x}_j\\|_2^2 = \\|\\mathbf{x}_i\\|_2^2 + \\|\\mathbf{x}_j\\|_2^2 - 2 \\mathbf{x}_i^\\top \\mathbf{x}_j$$
$$\\mathbf{D} = \\sqrt{\\max\\left( \\operatorname{diag}(\\mathbf{X}_1 \\mathbf{X}_1^\\top) \\mathbf{1}_{N_2}^\\top + \\mathbf{1}_{N_1} \\operatorname{diag}(\\mathbf{X}_2 \\mathbf{X}_2^\\top)^\\top - 2 \\mathbf{X}_1 \\mathbf{X}_2^\\top, \\; 0 \\right)}$$

### 2. Locally Weighted Kernel Regression
For query point $\\mathbf{x}_*$ and kernel bandwidth $h$, Nadaraya-Watson computes the normalized conditional expectation:
$$\\hat{m}(\\mathbf{x}_*) = \\frac{\\sum_{i=1}^N K_h(\\mathbf{x}_* - \\mathbf{x}_i) y_i}{\\sum_{i=1}^N K_h(\\mathbf{x}_* - \\mathbf{x}_i)}, \\quad K_h(\\mathbf{u}) = \\exp\\left( -\\frac{\\|\\mathbf{u}\\|_2^2}{2 h^2} \\right)$$

**Asymptotic Complexity:** Exact k-NN Search: $\\mathcal{O}(N D + N \\log k)$ time, $\\mathcal{O}(N D)$ storage; Nadaraya-Watson: $\\mathcal{O}(N D)$ time per query point."""
cells.append(nbf.v4.new_markdown_cell(c2))

# Cell 3: Code
c3 = """import torch
import core

env = core.setup_env(seed=42)
N_train, N_test, D, K = 64, 16, 8, 5
X_train = torch.randn(N_train, D, dtype=torch.float32)
w_true = torch.tensor([1.0, -0.5, 0.8, -1.2, 0.4, -0.3, 0.6, -0.7])
y_train = torch.sin(X_train @ w_true) + 0.05 * torch.randn(N_train)
X_test = torch.randn(N_test, D, dtype=torch.float32)

print(f"Setup complete on {env.device}: Train X={X_train.shape}, Test X={X_test.shape}, K={K}")"""
cells.append(nbf.v4.new_code_cell(c3))

# Cell 4: Code
c4 = """class NonparametricRegression:
    @staticmethod
    def pairwise_sq_distance(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        x1_sq = torch.sum(X1 ** 2, dim=1, keepdim=True)
        x2_sq = torch.sum(X2 ** 2, dim=1, keepdim=True).T
        dist_sq = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
        return torch.clamp(dist_sq, min=0.0)

    @staticmethod
    def knn_predict(X_train: torch.Tensor, y_train: torch.Tensor, X_query: torch.Tensor, k: int = 5, eps: float = 1e-5) -> torch.Tensor:
        dist_sq = NonparametricRegression.pairwise_sq_distance(X_train, X_query)
        dists = torch.sqrt(dist_sq.T) # [N_query, N_train]
        vals, idxs = torch.topk(dists, k=k, largest=False)
        weights = 1.0 / (vals + eps)
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        return torch.sum(weights * y_train[idxs], dim=1)

    @staticmethod
    def nadaraya_watson_predict(X_train: torch.Tensor, y_train: torch.Tensor, X_query: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
        dist_sq = NonparametricRegression.pairwise_sq_distance(X_train, X_query) # [N_train, N_query]
        weights = torch.exp(-0.5 * dist_sq / (bandwidth ** 2))
        norm_weights = weights / torch.clamp(torch.sum(weights, dim=0, keepdim=True), min=1e-8)
        return torch.sum(norm_weights * y_train.unsqueeze(1), dim=0)"""
cells.append(nbf.v4.new_code_cell(c4))

# Cell 5: Code
c5 = """# Quantitative Invariants & Parity Verification
dist_sq = NonparametricRegression.pairwise_sq_distance(X_train, X_test)
pred_knn = NonparametricRegression.knn_predict(X_train, y_train, X_test, k=K)
pred_nw = NonparametricRegression.nadaraya_watson_predict(X_train, y_train, X_test, bandwidth=1.0)

# Invariant 1: Distance non-negativity and self-distance zero identity
self_dist = torch.diagonal(NonparametricRegression.pairwise_sq_distance(X_train, X_train))
assert torch.all(dist_sq >= 0.0), "Pairwise distance squared must be non-negative"
assert torch.all(self_dist < 1e-5), "Self distance must evaluate to 0"

# Invariant 2: Prediction finiteness and bounded output dynamic range
assert torch.isfinite(pred_knn).all(), "k-NN predictions contain non-finite numbers"
assert torch.isfinite(pred_nw).all(), "Nadaraya-Watson predictions contain non-finite numbers"

# Invariant 3: Concordance between localized k-NN and kernel smoothing
mean_abs_diff = torch.mean(torch.abs(pred_knn - pred_nw)).item()
assert mean_abs_diff < 0.5, f"Discrepancy between k-NN and NW too high: {mean_abs_diff}"
print(f"Verification passed: self_dist_max={self_dist.max():.2e}, mean_abs_diff={mean_abs_diff:.4f}")"""
cells.append(nbf.v4.new_code_cell(c5))

# Cell 6: Code
c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_dist = benchmark_fn(lambda: NonparametricRegression.pairwise_sq_distance(X_train, X_test), iters=30, name="Pairwise Dist GEMM (64x16)")
b_knn = benchmark_fn(lambda: NonparametricRegression.knn_predict(X_train, y_train, X_test, k=5), iters=30, name="Distance-Weighted k-NN (k=5)")
b_nw = benchmark_fn(lambda: NonparametricRegression.nadaraya_watson_predict(X_train, y_train, X_test, 1.0), iters=30, name="Nadaraya-Watson (h=1.0)")
print(format_benchmark_table([b_dist, b_knn, b_nw]))"""
cells.append(nbf.v4.new_code_cell(c6))

# Cell 7: Code
c7 = """# Pathological Misuse: Unclamped Floating-Point Cancellation vs Clamped Euclidean Distance
# ANTI-PATTERN: Identical vectors with float32 cancellation yield small negative numbers
X_dup = X_train[:10].clone()
x_sq = torch.sum(X_dup ** 2, dim=1, keepdim=True)
unclamped_dist_sq = x_sq + x_sq.T - 2.0 * (X_dup @ X_dup.T)

# Negative values cause torch.sqrt to produce catastrophic NaNs
nan_count = torch.isnan(torch.sqrt(unclamped_dist_sq)).sum().item()

# Corrected: Explicit clamp to 0.0 eliminates floating-point sign flips
clamped_dists = torch.sqrt(torch.clamp(unclamped_dist_sq, min=0.0))
assert torch.isfinite(clamped_dists).all(), "Clamped distances must be completely finite"

print(f"[FAILURE] Unclamped Euclidean sqrt produced {nan_count} catastrophic NaNs on identical vectors")
print(f"[CORRECT] Clamped self-distances max error: {torch.max(torch.diagonal(clamped_dists)):.2e} (Clean 0.0)")
assert nan_count >= 1, "Unclamped floating point identity cancellation must trigger NaNs" """
cells.append(nbf.v4.new_code_cell(c7))

# Cell 8: Markdown
c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **The Cover-Hart Guarantee:** 1-NN error never exceeds twice the theoretical minimum Bayes error ($R_{1\\text{-NN}} \\le 2 R^*$), providing an asymptotic performance ceiling for any complex parametric model.
2. **Attention as Soft k-NN:** Transformer self-attention replaces hard discrete top-$k$ index gathering with continuous softmax-normalized Nadaraya-Watson kernel smoothing across representation space.
3. **Cancellation Resilience:** Always apply `clamp(min=0.0)` prior to calling `sqrt()` on expanded matrix products to eliminate IEEE 754 subtraction errors.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [03_knn_prototype_and_locally_weighted_regression.md](../../markdown/02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.md)
- **PyTorch 2.x Module:** [03_knn_prototype_and_locally_weighted_regression.py](../../pytorch2/02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.py)
- **Functional JAX Module:** [03_knn_prototype_and_locally_weighted_regression.py](../../jax/02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.py)
- **Native Rust Candle Track:** [03_knn_prototype_and_locally_weighted_regression.rs](../../candle/02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.rs)
- **Hardware WGSL Kernel:** [03_knn_prototype_and_locally_weighted_regression.rs](../../kernels/02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.rs)
- **Golden Reference Checkpoint:** [03_knn_prototype_and_locally_weighted_regression.safetensors](../../weights/02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 02. Support Vector Machines](02_support_vector_machines_and_kernel_methods.ipynb) | [Category Index](../README.md) | [Next: 04. Clustering, k-Means & GMM EM](04_clustering_kmeans_spectral_and_gmm_em.ipynb) &rarr;"""
cells.append(nbf.v4.new_markdown_cell(c8))

nb = nbf.v4.new_notebook(cells=cells)
out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "03_knn_prototype_and_locally_weighted_regression.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote notebook to", out_path)
