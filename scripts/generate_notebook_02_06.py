"""Generate 8-cell educational notebook for Topic 06:
06_naive_bayes_bayesian_regression_and_gaussian_processes.ipynb
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf

def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 06. Naive Bayes, Bayesian Linear Regression & Gaussian Processes

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Thomas Bayes (1763) *An Essay towards solving a Problem in the Doctrine of Chances*; Carl Edward Rasmussen & Christopher K. I. Williams (2006) *Gaussian Processes for Machine Learning*; Michalis K. Titsias (2009) *Variational Learning of Inducing Variables in Sparse Gaussian Processes*.
- **Purpose:** Rigorous unified implementation of generative conditional independence classifiers, analytical conjugate parameter posteriors, and non-parametric function-space GP inference.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use Naive Bayes when features exhibit strong multi-way correlations or dense covariance dependencies that violate conditional independence. Do not use Exact Gaussian Processes when dataset sample size exceeds $N > 10^4$ without inducing-point sparse approximations (SVGP), as $O(N^3)$ Cholesky factorization and $O(N^2)$ kernel memory become intractable."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Formulation & Asymptotic Complexity

### 1. Gaussian Naive Bayes in Log-Domain
Assuming feature independence conditioned on class $y = c$, the joint probability is evaluated via log-probabilities:
$$\\log p(c, \\mathbf{x}) = \\log \\pi_c - \\frac{1}{2} \\sum_{d=1}^D \\left[ \\log(2\\pi \\sigma_{cd}^2) + \\frac{(x_d - \\mu_{cd})^2}{\\sigma_{cd}^2} \\right]$$
Posterior probabilities are recovered via numerically stable Log-Sum-Exp normalization:
$$p(y = c \\mid \\mathbf{x}) = \\exp\\left( \\log p(c, \\mathbf{x}) - \\text{LogSumExp}_{k} \\log p(k, \\mathbf{x}) \\right)$$

### 2. Conjugate Bayesian Linear Regression
With prior $\\mathbf{w} \\sim \\mathcal{N}(\\mathbf{0}, \\alpha^{-1}\\mathbf{I})$ and likelihood variance $\\beta^{-1}$, the analytical posterior covariance $\\mathbf{S}_N$ and mean $\\mathbf{m}_N$ are:
$$\\mathbf{S}_N = (\\alpha \\mathbf{I} + \\beta \\mathbf{X}^\\top \\mathbf{X})^{-1}, \\quad \\mathbf{m}_N = \\beta \\mathbf{S}_N \\mathbf{X}^\\top \\mathbf{y}$$
$$\\mu_* = \\mathbf{x}_*^\\top \\mathbf{m}_N, \\quad \\sigma_*^2 = \\frac{1}{\\beta} + \\mathbf{x}_*^\\top \\mathbf{S}_N \\mathbf{x}_*$$

### 3. Exact Gaussian Process Regression via Cholesky
Given kernel Gram matrix $\\mathbf{K}_y = \\mathbf{K} + \\sigma_n^2 \\mathbf{I} = \\mathbf{L} \\mathbf{L}^\\top$:
$$\\boldsymbol{\\alpha} = \\mathbf{L}^{-\\top} \\mathbf{L}^{-1} \\mathbf{y}, \\quad \\boldsymbol{\\mu}_* = \\mathbf{K}_* \\boldsymbol{\\alpha}, \\quad \\sigma_*^2 = k(\\mathbf{x}_*, \\mathbf{x}_*) + \\sigma_n^2 - \\|\\mathbf{L}^{-1}\\mathbf{k}_*\\|_2^2$$

**Complexity:** Naive Bayes: $\\mathcal{O}(N D C)$ time, $\\mathcal{O}(D C)$ memory; Bayesian Regression: $\\mathcal{O}(N D^2 + D^3)$; Exact GP: $\\mathcal{O}(N^3)$ Cholesky time, $\\mathcal{O}(N^2)$ memory; Sparse Inducing GP: $\\mathcal{O}(N M^2 + M^3)$ time, $\\mathcal{O}(N M)$ memory."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import core

env = core.setup_env(seed=42)
N, D, C = 32, 4, 2

# Generate synthetic Gaussian clusters and regression targets
X = torch.randn(N, D)
true_w = torch.tensor([1.5, -2.0, 0.8, -0.5])
y_reg = X @ true_w + 0.1 * torch.randn(N)
y_cls = (X[:, 0] + X[:, 1] > 0.0).long()

print(f"Setup complete on {env.device}: Observations X={X.shape}, Classes={C}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class ProbabilisticToolkit:
    @staticmethod
    def naive_bayes_log_proba(X: torch.Tensor, y: torch.Tensor, var_smoothing: float = 1e-4) -> torch.Tensor:
        C = len(torch.unique(y))
        means = torch.stack([X[y == c].mean(dim=0) for c in range(C)], dim=0)
        vars_ = torch.stack([X[y == c].var(dim=0, unbiased=False) + var_smoothing for c in range(C)], dim=0)
        priors = torch.tensor([(y == c).float().mean() for c in range(C)])
        diff_sq = (X.unsqueeze(1) - means.unsqueeze(0)) ** 2
        log_gauss = -0.5 * (torch.log(2.0 * torch.pi * vars_).unsqueeze(0) + diff_sq / vars_.unsqueeze(0))
        log_joint = torch.log(priors).unsqueeze(0) + log_gauss.sum(dim=-1)
        return log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)

    @staticmethod
    def bayesian_linear_regression(X: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, beta: float = 2.0):
        D = X.shape[1]
        S_N = torch.linalg.inv(alpha * torch.eye(D) + beta * (X.T @ X))
        m_N = beta * (S_N @ (X.T @ y))
        pred_mean = X @ m_N
        pred_var = (1.0 / beta) + torch.sum((X @ S_N) * X, dim=-1)
        return m_N, S_N, pred_mean, pred_var

    @staticmethod
    def gp_regression(X: torch.Tensor, y: torch.Tensor, lengthscale: float = 1.0, noise_var: float = 1e-4):
        dist_sq = torch.cdist(X, X) ** 2
        K = torch.exp(-0.5 * dist_sq / (lengthscale ** 2))
        L = torch.linalg.cholesky(K + noise_var * torch.eye(X.shape[0]))
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
        v = torch.linalg.solve_triangular(L, K, upper=False)
        pred_var = torch.clamp((1.0 + noise_var) - torch.sum(v ** 2, dim=0), min=1e-8)
        return alpha, K @ alpha, pred_var"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Invariants & Precision Verification
log_probs = ProbabilisticToolkit.naive_bayes_log_proba(X, y_cls)
m_N, S_N, b_mean, b_var = ProbabilisticToolkit.bayesian_linear_regression(X, y_reg, 1.0, 2.0)
gp_alpha, gp_mean, gp_var = ProbabilisticToolkit.gp_regression(X, y_reg, 1.0, 1e-4)

# Invariant 1: Sum of normalized probabilities equals 1.0 across all observations
prob_sum_err = torch.max(torch.abs(torch.sum(torch.exp(log_probs), dim=-1) - 1.0)).item()
assert prob_sum_err < 1e-6, f"Naive Bayes probabilities do not sum to 1: {prob_sum_err}"

# Invariant 2: Bayesian regression predictive variance is strictly greater than noise variance (1/beta = 0.5)
assert (b_var >= 0.5 - 1e-6).all(), "Bayesian predictive variance cannot fall below observation noise"

# Invariant 3: Exact GP interpolation error on training points must be close to noise scale
gp_fit_mse = torch.mean((gp_mean - y_reg) ** 2).item()
assert gp_fit_mse < 0.05, f"GP fit error exceeds expected threshold: {gp_fit_mse}"
print(f"Verification passed: prob_sum_err={prob_sum_err:.2e}, min_b_var={b_var.min():.4f}, gp_fit_mse={gp_fit_mse:.4f}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_nb = benchmark_fn(lambda: ProbabilisticToolkit.naive_bayes_log_proba(X, y_cls), iters=30, name="Gaussian Naive Bayes (32x4)")
b_bayes = benchmark_fn(lambda: ProbabilisticToolkit.bayesian_linear_regression(X, y_reg), iters=30, name="Bayesian Linear Regression")
b_gp = benchmark_fn(lambda: ProbabilisticToolkit.gp_regression(X, y_reg), iters=30, name="Exact GP (Cholesky 32x32)")
print(format_benchmark_table([b_nb, b_bayes, b_gp]))"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Raw Likelihood Multiplication vs Log-Domain Evaluation
# ANTI-PATTERN: In high dimensions (D=100), raw likelihood multiplication underflows to exact 0.0
D_high = 100
X_high = torch.randn(1, D_high)
var_high = torch.ones(D_high) * 0.5

# Raw likelihood product in float32
raw_probs = (1.0 / torch.sqrt(2.0 * torch.pi * var_high)) * torch.exp(-0.5 * (X_high ** 2) / var_high)
raw_product = torch.prod(raw_probs).item()

# Numerically stable log-domain sum
log_probs = -0.5 * (torch.log(2.0 * torch.pi * var_high) + (X_high ** 2) / var_high)
log_sum = torch.sum(log_probs).item()

print(f"[FAILURE] Raw Probability Product in D=100: {raw_product} (Catastrophic Floating-Point Underflow)")
print(f"[CORRECT] Log-Domain Likelihood Sum:       {log_sum:.4f} (Exact Finite Representation)")
assert raw_product == 0.0, "Raw product in 100 dimensions must underflow to float32 zero"
assert log_sum < -50.0, "Log sum accurately captures finite negative log-likelihood" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **The Log-Domain Axiom:** Multiplying probability densities over high-dimensional observations causes instantaneous IEEE 754 floating-point underflow. All generative inference must occur in the log-domain with `LogSumExp` normalization.
2. **Epistemic vs Aleatoric Uncertainty:** Bayesian linear regression and GPs explicitly distinguish between irreducible measurement noise $\\sigma_n^2$ (aleatoric) and finite data scarcity $\\mathbf{x}_*^\\top \\mathbf{S}_N \\mathbf{x}_*$ (epistemic).
3. **Cholesky Superiority:** Direct matrix inversion of kernel Gram matrices $\\mathbf{K}_y$ squares condition numbers and triggers numerical instability; triangular Cholesky decomposition ensures strict positive-definiteness.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [06_naive_bayes_bayesian_regression_and_gaussian_processes.md](../../markdown/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.md)
- **PyTorch 2.x Module:** [06_naive_bayes_bayesian_regression_and_gaussian_processes.py](../../pytorch2/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.py)
- **Functional JAX Module:** [06_naive_bayes_bayesian_regression_and_gaussian_processes.py](../../jax/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.py)
- **Native Rust Candle Track:** [06_naive_bayes_bayesian_regression_and_gaussian_processes.rs](../../candle/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.rs)
- **Hardware WGSL Kernel:** [06_naive_bayes_bayesian_regression_and_gaussian_processes.rs](../../kernels/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.rs)
- **Golden Reference Checkpoint:** [06_naive_bayes_bayesian_regression_and_gaussian_processes.safetensors](../../weights/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 05. Dimensionality Reduction (PCA/LDA/t-SNE/UMAP)](05_dimensionality_reduction_pca_lda_ica_tsne_umap.ipynb) | [Category Index](../README.md) | [Next: 07. Differentiable Trees (NODE, GBDT & GAM/NAM)](07_differentiable_trees_node_soft_gbdt_and_gam_nam.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "06_naive_bayes_bayesian_regression_and_gaussian_processes.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)

if __name__ == "__main__":
    build_notebook()
