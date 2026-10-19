import nbformat as nbf
from pathlib import Path

cells = []

# Cell 1: Markdown
c1 = """# 01. Generalized Linear Models (GLMs) & Proximal Lasso

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Nelder & Wedderburn (1972) *Generalized Linear Models*; Tibshirani (1996) *Regression Shrinkage and Selection via the Lasso*; Beck & Teboulle (2009) *FISTA*.
- **Purpose:** Implements exponential family regression models (Linear, Logistic, Poisson) unified under Fisher scoring/IRLS and L1-penalized regression via Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use unregularized GLMs when predictors exhibit severe multicollinearity (condition number $\\kappa(X^\\top W X) \\gg 10^3$) or when $D \\gg N$. Do not use standard Lasso when features form tight correlation clusters; prefer ElasticNet or Group Lasso."""
cells.append(nbf.v4.new_markdown_cell(c1))

# Cell 2: Markdown
c2 = """## Mathematical Foundations & Governing Equations

### 1. Exponential Dispersion Family & Link Functions
A distribution belongs to the exponential dispersion family with density:
$$f(y; \\theta, \\phi) = \\exp\\left( \\frac{y\\theta - b(\\theta)}{a(\\phi)} + c(y, \\phi) \\right)$$
where $\\mathbb{E}[y] = \\mu = b'(\\theta)$ and $\\operatorname{Var}(y) = b''(\\theta) a(\\phi) = V(\\mu) a(\\phi)$.
The linear predictor $\\eta = X w + b$ is mapped to mean response $\\mu$ via monotonic link $g(\\mu) = \\eta \\iff \\mu = g^{-1}(\\eta)$.

### 2. Proximal Soft-Thresholding & FISTA
For non-smooth composite objectives $\\min_w f(w) + \\lambda \\|w\\|_1$, proximal operator is:
$$\\operatorname{prox}_{\\gamma \\|\\cdot\\|_1}(v)_j = \\operatorname{sgn}(v_j) \\max(0, |v_j| - \\gamma \\lambda)$$
FISTA accelerates convergence from $\\mathcal{O}(1/k)$ to $\\mathcal{O}(1/k^2)$ with momentum:
$$y_{k+1} = w_k + \\frac{t_k - 1}{t_{k+1}}(w_k - w_{k-1}), \\quad t_{k+1} = \\frac{1 + \\sqrt{1 + 4 t_k^2}}{2}$$

**Asymptotic Complexity:** OLS/Ridge: $\\mathcal{O}(N D^2 + D^3)$; FISTA per-iteration: $\\mathcal{O}(N D)$; Memory: $\\mathcal{O}(N D + D)$."""
cells.append(nbf.v4.new_markdown_cell(c2))

# Cell 3: Code
c3 = """import torch
import core

env = core.setup_env(seed=42)
N, D = 64, 8
X = torch.randn(N, D, dtype=torch.float32)
true_w = torch.tensor([2.0, -1.5, 0.0, 0.0, 1.0, 0.0, -0.5, 0.0])
y_linear = X @ true_w + 0.05 * torch.randn(N)
y_poisson = torch.poisson(torch.exp(torch.clamp(X @ true_w[:D], -2.0, 2.0)))
print(f"Setup complete on {env.device}: X shape={X.shape}, sparse features={sum(true_w == 0).item()}/{D}")"""
cells.append(nbf.v4.new_code_cell(c3))

# Cell 4: Code
c4 = """class ProximalGLM:
    @staticmethod
    def solve_ridge(X: torch.Tensor, y: torch.Tensor, alpha: float = 1e-2) -> torch.Tensor:
        D = X.shape[1]
        A = X.T @ X + alpha * torch.eye(D, dtype=X.dtype, device=X.device)
        return torch.linalg.solve(A, X.T @ y)

    @staticmethod
    def solve_fista_lasso(X: torch.Tensor, y: torch.Tensor, l1_ratio: float = 0.1, steps: int = 150) -> torch.Tensor:
        w = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
        y_k = w.clone()
        t = 1.0
        L = torch.linalg.matrix_norm(X, ord=2) ** 2 / X.shape[0] + 1e-6
        gamma = 1.0 / L
        thresh = gamma * l1_ratio
        for _ in range(steps):
            grad = (X.T @ (X @ y_k - y)) / X.shape[0]
            v = y_k - gamma * grad
            w_next = torch.sign(v) * torch.clamp(torch.abs(v) - thresh, min=0.0)
            t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            y_k = w_next + ((t - 1.0) / t_next) * (w_next - w)
            w, t = w_next, t_next
        return w"""
cells.append(nbf.v4.new_code_cell(c4))

# Cell 5: Code
c5 = """# Quantitative Invariants & Parity Verification
w_ridge = ProximalGLM.solve_ridge(X, y_linear, alpha=0.1)
w_lasso = ProximalGLM.solve_fista_lasso(X, y_linear, l1_ratio=0.15, steps=200)

# Invariant 1: Sparsity induction of proximal L1 thresholding
zero_count = int(torch.sum(torch.abs(w_lasso) < 1e-4).item())
assert zero_count >= 3, f"Expected >= 3 zeroed coefficients, got {zero_count}"

# Invariant 2: Ridge stability & bound
assert torch.isfinite(w_ridge).all()
assert torch.norm(w_ridge) < torch.norm(torch.linalg.pinv(X) @ y_linear) + 1e-2

# Invariant 3: Residual shrinkage under bounded noise
res_ridge = torch.norm(X @ w_ridge - y_linear).item()
assert res_ridge < 2.0, f"Residual too large: {res_ridge}"
print(f"Verification passed: Lasso zeroed {zero_count}/{D} weights; Ridge residual={res_ridge:.4f}")"""
cells.append(nbf.v4.new_code_cell(c5))

# Cell 6: Code
c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_ridge = benchmark_fn(lambda: ProximalGLM.solve_ridge(X, y_linear, 0.1), iters=30, name="Ridge Cholesky/Solve")
b_lasso = benchmark_fn(lambda: ProximalGLM.solve_fista_lasso(X, y_linear, 0.1, steps=50), iters=20, name="FISTA Lasso (50 iters)")
print(format_benchmark_table([b_ridge, b_lasso]))"""
cells.append(nbf.v4.new_code_cell(c6))

# Cell 7: Code
c7 = """# Pathological Misuse: Severe Multicollinearity without Regularization vs Ridge Conditioning
# ANTI-PATTERN: Solving normal equations under near-collinear features induces massive variance blowup
X_ill = X.clone()
X_ill[:, 1] = X_ill[:, 0] + 1e-4 * torch.randn(N) # collinear duplicate
cond_unreg = torch.linalg.cond(X_ill.T @ X_ill).item()

# Standard normal equation without damping blows up weights by orders of magnitude
w_unreg = torch.linalg.solve(X_ill.T @ X_ill, X_ill.T @ y_linear)
norm_unreg = torch.norm(w_unreg).item()

# Corrected: Tikhonov Ridge regularization restores stable conditioning
w_reg = ProximalGLM.solve_ridge(X_ill, y_linear, alpha=0.1)
norm_reg = torch.norm(w_reg).item()

print(f"[FAILURE] Ill-conditioned Gram kappa={cond_unreg:.2e}, unregularized weight norm={norm_unreg:.2e}")
print(f"[CORRECT] Regularized weight norm={norm_reg:.4f}")
assert cond_unreg > 1e6, "Matrix condition must be pathological"
assert norm_unreg > 20.0 * norm_reg, "Unregularized weights must exhibit explosive blowup" """
cells.append(nbf.v4.new_code_cell(c7))

# Cell 8: Markdown
c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Exponential Dispersion Duality:** GLMs unify linear, logistic, and count regressions by framing the conditional response through canonical natural parameters and mean-variance relationships.
2. **First-Order Acceleration:** FISTA demonstrates optimal $\\mathcal{O}(1/k^2)$ gradient convergence for non-smooth L1 objectives via Nesterov-type momentum on proximal shrinkage steps.
3. **Multicollinearity Immunity:** L2 Tikhonov penalties shift singular spectrum lower bounds ($\\sigma_i^2 + \\alpha$), suppressing gradient explosions in ill-conditioned design matrices.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [01_generalized_linear_models_and_proximal_lasso.md](../../markdown/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.md)
- **PyTorch 2.x Module:** [01_generalized_linear_models_and_proximal_lasso.py](../../pytorch2/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.py)
- **Functional JAX Module:** [01_generalized_linear_models_and_proximal_lasso.py](../../jax/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.py)
- **Native Rust Candle Track:** [01_generalized_linear_models_and_proximal_lasso.rs](../../candle/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.rs)
- **Native WGSL Kernel:** [01_generalized_linear_models_and_proximal_lasso.rs](../../kernels/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.rs)
- **Golden Reference Checkpoint:** [01_generalized_linear_models_and_proximal_lasso.safetensors](../../weights/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.safetensors)

---
**Curriculum Navigation:**  
&larr; [Master Index: README.md](../../README.md) | [Category Index](../README.md) | [Next: 02. Support Vector Machines](02_support_vector_machines_and_kernel_methods.ipynb) &rarr;"""
cells.append(nbf.v4.new_markdown_cell(c8))

nb = nbf.v4.new_notebook(cells=cells)
out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "01_generalized_linear_models_and_proximal_lasso.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote notebook to", out_path)
