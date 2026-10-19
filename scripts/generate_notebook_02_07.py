"""Generate 8-cell educational notebook for Topic 07:
07_differentiable_trees_node_soft_gbdt_and_gam_nam.ipynb
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf

def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 07. Differentiable Trees: NODE, Soft-GBDT & Neural Additive Models

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Leo Breiman et al. (1984) *CART*; Jerome H. Friedman (2001) *GBDT*; Nicholas Frosst & Geoffrey Hinton (2017) *Soft Decision Trees*; Sergei Popov et al. (2019) *NODE*; Rishabh Agarwal et al. (2021) *Neural Additive Models*.
- **Purpose:** Bridges discrete non-differentiable decision tree heuristics with smooth gradient backpropagation via oblivious tensorized split routing and modular additive networks.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use soft decision trees with very small routing temperature at initialization, as saturated sigmoid gates eliminate gradient flow to interior splits. Do not deploy pure Neural Additive Models on data dominated by complex multi-feature interactions, as univariate models cannot model multiplicative cross-feature terms without explicit interaction pairs."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Neural Oblivious Decision Ensembles (NODE)
Oblivious trees restrict all splits at depth $l$ to share linear projections $p_l(\\mathbf{x}) = \\sigma(\\mathbf{w}_l^\\top \\mathbf{x} + b_l)$. Leaf routing over $2^d$ leaves is the Kronecker outer product:
$$\\boldsymbol{\\pi}(\\mathbf{x}) = \\bigotimes_{l=1}^d \\begin{bmatrix} 1 - p_l(\\mathbf{x}) \\\\ p_l(\\mathbf{x}) \\end{bmatrix} \\in \\mathbb{R}^{2^d}, \\quad \\hat{y}_t(\\mathbf{x}) = \\boldsymbol{\\pi}(\\mathbf{x})^\\top \\mathbf{v}_t$$

### 2. Neural Additive Models (NAM)
NAM decomposes predictions into independent univariate neural functions:
$$\\hat{y}(\\mathbf{x}) = \\beta_0 + \\sum_{j=1}^D f_j(x_j; \\boldsymbol{\\theta}_j)$$

### 3. Pool-Adjacent-Violators (PAV) Isotonic Regression
Given ordered inputs $x_{(1)} \\le \\dots \\le x_{(N)}$, PAV solves $\\min_{\\hat{y}_1 \\le \\dots \\le \\hat{y}_N} \\sum_i (y_i - \\hat{y}_i)^2$ by iteratively pooling non-monotonic adjacent violators into weighted block averages.

**Complexity:** NODE: $\\mathcal{O}(N T (d D + 2^d))$ time; NAM: $\\mathcal{O}(N D H)$ time; PAV: $\\mathcal{O}(N \\log N)$ sort + $\\mathcal{O}(N)$ pooling."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import core

env = core.setup_env(seed=42)
N, D = 32, 4
X = torch.randn(N, D)
y_noisy = torch.linspace(0.0, 5.0, N) + 0.5 * torch.randn(N)
print(f"Setup complete on {env.device}: Observations X={X.shape}, Targets={y_noisy.shape}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class DifferentiableTreeToolkit:
    @staticmethod
    def node_forward(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, leaves: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(X.shape[0], device=X.device)
        for t in range(W.shape[0]):
            p0 = torch.sigmoid(X @ W[t, 0] + b[t, 0])
            p1 = torch.sigmoid(X @ W[t, 1] + b[t, 1])
            leaf_probs = torch.stack([(1-p0)*(1-p1), (1-p0)*p1, p0*(1-p1), p0*p1], dim=1)
            out = out + (leaf_probs @ leaves[t])
        return out

    @staticmethod
    def nam_forward(X: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, bias0: float) -> torch.Tensor:
        out = torch.full((X.shape[0],), bias0, device=X.device)
        for j in range(X.shape[1]):
            h = torch.relu(X[:, j].unsqueeze(1) @ w1[j].T + b1[j])
            out = out + (h @ w2[j].T + b2[j]).squeeze(1)
        return out

    @staticmethod
    def pav_isotonic(y: torch.Tensor) -> torch.Tensor:
        blocks = [[i, i, 1.0, float(y[i])] for i in range(len(y))]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][3]/blocks[i][2] > blocks[i+1][3]/blocks[i+1][2]:
                blocks[i][1], blocks[i][2], blocks[i][3] = blocks[i+1][1], blocks[i][2] + blocks[i+1][2], blocks[i][3] + blocks[i+1][3]
                blocks.pop(i + 1)
                i = max(0, i - 1)
            else:
                i += 1
        out = torch.empty_like(y)
        for s, e, w, v in blocks: out[s:e+1] = v / w
        return out"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Invariants & Precision Verification
torch.manual_seed(42)
W_split, b_split, leaf_vals = torch.randn(2, 2, 4) * 0.5, torch.randn(2, 2) * 0.1, torch.randn(2, 4)
w1, b1 = torch.randn(4, 8, 1) * 0.5, torch.randn(4, 8) * 0.1
w2, b2 = torch.randn(4, 1, 8) * 0.5, torch.randn(4, 1) * 0.1

node_out = DifferentiableTreeToolkit.node_forward(X, W_split, b_split, leaf_vals)
iso_fit = DifferentiableTreeToolkit.pav_isotonic(y_noisy)

# Invariant 1: NODE ensemble output bounded by sum of extreme leaf values
min_p, max_p = leaf_vals.min(dim=1).values.sum().item(), leaf_vals.max(dim=1).values.sum().item()
assert (node_out <= max_p + 1e-5).all() and (node_out >= min_p - 1e-5).all()

# Invariant 2: PAV output is monotonically non-decreasing
diffs = iso_fit[1:] - iso_fit[:-1]
assert (diffs >= -1e-6).all(), f"PAV monotonicity violated: min diff = {diffs.min()}"

# Invariant 3: NAM zero-interaction property across features
X_a, X_b = torch.randn(16, 4), torch.randn(16, 4)
X_a[:, 0], X_b[:, 0] = 1.5, 1.5
X_az, X_bz = X_a.clone(), X_b.clone()
X_az[:, 0], X_bz[:, 0] = 0.0, 0.0
eff_a = DifferentiableTreeToolkit.nam_forward(X_a, w1, b1, w2, b2, 0.25) - DifferentiableTreeToolkit.nam_forward(X_az, w1, b1, w2, b2, 0.25)
eff_b = DifferentiableTreeToolkit.nam_forward(X_b, w1, b1, w2, b2, 0.25) - DifferentiableTreeToolkit.nam_forward(X_bz, w1, b1, w2, b2, 0.25)
interr = torch.max(torch.abs(eff_a - eff_b)).item()
assert interr < 1e-5, f"NAM feature interactions must be zero: {interr}"
print(f"Verification passed: bounds=[{min_p:.2f}, {max_p:.2f}], monotonic_min={diffs.min():.4f}, interr={interr:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_node = benchmark_fn(lambda: DifferentiableTreeToolkit.node_forward(X, W_split, b_split, leaf_vals), iters=30, name="NODE Forward (2 trees, d=2)")
b_nam = benchmark_fn(lambda: DifferentiableTreeToolkit.nam_forward(X, w1, b1, w2, b2, 0.25), iters=30, name="NAM Forward (4 features, H=8)")
b_pav = benchmark_fn(lambda: DifferentiableTreeToolkit.pav_isotonic(y_noisy), iters=30, name="PAV Isotonic Regression (N=32)")
print(format_benchmark_table([b_node, b_nam, b_pav]))"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Overly Steep Temperature in Soft Decision Trees
# ANTI-PATTERN: Large initial inverse temperature beta=100 causes vanishing gradients
x_gate = torch.tensor([0.5, -0.5, 1.0], requires_grad=True)

torch.sigmoid(100.0 * x_gate).sum().backward()
grad_bad = x_gate.grad.clone()
x_gate.grad.zero_()
torch.sigmoid(1.0 * x_gate).sum().backward()
grad_good = x_gate.grad.clone()

print(f"[FAILURE] Saturated Gate Gradients (beta=100): max_grad = {grad_bad.abs().max():.2e} (Vanishing Gradient)")
print(f"[CORRECT] Smooth Gate Gradients (beta=1.0):     max_grad = {grad_good.abs().max():.4f} (Healthy Gradient Flow)")
assert grad_bad.abs().max() < 1e-15, "Steep temperature must saturate and zero out backprop gradients"
assert grad_good.abs().max() > 0.1, "Healthy temperature ensures strong gradient signal to split parameters" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Oblivious Structure for SIMD Concurrency:** Sharing split criteria across depth levels converts hierarchical binary routing into dense Kronecker products, eliminating branch divergence.
2. **Interpretability via Structural Modularity:** Neural Additive Models prove that neural nets match boosted tree accuracy on tabular data while maintaining exact additive interpretability.
3. **Smooth Annealing:** Starting training with soft routing ($\beta \\approx 1.0$) and gradually sharpening routing preserves differentiability while avoiding vanishing gradients.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [07_differentiable_trees_node_soft_gbdt_and_gam_nam.md](../../markdown/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.md)
- **PyTorch 2.x Module:** [07_differentiable_trees_node_soft_gbdt_and_gam_nam.py](../../pytorch2/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.py)
- **Functional JAX Module:** [07_differentiable_trees_node_soft_gbdt_and_gam_nam.py](../../jax/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.py)
- **Native Rust Candle Track:** [07_differentiable_trees_node_soft_gbdt_and_gam_nam.rs](../../candle/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.rs)
- **Hardware WGSL Kernel:** [07_differentiable_trees_node_soft_gbdt_and_gam_nam.rs](../../kernels/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.rs)
- **Golden Reference Checkpoint:** [07_differentiable_trees_node_soft_gbdt_and_gam_nam.safetensors](../../weights/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 06. Naive Bayes, Bayesian Regression & GPs](06_naive_bayes_bayesian_regression_and_gaussian_processes.ipynb) | [Category Index](../README.md) | [Next: 08. Matrix Factorization (ALS, NMF, BPR & FM)](08_matrix_factorization_als_nmf_bpr_and_fm.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "07_differentiable_trees_node_soft_gbdt_and_gam_nam.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)

if __name__ == "__main__":
    build_notebook()
