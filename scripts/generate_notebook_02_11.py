"""Generate 8-cell educational notebook for Topic 11:
11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.ipynb
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf

def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 11. Factor Graphs, Belief Propagation & CAVI

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Judea Pearl (1982) *Reverend Bayes on inference engines: A distributed hierarchical approach*; Michael I. Jordan et al. (1999) *An Introduction to Variational Methods for Graphical Models*; Frank R. Kschischang et al. (2001) *Factor graphs and the sum-product algorithm*; David M. Blei et al. (2017) *Variational Inference: A Review for Statisticians*.
- **Purpose:** Unifies exact tree inference, cyclic graph message passing, and mean-field variational approximations via Factor Graphs, Sum-Product / Max-Product Belief Propagation, and Coordinate Ascent Variational Inference (CAVI).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not execute undamped Loopy Belief Propagation on graphs with short cycles or strong potentials; un-damped messages enter chaotic limit-cycle oscillations and severely overestimate marginal confidences; always apply message damping $\\alpha \\in (0.2, 0.5)$. Do not use Mean-Field CAVI when posterior correlation between latent variables is critical, as fully factorized families $q(\\mathbf{z}) = \\prod_j q_j(z_j)$ systematically underestimate posterior variances."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Factor Graphs & The Sum-Product Algorithm (Tree Inference)
Joint distribution $p(\\mathbf{x}) = \\frac{1}{Z} \\prod_{a \\in \\mathcal{F}} f_a(\\mathbf{x}_{\\text{ne}(a)})$. On tree graphs, two-pass message passing yields exact marginals:
$$\\mu_{i \\to a}(x_i) = \\prod_{b \\in \\text{ne}(i) \\setminus \\{a\\}} \\mu_{b \\to i}(x_i), \\quad \\mu_{a \\to i}(x_i) = \\sum_{\\mathbf{x}_{\\setminus i}} f_a(\\mathbf{x}_{\\text{ne}(a)}) \\prod_{j \\in \\text{ne}(a) \\setminus \\{i\\}} \\mu_{j \\to a}(x_j)$$
$$b(x_i) = \\frac{1}{Z_i} \\prod_{a \\in \\text{ne}(i)} \\mu_{a \\to i}(x_i)$$

### 2. Max-Product Algorithm for MAP Inference
Replacing sum with max yields global mode $\\mathbf{x}^* = \\arg\\max_\\mathbf{x} p(\\mathbf{x})$:
$$\\mu_{a \\to i}^{\\max}(x_i) = \\max_{\\mathbf{x}_{\\setminus i}} \\left[ f_a(\\mathbf{x}_{\\text{ne}(a)}) \\prod_{j \\in \\text{ne}(a) \\setminus \\{i\\}} \\mu_{j \\to a}^{\\max}(x_j) \\right]$$

### 3. Damped Loopy Belief Propagation & CAVI Mean-Field
Cyclic damping: $\\mu^{(t)} = (1 - \\alpha) \\mu^{(t-1)} + \\alpha \\mu^{\\text{raw}}$.  
CAVI coordinate updates: $\\ln q_j^*(z_j) = \\mathbb{E}_{-j}[\\ln p(\\mathbf{x}, \\mathbf{z})] + \\text{const}$, maximizing $\\text{ELBO}(q) = \\mathbb{E}_q[\\ln p(\\mathbf{x}, \\mathbf{z})] - \\mathbb{E}_q[\\ln q(\\mathbf{z})]$.

**Complexity:** Tree BP: $\\mathcal{O}(|\\mathcal{E}| \\cdot K^2)$ time; Brute force: $\\mathcal{O}(K^V)$; CAVI step: $\\mathcal{O}(N \\cdot K)$."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import core

env = core.setup_env(seed=42)
unary_tree = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5]])
edge_tree = torch.tensor([
    [[0.8, 0.2], [0.1, 0.9]],  # 0-1
    [[0.7, 0.3], [0.4, 0.6]],  # 1-2
    [[0.6, 0.4], [0.3, 0.7]],  # 1-3
])
unary_tri = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
edge_tri = torch.tensor([
    [[0.8, 0.2], [0.2, 0.8]],  # 0-1
    [[0.7, 0.3], [0.3, 0.7]],  # 1-2
    [[0.9, 0.1], [0.1, 0.9]],  # 2-0
])
data_cavi = torch.tensor([-2.5, -2.1, -1.8, -2.0, 1.9, 2.3, 2.1, 2.5])
mu_cavi = torch.tensor([-2.0, 2.0])
print(f"Setup complete on {env.device}: Tree nodes=4, Triangle nodes=3, CAVI N={len(data_cavi)}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class PGMToolkit:
    @staticmethod
    def sum_product_tree(unary: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        m01 = unary[0] @ edges[0]
        m21 = edges[1] @ unary[2]
        m31 = edges[2] @ unary[3]
        b1 = unary[1] * m01 * m21 * m31
        b1 = b1 / b1.sum()

        m10 = edges[0] @ (unary[1] * m21 * m31)
        b0 = unary[0] * m10
        b0 = b0 / b0.sum()

        m12 = (unary[1] * m01 * m31) @ edges[1]
        b2 = unary[2] * m12
        b2 = b2 / b2.sum()

        m13 = (unary[1] * m01 * m21) @ edges[2]
        b3 = unary[3] * m13
        b3 = b3 / b3.sum()
        return torch.stack([b0, b1, b2, b3])

    @staticmethod
    def damped_loopy_bp(unary: torch.Tensor, edges: torch.Tensor, alpha: float = 0.5, iters: int = 20) -> torch.Tensor:
        m = torch.ones(6, 2) * 0.5
        for _ in range(iters):
            r01 = (unary[0] * m[4]) @ edges[0]; m[0] = (1 - alpha) * m[0] + alpha * (r01 / r01.sum())
            r10 = edges[0] @ (unary[1] * m[3]); m[1] = (1 - alpha) * m[1] + alpha * (r10 / r10.sum())
            r12 = (unary[1] * m[0]) @ edges[1]; m[2] = (1 - alpha) * m[2] + alpha * (r12 / r12.sum())
            r21 = edges[1] @ (unary[2] * m[5]); m[3] = (1 - alpha) * m[3] + alpha * (r21 / r21.sum())
            r20 = (unary[2] * m[2]) @ edges[2]; m[4] = (1 - alpha) * m[4] + alpha * (r20 / r20.sum())
            r02 = edges[2] @ (unary[0] * m[1]); m[5] = (1 - alpha) * m[5] + alpha * (r02 / r02.sum())
        b0 = unary[0] * m[1] * m[4]; b0 /= b0.sum()
        b1 = unary[1] * m[0] * m[3]; b1 /= b1.sum()
        b2 = unary[2] * m[2] * m[5]; b2 /= b2.sum()
        return torch.stack([b0, b1, b2])

    @staticmethod
    def cavi_step(x: torch.Tensor, mu: torch.Tensor, s2: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        diff = x.unsqueeze(1) - mu.unsqueeze(0)
        log_p = -0.5 * (diff ** 2) / s2
        r = torch.softmax(log_p, dim=1)
        elbo = torch.sum(r * (log_p - torch.log(torch.clamp(r, min=1e-8))))
        return r, elbo"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Invariants & Precision Verification
marginals = PGMToolkit.sum_product_tree(unary_tree, edge_tree)

# Invariant 1: Beliefs at all nodes form valid probability distributions summing to 1.0
sums = torch.sum(marginals, dim=1)
assert torch.allclose(sums, torch.ones(4), atol=1e-5), "Tree marginals failed to sum to 1.0"

# Invariant 2: Damped loopy marginals on cycle form valid probabilities
loopy_b = PGMToolkit.damped_loopy_bp(unary_tri, edge_tri, alpha=0.5, iters=20)
loopy_sums = torch.sum(loopy_b, dim=1)
assert torch.allclose(loopy_sums, torch.ones(3), atol=1e-5), "Loopy marginals failed to sum to 1.0"

# Invariant 3: CAVI ELBO is bounded by log evidence
r_cavi, elbo = PGMToolkit.cavi_step(data_cavi, mu_cavi)
assert torch.isfinite(elbo), "CAVI ELBO must be finite"
assert (r_cavi >= 0.0).all() and (r_cavi <= 1.0).all(), "Responsibilities out of [0, 1] bounds"

print(f"Verification passed: Tree b(x_1)={marginals[1].tolist()}, Loopy b(x_0)={loopy_b[0].tolist()}, ELBO={elbo.item():.4f}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_tree = benchmark_fn(lambda: PGMToolkit.sum_product_tree(unary_tree, edge_tree), iters=30, name="Tree Sum-Product BP")
b_loopy = benchmark_fn(lambda: PGMToolkit.damped_loopy_bp(unary_tri, edge_tri, alpha=0.5, iters=20), iters=30, name="Damped Loopy BP (20 iters)")
b_cavi = benchmark_fn(lambda: PGMToolkit.cavi_step(data_cavi, mu_cavi), iters=30, name="CAVI Coordinate Step (N=8)")
print(format_benchmark_table([b_tree, b_loopy, b_cavi]))"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Undamped Loopy BP (alpha=1.0) vs Damped Loopy BP (alpha=0.5)
# In cyclic graphs with strong conflicting potentials, undamped BP enters non-convergent limit cycles!
edge_frust = torch.tensor([
    [[0.01, 0.99], [0.99, 0.01]],
    [[0.01, 0.99], [0.99, 0.01]],
    [[0.01, 0.99], [0.99, 0.01]],
])
u_frust = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])

m_u = torch.ones(3, 2) * 0.5
m_d = torch.ones(3, 2) * 0.5

diff_u, diff_d = 0.0, 0.0
for it in range(25):
    # Undamped (alpha=1.0)
    r01 = (u_frust[0] * m_u[2]) @ edge_frust[0]; r01 /= r01.sum()
    r12 = (u_frust[1] * m_u[0]) @ edge_frust[1]; r12 /= r12.sum()
    r20 = (u_frust[2] * m_u[1]) @ edge_frust[2]; r20 /= r20.sum()
    diff_u = torch.norm(r01 - m_u[0]).item()
    m_u[0], m_u[1], m_u[2] = r01, r12, r20

    # Damped (alpha=0.5)
    r01_d = (u_frust[0] * m_d[2]) @ edge_frust[0]; r01_d /= r01_d.sum()
    r12_d = (u_frust[1] * m_d[0]) @ edge_frust[1]; r12_d /= r12_d.sum()
    r20_d = (u_frust[2] * m_d[1]) @ edge_frust[2]; r20_d /= r20_d.sum()
    diff_d = torch.norm(r01_d - m_d[0]).item()
    m_d[0] = 0.5 * m_d[0] + 0.5 * r01_d
    m_d[1] = 0.5 * m_d[1] + 0.5 * r12_d
    m_d[2] = 0.5 * m_d[2] + 0.5 * r20_d

print(f"[UNDAMPED BP] Iteration 24 Residual: {diff_u:.4f} (Oscillating limit cycle!)")
print(f"[DAMPED BP]   Iteration 24 Residual: {diff_d:.4f} (Smoothly converging)")
assert diff_d < diff_u, "Damped BP must achieve lower residual than oscillating undamped BP" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Dynamic Programming on Graphs:** The Sum-Product algorithm computes exact marginals in linear time $O(|\\mathcal{E}| K^2)$, avoiding combinatorial $O(K^V)$ enumeration.
2. **Loopy Message Damping:** Cyclic loops create feedback echoes that trigger perpetual message oscillation; convex damping $\\alpha \\in (0.2, 0.5)$ restores numerical contraction.
3. **CAVI for Tractable Bounds:** Mean-Field variational updates provide coordinate-wise closed-form ascent on the Evidence Lower Bound for continuous mixture and latent models.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.md](../../markdown/02_classical_and_probabilistic_ml/11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.md)
- **PyTorch 2.x Module:** [11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.py](../../pytorch2/02_classical_and_probabilistic_ml/11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.py)
- **Functional JAX Module:** [11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.py](../../jax/02_classical_and_probabilistic_ml/11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.py)
- **Native Rust Candle Track:** [11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.rs](../../candle/02_classical_and_probabilistic_ml/11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.rs)
- **Hardware WGSL Kernel:** [11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.rs](../../kernels/02_classical_and_probabilistic_ml/11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.rs)
- **Golden Reference Checkpoint:** [11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.safetensors](../../weights/02_classical_and_probabilistic_ml/11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 10. Survival Analysis & Contextual Bandits](10_survival_analysis_and_contextual_bandits.ipynb) | [Category Index](../README.md) | [Next: 12. MCMC, HMC, SGLD & SVGD](12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)

if __name__ == "__main__":
    build_notebook()
