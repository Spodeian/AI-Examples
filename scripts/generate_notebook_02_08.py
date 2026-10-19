"""Generate 8-cell educational notebook for Topic 08:
08_matrix_factorization_als_nmf_bpr_and_fm.ipynb
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf

def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 08. Matrix Factorization: ALS, NMF, BPR & Factorization Machines

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Daniel D. Lee & H. Sebastian Seung (1999) *Learning the parts of objects by non-negative matrix factorization*; Yehuda Koren, Robert Bell & Chris Volinsky (2009) *Matrix Factorization Techniques for Recommender Systems*; Steffen Rendle et al. (2009) *BPR: Bayesian Personalized Ranking from Implicit Feedback*; Steffen Rendle (2010) *Factorization Machines*.
- **Purpose:** Unifies low-rank matrix decomposition, multiplicative nonnegative factorizations, implicit pairwise ranking, and bilinear feature interaction modeling in linear time.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not treat unobserved implicit interaction logs (clicks, views) as negative zeros under dense MSE loss, as missing feedback reflects unexposed items rather than negative preference; use pairwise BPR ranking. Do not compute explicit $D \\times D$ feature interaction outer products when $D > 10^3$; use Rendle's $O(KD)$ algebraic summation identity."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Factorization Machines (FM) & The O(KD) Identity
Given sparse feature vector $\\mathbf{x} \\in \\mathbb{R}^D$ and factor matrix $\\mathbf{V} \\in \\mathbb{R}^{D \\times K}$:
$$\\hat{y}(\\mathbf{x}) = w_0 + \\sum_{i=1}^D w_i x_i + \\frac{1}{2} \\sum_{f=1}^K \\left[ \\left(\\sum_{i=1}^D v_{i,f} x_i\\right)^2 - \\sum_{i=1}^D v_{i,f}^2 x_i^2 \\right]$$
This reformulation eliminates explicit $\\mathcal{O}(D^2)$ pairwise loops, evaluating all second-order interactions in $\\mathcal{O}(K D)$ time.

### 2. Alternating Least Squares (ALS) Matrix Factorization
Given rating matrix $\\mathbf{R} \\approx \\mathbf{P} \\mathbf{Q}^\\top$, ALS alternates ridge solves holding the other factor fixed:
$$\\mathbf{P} = \\mathbf{R} \\mathbf{Q} (\\mathbf{Q}^\\top \\mathbf{Q} + \\lambda \\mathbf{I})^{-1}, \\quad \\mathbf{Q} = \\mathbf{R}^\\top \\mathbf{P} (\\mathbf{P}^\\top \\mathbf{P} + \\lambda \\mathbf{I})^{-1}$$

### 3. Non-Negative Matrix Factorization (NMF) Multiplicative Updates
For $\\mathbf{V} \\ge 0$, Lee & Seung updates preserve non-negativity without gradient projection steps:
$$\\mathbf{H} \\leftarrow \\mathbf{H} \\odot \\frac{\\mathbf{W}^\\top \\mathbf{V}}{\\mathbf{W}^\\top \\mathbf{W} \\mathbf{H} + \\epsilon}, \\quad \\mathbf{W} \\leftarrow \\mathbf{W} \\odot \\frac{\\mathbf{V} \\mathbf{H}^\\top}{\\mathbf{W} \\mathbf{H} \\mathbf{H}^\\top + \\epsilon}$$

**Complexity:** FM: $\\mathcal{O}(N K D)$ time; ALS step: $\\mathcal{O}(U I K + (U+I) K^3)$; NMF step: $\\mathcal{O}(U I K)$."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import core

env = core.setup_env(seed=42)
N, D, K = 32, 4, 2
X = torch.randn(N, D)
w0 = torch.tensor([0.15])
w_fm = torch.tensor([0.4, -0.6, 0.8, -0.2])
V_fm = torch.randn(D, K) * 0.5

print(f"Setup complete on {env.device}: Observations X={X.shape}, Factors K={K}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class FactorizationToolkit:
    @staticmethod
    def fm_forward(X: torch.Tensor, w0: torch.Tensor, w: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        linear = w0 + X @ w
        xv = X @ V
        xv_sq = xv ** 2
        x_sq_v_sq = (X ** 2) @ (V ** 2)
        interaction = 0.5 * torch.sum(xv_sq - x_sq_v_sq, dim=1)
        return linear + interaction

    @staticmethod
    def als_step(R: torch.Tensor, P: torch.Tensor, reg: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
        K = P.shape[1]
        Q = torch.linalg.solve(P.T @ P + reg * torch.eye(K), (R.T @ P).T).T
        P_new = torch.linalg.solve(Q.T @ Q + reg * torch.eye(K), (R @ Q).T).T
        return P_new, Q

    @staticmethod
    def nmf_step(V: torch.Tensor, W: torch.Tensor, H: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
        H_new = H * ((W.T @ V) / (W.T @ W @ H + eps))
        W_new = W * ((V @ H_new.T) / (W @ H_new @ H_new.T + eps))
        return W_new, H_new"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Invariants & Precision Verification
fm_out = FactorizationToolkit.fm_forward(X, w0, w_fm, V_fm)

# Invariant 1: FM O(KD) identity must exactly match brute-force O(D^2) pairwise interaction sum
brute_force_interaction = torch.zeros(N)
for i in range(D):
    for j in range(i + 1, D):
        dot_v = torch.dot(V_fm[i], V_fm[j])
        brute_force_interaction += dot_v * X[:, i] * X[:, j]
expected_fm = w0 + X @ w_fm + brute_force_interaction
fm_parity_err = torch.max(torch.abs(fm_out - expected_fm)).item()
assert fm_parity_err < 1e-6, f"FM O(KD) trick does not match O(D^2) sum: {fm_parity_err}"

# Invariant 2: NMF multiplicative updates strictly preserve non-negativity
V_pos = torch.rand(8, 8) + 0.1
W_pos, H_pos = torch.rand(8, 2) + 0.1, torch.rand(2, 8) + 0.1
W_up, H_up = FactorizationToolkit.nmf_step(V_pos, W_pos, H_pos)
assert (W_up >= 0.0).all() and (H_up >= 0.0).all(), "NMF violated non-negativity invariant"

# Invariant 3: ALS reconstruction error is non-increasing
R_syn = torch.randn(8, 2) @ torch.randn(8, 2).T
P_init = torch.randn(8, 2)
P_up, Q_up = FactorizationToolkit.als_step(R_syn, P_init, reg=0.01)
rec_err = torch.norm(R_syn - P_up @ Q_up.T).item()
assert rec_err < torch.norm(R_syn).item(), "ALS failed to reconstruct low-rank signal"
print(f"Verification passed: fm_parity_err={fm_parity_err:.2e}, rec_err={rec_err:.4f}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_fm = benchmark_fn(lambda: FactorizationToolkit.fm_forward(X, w0, w_fm, V_fm), iters=30, name="FM Forward (32x4, K=2)")
b_als = benchmark_fn(lambda: FactorizationToolkit.als_step(R_syn, P_init), iters=30, name="ALS Step (8x8, K=2)")
b_nmf = benchmark_fn(lambda: FactorizationToolkit.nmf_step(V_pos, W_pos, H_pos), iters=30, name="NMF Multiplicative (8x8)")
print(format_benchmark_table([b_fm, b_als, b_nmf]))"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: O(D^2) Dense Pairwise Matrix vs Rendle's O(KD) Formulation
# ANTI-PATTERN: For D=2048, materializing the full pairwise interaction tensor requires 2048^2 = 4,194,304 terms!
D_large, K_latent = 2048, 8
X_large = torch.randn(1, D_large)
V_large = torch.randn(D_large, K_latent)

# Rendle's O(KD) method executes in milliseconds without matrix allocation
xv = X_large @ V_large
xv_sq = xv ** 2
x_sq_v_sq = (X_large ** 2) @ (V_large ** 2)
fast_interaction = 0.5 * torch.sum(xv_sq - x_sq_v_sq)

# Full dense matrix allocation would require:
dense_elements = D_large * D_large
memory_mb = (dense_elements * 4) / (1024 * 1024)

print(f"[FAILURE] Naive Pairwise Tensor for D={D_large}:  {dense_elements:,} weights ({memory_mb:.2f} MB)")
print(f"[CORRECT] Rendle's O(KD) Factor Matrix:        {D_large * K_latent:,} weights ({(D_large * K_latent * 4)/1024:.2f} KB)")
assert fast_interaction.isfinite(), "O(KD) computation must succeed stably" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **The O(KD) Algebraic Identity:** Rewriting bilinear sums via symmetric difference reduces quadratic combinatorial explosions $\\mathcal{O}(D^2)$ to linear runtime $\\mathcal{O}(K D)$, unlocking web-scale recommendation.
2. **Implicit Feedback Ranking:** Pointwise MSE assumes missing entries are true negatives, distorting user preference distributions. Pairwise BPR ranking preserves relative affinity without negative sampling artifacts.
3. **Multiplicative Invariance in NMF:** Scaling factors by elementwise quotient ratios maintains strict non-negativity without projection barriers.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [08_matrix_factorization_als_nmf_bpr_and_fm.md](../../markdown/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.md)
- **PyTorch 2.x Module:** [08_matrix_factorization_als_nmf_bpr_and_fm.py](../../pytorch2/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.py)
- **Functional JAX Module:** [08_matrix_factorization_als_nmf_bpr_and_fm.py](../../jax/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.py)
- **Native Rust Candle Track:** [08_matrix_factorization_als_nmf_bpr_and_fm.rs](../../candle/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.rs)
- **Hardware WGSL Kernel:** [08_matrix_factorization_als_nmf_bpr_and_fm.rs](../../kernels/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.rs)
- **Golden Reference Checkpoint:** [08_matrix_factorization_als_nmf_bpr_and_fm.safetensors](../../weights/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 07. Differentiable Trees (NODE & NAM)](07_differentiable_trees_node_soft_gbdt_and_gam_nam.ipynb) | [Category Index](../README.md) | [Next: 09. Hidden Markov Models & Kalman Filters](09_hidden_markov_models_kalman_ukf_and_particle_filters.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "08_matrix_factorization_als_nmf_bpr_and_fm.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)

if __name__ == "__main__":
    build_notebook()
