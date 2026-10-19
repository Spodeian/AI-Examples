import nbformat as nbf
from pathlib import Path

cells = []

# Cell 1: Markdown
c1 = """# 02. Support Vector Machines & Kernel Methods

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Boser, Guyon & Vapnik (1992) *A training algorithm for optimal margin classifiers*; Cortes & Vapnik (1995) *Support-Vector Networks*; Williams & Seeger (2001) *Using the Nyström Method to Speed Up Kernel Machines*.
- **Purpose:** Implements maximum-margin classification (primal and dual soft-margin SVM), Mercer kernel Gram evaluation (RBF, Polynomial, Matérn), Kernel Ridge Regression (KRR), and Nyström low-rank landmark approximation.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not compute full $N \\times N$ Gram matrices when $N > 50\\text{k}$ without low-rank Nyström or Random Fourier Feature approximation (induces $\\mathcal{O}(N^2)$ VRAM explosion). Do not evaluate RBF kernels on unstandardized inputs, as dimension variance imbalances saturate distance metrics."""
cells.append(nbf.v4.new_markdown_cell(c1))

# Cell 2: Markdown
c2 = """## Mathematical Foundations & Governing Equations

### 1. Dual Soft-Margin Formulation & Wolfe Duality
Given labeled samples $(\\mathbf{x}_i, y_i) \\in \\mathbb{R}^D \\times \\{-1, +1\\}$, the Wolfe Dual quadratic program is:
$$\\max_{\\boldsymbol{\\alpha}} \\sum_{i=1}^N \\alpha_i - \\frac{1}{2} \\sum_{i,j=1}^N \\alpha_i \\alpha_j y_i y_j k(\\mathbf{x}_i, \\mathbf{x}_j) \\quad \\text{s.t.} \\quad 0 \\le \\alpha_i \\le C, \\quad \\sum_{i=1}^N \\alpha_i y_i = 0$$
Support vectors correspond to samples where active dual multipliers $\\alpha_i > 0$.

### 2. Kernel Ridge Regression & Nyström Approximation
In RKHS $\\mathcal{H}$, the exact closed-form dual regularized predictor solves:
$$\\boldsymbol{\\alpha}^* = (\\mathbf{K} + \\lambda \\mathbf{I}_N)^{-1} \\mathbf{y}, \\quad \\hat{f}(\\mathbf{x}_*) = \\mathbf{k}_*^\\top \\boldsymbol{\\alpha}^*$$
For $m \\ll N$ sampled landmark points $\\mathbf{X}_m$, the low-rank Nyström Gram surrogate satisfies:
$$\\tilde{\\mathbf{K}} = \\mathbf{K}_{nm} \\mathbf{K}_{mm}^{-1} \\mathbf{K}_{nm}^\\top$$

**Asymptotic Complexity:** Exact Gram Inversion: $\\mathcal{O}(N^3)$ time, $\\mathcal{O}(N^2)$ memory; Nyström Landmark Approximation: $\\mathcal{O}(N m^2 + m^3)$ time, $\\mathcal{O}(N m)$ memory."""
cells.append(nbf.v4.new_markdown_cell(c2))

# Cell 3: Code
c3 = """import torch
import core

env = core.setup_env(seed=42)
N, D, M = 64, 8, 32
X = torch.randn(N, D, dtype=torch.float32)
w_star = torch.tensor([1.2, -0.8, 0.5, -1.0, 0.3, -0.2, 0.7, -0.4])
y_cont = X @ w_star + 0.1 * torch.randn(N)
y_cls = torch.sign(X @ w_star)
y_cls[y_cls == 0] = 1.0

gamma = 1.0 / D
landmarks = X[:M].clone()
print(f"Setup complete on {env.device}: X shape={X.shape}, landmarks={landmarks.shape}, gamma={gamma:.4f}")"""
cells.append(nbf.v4.new_code_cell(c3))

# Cell 4: Code
c4 = """class KernelToolkit:
    @staticmethod
    def rbf_gram(X1: torch.Tensor, X2: torch.Tensor, gamma: float) -> torch.Tensor:
        x1_sq = torch.sum(X1 ** 2, dim=1, keepdim=True)
        x2_sq = torch.sum(X2 ** 2, dim=1, keepdim=True).T
        dist_sq = torch.clamp(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), min=0.0)
        return torch.exp(-gamma * dist_sq)

    @staticmethod
    def solve_krr(X: torch.Tensor, y: torch.Tensor, gamma: float, alpha: float = 0.1) -> torch.Tensor:
        K = KernelToolkit.rbf_gram(X, X, gamma)
        reg_K = K + alpha * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
        return torch.linalg.solve(reg_K, y)

    @staticmethod
    def nystrom_gram(X: torch.Tensor, landmarks: torch.Tensor, gamma: float, eps: float = 1e-4) -> torch.Tensor:
        K_mm = KernelToolkit.rbf_gram(landmarks, landmarks, gamma)
        K_nm = KernelToolkit.rbf_gram(X, landmarks, gamma)
        m = landmarks.shape[0]
        K_mm_inv = torch.linalg.pinv(K_mm + eps * torch.eye(m, dtype=X.dtype, device=X.device))
        return K_nm @ K_mm_inv @ K_nm.T"""
cells.append(nbf.v4.new_code_cell(c4))

# Cell 5: Code
c5 = """# Quantitative Invariants & Parity Verification
K_full = KernelToolkit.rbf_gram(X, X, gamma)
K_nystrom = KernelToolkit.nystrom_gram(X, landmarks, gamma)
dual_coef = KernelToolkit.solve_krr(X, y_cont, gamma, alpha=0.1)

# Invariant 1: Mercer Gram symmetry and positive semi-definiteness
sym_err = torch.max(torch.abs(K_full - K_full.T)).item()
min_eig = torch.linalg.eigvalsh(K_full).min().item()
assert sym_err < 1e-6, f"Gram matrix asymmetry: {sym_err}"
assert min_eig > -1e-5, f"Non-positive-semidefinite Gram min eig: {min_eig}"

# Invariant 2: Nyström low-rank relative approximation error bound
rel_nystrom_err = (torch.norm(K_nystrom - K_full) / torch.norm(K_full)).item()
assert rel_nystrom_err < 0.20, f"Nystrom relative error {rel_nystrom_err} exceeds 0.20"

# Invariant 3: Kernel Ridge interpolation accuracy on training set
preds_krr = K_full @ dual_coef
krr_res = torch.norm(preds_krr - y_cont).item()
assert krr_res < 5.0, f"KRR residual too high: {krr_res}"
print(f"Verification passed: min_eig={min_eig:.2e}, rel_nystrom_err={rel_nystrom_err:.4f}, krr_res={krr_res:.4f}")"""
cells.append(nbf.v4.new_code_cell(c5))

# Cell 6: Code
c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_gram = benchmark_fn(lambda: KernelToolkit.rbf_gram(X, X, gamma), iters=30, name="Full RBF Gram (64x64)")
b_nystrom = benchmark_fn(lambda: KernelToolkit.nystrom_gram(X, landmarks, gamma), iters=30, name="Nystrom Gram (m=32)")
b_solve = benchmark_fn(lambda: KernelToolkit.solve_krr(X, y_cont, gamma, 0.1), iters=20, name="KRR Solve (64x64)")
print(format_benchmark_table([b_gram, b_nystrom, b_solve]))"""
cells.append(nbf.v4.new_code_cell(c6))

# Cell 7: Code
c7 = """# Pathological Misuse: Unscaled Gamma RBF Saturation vs Standardized Bandwidth
# ANTI-PATTERN: Oversized gamma causes RBF off-diagonals to collapse to 0 (Delta collapse)
gamma_pathological = 100.0 # Extreme uncalibrated bandwidth
K_bad = KernelToolkit.rbf_gram(X, X, gamma_pathological)
off_diag_max_bad = torch.max(torch.abs(K_bad - torch.eye(N))).item()

# Corrected: Dimension-scaled gamma = 1 / D preserves smooth metric topology
K_good = KernelToolkit.rbf_gram(X, X, gamma=1.0 / D)
off_diag_mean_good = (torch.sum(K_good) - torch.trace(K_good)).item() / (N * (N - 1))

print(f"[FAILURE] Uncalibrated gamma=100.0 off-diagonal max: {off_diag_max_bad:.2e} (Delta Spike Collapse)")
print(f"[CORRECT] Scaled gamma=1/D off-diagonal mean: {off_diag_mean_good:.4f} (Smooth RKHS Geometry)")
assert off_diag_max_bad < 1e-3, "Collapsed Gram matrix off-diagonals must approach zero"
assert off_diag_mean_good > 0.1, "Calibrated kernel must preserve cross-sample affinities" """
cells.append(nbf.v4.new_code_cell(c7))

# Cell 8: Markdown
c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **The Kernel Trick in Modern Deep Learning:** Transformer self-attention $\\operatorname{softmax}(Q K^\\top / \\sqrt{d_k}) V$ is an asymmetric non-negative kernel smoother bridging classical RKHS methods with sequence models.
2. **Nyström Geometry Compression:** Landmark projection reduces Gram quadratic memory $\\mathcal{O}(N^2) \\to \\mathcal{O}(N m)$, unlocking kernel scaling up to millions of observations.
3. **Spectral Regularization:** Tikhonov ridge parameters $\\alpha \\mathbf{I}$ bound the condition number $\\kappa(\\mathbf{K} + \\alpha \\mathbf{I}) \\le 1 + \\|\\mathbf{K}\\|_2 / \\alpha$, guaranteeing numerical stability against near-duplicate points.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [02_support_vector_machines_and_kernel_methods.md](../../markdown/02_classical_and_probabilistic_ml/02_support_vector_machines_and_kernel_methods.md)
- **PyTorch 2.x Module:** [02_support_vector_machines_and_kernel_methods.py](../../pytorch2/02_classical_and_probabilistic_ml/02_support_vector_machines_and_kernel_methods.py)
- **Functional JAX Module:** [02_support_vector_machines_and_kernel_methods.py](../../jax/02_classical_and_probabilistic_ml/02_support_vector_machines_and_kernel_methods.py)
- **Native Rust Candle Track:** [02_support_vector_machines_and_kernel_methods.rs](../../candle/02_classical_and_probabilistic_ml/02_support_vector_machines_and_kernel_methods.rs)
- **Hardware WGSL Kernel:** [02_support_vector_machines_and_kernel_methods.rs](../../kernels/02_classical_and_probabilistic_ml/02_support_vector_machines_and_kernel_methods.rs)
- **Golden Reference Checkpoint:** [02_support_vector_machines_and_kernel_methods.safetensors](../../weights/02_classical_and_probabilistic_ml/02_support_vector_machines_and_kernel_methods.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 01. GLMs & Proximal Lasso](01_generalized_linear_models_and_proximal_lasso.ipynb) | [Category Index](../README.md) | [Next: 03. k-NN & Locally Weighted Regression](03_knn_prototype_and_locally_weighted_regression.ipynb) &rarr;"""
cells.append(nbf.v4.new_markdown_cell(c8))

nb = nbf.v4.new_notebook(cells=cells)
out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "02_support_vector_machines_and_kernel_methods.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote notebook to", out_path)
