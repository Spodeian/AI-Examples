import nbformat as nbf
from pathlib import Path

cells = []

# Cell 1: Markdown
c1 = """# 05. Dimensionality Reduction: PCA, LDA, ICA & Manifold Learning

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Pearson (1901) *On lines and planes of closest fit to systems of points in space*; Fisher (1936) *The use of multiple measurements in taxonomic problems*; Schölkopf, Smola & Müller (1998) *Nonlinear component analysis as a kernel eigenvalue problem*; van der Maaten & Hinton (2008) *Visualizing data using t-SNE*.
- **Purpose:** Implements linear and non-linear manifold projection via SVD-based PCA, supervised Fisher Linear Discriminant Analysis (LDA), and double-centered Kernel PCA.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not compute PCA on uncentered data, as the first component spuriously points to the dataset mean offset rather than maximum variance. Do not apply linear PCA to non-linear Swiss-roll or ring manifolds where geodesic distance is ignored; use Kernel PCA, t-SNE, or UMAP."""
cells.append(nbf.v4.new_markdown_cell(c1))

# Cell 2: Markdown
c2 = """## Mathematical Foundations & Governing Equations

### 1. Principal Component Analysis via Thin SVD
Given centered observations $\\tilde{\\mathbf{X}} = \\mathbf{X} - \\mathbf{1}\\bar{\\mathbf{x}}^\\top \\in \\mathbb{R}^{N \\times D}$, thin SVD satisfies:
$$\\tilde{\\mathbf{X}} = \\mathbf{U} \\mathbf{S} \\mathbf{V}^\\top, \\quad \\mathbf{Z} = \\tilde{\\mathbf{X}} \\mathbf{V}_{:, 1:K} = \\mathbf{U}_{:, 1:K} \\mathbf{S}_{1:K}$$
The variance explained by component $k$ is $\\lambda_k = S_k^2 / (N - 1)$.

### 2. Fisher's Linear Discriminant Analysis (LDA)
For binary class targets $y \\in \\{0, 1\\}$, the optimal direction $\\mathbf{w}$ maximizing Rayleigh quotient $\\frac{\\mathbf{w}^\\top \\mathbf{S}_B \\mathbf{w}}{\\mathbf{w}^\\top \\mathbf{S}_W \\mathbf{w}}$ is:
$$\\mathbf{w}^* = (\\mathbf{S}_W + \\epsilon \\mathbf{I})^{-1} (\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_0), \\quad \\mathbf{S}_W = \\sum_{c \\in \\{0, 1\\}} \\sum_{i \\in \\mathcal{C}_c} (\\mathbf{x}_i - \\boldsymbol{\\mu}_c)(\\mathbf{x}_i - \\boldsymbol{\\mu}_c)^\\top$$

**Asymptotic Complexity:** SVD PCA: $\\mathcal{O}(N D \\min(N, D))$ time; Fisher LDA: $\\mathcal{O}(N D + D^3)$ time; Kernel PCA: $\\mathcal{O}(N^3)$ time, $\\mathcal{O}(N^2)$ memory."""
cells.append(nbf.v4.new_markdown_cell(c2))

# Cell 3: Code
c3 = """import torch
import core

env = core.setup_env(seed=42)
N, D, K = 64, 8, 3
w_latent = torch.randn(D, K)
z_true = torch.randn(N, K)
X = z_true @ w_latent.T + 0.05 * torch.randn(N, D)
y_cls = (X[:, 0] > 0.0).long()

print(f"Setup complete on {env.device}: Observations X={X.shape}, Classes={len(torch.unique(y_cls))}")"""
cells.append(nbf.v4.new_code_cell(c3))

# Cell 4: Code
c4 = """class ReductionToolkit:
    @staticmethod
    def pca(X: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.mean(X, dim=0, keepdim=True)
        X_c = X - mean
        U, S, Vt = torch.linalg.svd(X_c, full_matrices=False)
        V_top = Vt[:k, :].T
        proj = X_c @ V_top
        return proj, V_top, mean

    @staticmethod
    def lda(X: torch.Tensor, y: torch.Tensor, reg: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.mean(X, dim=0, keepdim=True)
        X_c = X - mean
        D = X.shape[1]
        X0, X1 = X_c[y == 0], X_c[y == 1]
        mu0, mu1 = X0.mean(dim=0), X1.mean(dim=0)
        Sw = (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1) + reg * torch.eye(D, device=X.device)
        w = torch.linalg.solve(Sw, (mu1 - mu0).unsqueeze(1)).squeeze(1)
        w_norm = w / torch.norm(w)
        return X_c @ w_norm, w_norm

    @staticmethod
    def kernel_pca(X: torch.Tensor, k: int = 2, gamma: float = 0.1) -> torch.Tensor:
        N = X.shape[0]
        K = torch.exp(-gamma * (torch.cdist(X, X) ** 2))
        one_n = torch.full((N, N), 1.0 / N, device=X.device)
        K_tilde = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        eigvals, eigvecs = torch.linalg.eigh(K_tilde)
        eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)
        alphas = eigvecs[:, :k] / torch.sqrt(torch.clamp(eigvals[:k], min=1e-6))
        return K_tilde @ alphas"""
cells.append(nbf.v4.new_code_cell(c4))

# Cell 5: Code
c5 = """# Quantitative Invariants & Parity Verification
proj_pca, V_top, mean_x = ReductionToolkit.pca(X, k=3)
proj_lda, w_lda = ReductionToolkit.lda(X, y_cls, reg=1e-3)
proj_kpca = ReductionToolkit.kernel_pca(X, k=3, gamma=0.1)

# Invariant 1: Orthogonality of Principal Components (V_top^T V_top = I)
ortho_err = torch.max(torch.abs(V_top.T @ V_top - torch.eye(3))).item()
assert ortho_err < 1e-5, f"PCA components are not orthogonal: {ortho_err}"

# Invariant 2: Zero mean of centered PCA projections
pca_mean_norm = torch.norm(torch.mean(proj_pca, dim=0)).item()
assert pca_mean_norm < 1e-5, f"PCA projections are not zero-centered: {pca_mean_norm}"

# Invariant 3: Fisher LDA class separation ratio
mu0_proj = proj_lda[y_cls == 0].mean().item()
mu1_proj = proj_lda[y_cls == 1].mean().item()
separation = abs(mu1_proj - mu0_proj)
assert separation > 0.1, f"LDA class separation too low: {separation}"
print(f"Verification passed: ortho_err={ortho_err:.2e}, pca_mean_norm={pca_mean_norm:.2e}, separation={separation:.4f}")"""
cells.append(nbf.v4.new_code_cell(c5))

# Cell 6: Code
c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_pca = benchmark_fn(lambda: ReductionToolkit.pca(X, k=3), iters=30, name="SVD PCA (64x8 -> 3)")
b_lda = benchmark_fn(lambda: ReductionToolkit.lda(X, y_cls, 1e-3), iters=30, name="Fisher LDA (Binary)")
b_kpca = benchmark_fn(lambda: ReductionToolkit.kernel_pca(X, k=3, gamma=0.1), iters=20, name="Kernel PCA (64x64)")
print(format_benchmark_table([b_pca, b_lda, b_kpca]))"""
cells.append(nbf.v4.new_code_cell(c6))

# Cell 7: Code
c7 = """# Pathological Misuse: Uncentered SVD PCA vs Mean-Centered Principal Coordinates
# ANTI-PATTERN: Forgetting mean-centering forces 1st principal axis to point to data centroid
X_offset = X + 50.0 # Large spatial offset
_, S_uncentered, Vt_uncentered = torch.linalg.svd(X_offset, full_matrices=False)
v1_uncentered = Vt_uncentered[0]

# Centroid unit direction
centroid_dir = torch.mean(X_offset, dim=0) / torch.norm(torch.mean(X_offset, dim=0))
cos_sim_failure = torch.abs(torch.dot(v1_uncentered, centroid_dir)).item()

# Corrected: SVD on mean-centered data aligns strictly with intrinsic variance
X_centered = X_offset - torch.mean(X_offset, dim=0, keepdim=True)
_, S_centered, Vt_centered = torch.linalg.svd(X_centered, full_matrices=False)
cos_sim_correct = torch.abs(torch.dot(Vt_centered[0], centroid_dir)).item()

print(f"[FAILURE] Uncentered 1st axis alignment with mean centroid: {cos_sim_failure:.4f} (Spurious Offset Bias)")
print(f"[CORRECT] Centered 1st axis alignment with mean centroid: {cos_sim_correct:.4f} (True Manifold Variance)")
assert cos_sim_failure > 0.99, "Uncentered PCA must erroneously collapse onto mean offset"
assert cos_sim_correct < 0.35, "Centered PCA must decouple from absolute translation" """
cells.append(nbf.v4.new_code_cell(c7))

# Cell 8: Markdown
c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **The Translation Invariance Axiom:** PCA is defined exclusively on centralized covariance structures; omitting mean-centering converts the first eigenvector into a redundant translation tracker.
2. **Fisher Criterion for Compression:** LDA maximizes class margin separation per projection dimension, providing optimal supervised compression for downstream classifiers.
3. **Double-Centering in RKHS:** Kernel PCA implicit centering $H K H$ guarantees that nonlinear high-dimensional RKHS feature coordinates have empirical zero expectation.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [05_dimensionality_reduction_pca_lda_ica_tsne_umap.md](../../markdown/02_classical_and_probabilistic_ml/05_dimensionality_reduction_pca_lda_ica_tsne_umap.md)
- **PyTorch 2.x Module:** [05_dimensionality_reduction_pca_lda_ica_tsne_umap.py](../../pytorch2/02_classical_and_probabilistic_ml/05_dimensionality_reduction_pca_lda_ica_tsne_umap.py)
- **Functional JAX Module:** [05_dimensionality_reduction_pca_lda_ica_tsne_umap.py](../../jax/02_classical_and_probabilistic_ml/05_dimensionality_reduction_pca_lda_ica_tsne_umap.py)
- **Native Rust Candle Track:** [05_dimensionality_reduction_pca_lda_ica_tsne_umap.rs](../../candle/02_classical_and_probabilistic_ml/05_dimensionality_reduction_pca_lda_ica_tsne_umap.rs)
- **Hardware WGSL Kernel:** [05_dimensionality_reduction_pca_lda_ica_tsne_umap.rs](../../kernels/02_classical_and_probabilistic_ml/05_dimensionality_reduction_pca_lda_ica_tsne_umap.rs)
- **Golden Reference Checkpoint:** [05_dimensionality_reduction_pca_lda_ica_tsne_umap.safetensors](../../weights/02_classical_and_probabilistic_ml/05_dimensionality_reduction_pca_lda_ica_tsne_umap.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 04. Clustering (k-Means & GMM)](04_clustering_kmeans_spectral_and_gmm_em.ipynb) | [Category Index](../README.md) | [Next: 06. Naive Bayes, Bayesian Regression & GPs](06_naive_bayes_bayesian_regression_and_gaussian_processes.ipynb) &rarr;"""
cells.append(nbf.v4.new_markdown_cell(c8))

nb = nbf.v4.new_notebook(cells=cells)
out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "05_dimensionality_reduction_pca_lda_ica_tsne_umap.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote notebook to", out_path)
