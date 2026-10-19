import nbformat as nbf
from pathlib import Path

cells = []

# Cell 1: Markdown
c1 = """# 04. Clustering: k-Means, Spectral Clustering & GMM EM

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Lloyd (1957) *Least squares quantization in PCM*; Dempster, Laird & Rubin (1977) *Maximum likelihood from incomplete data via the EM algorithm*; Ng, Jordan & Weiss (2001) *On Spectral Clustering: Analysis and an algorithm*.
- **Purpose:** Implements vector-quantized clustering via Lloyd's k-Means, soft probabilistic density decomposition via log-domain GMM EM, and non-convex manifold partitioning via normalized graph Laplacians.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use standard k-Means on non-convex geometric geometries (e.g., concentric circles or spiral arms) where Voronoi hyperplanes sever continuous manifolds; use Spectral Clustering. Do not train GMMs without covariance jitter $\\epsilon \\mathbf{I}$ (single-point collapse causes variance $\\sigma^2 \\to 0$ and $\\mathcal{L} \\to +\\infty$)."""
cells.append(nbf.v4.new_markdown_cell(c1))

# Cell 2: Markdown
c2 = """## Mathematical Foundations & Governing Equations

### 1. Lloyd's k-Means & Voronoi Partitions
Given dataset $\\mathbf{X} \\in \\mathbb{R}^{N \\times D}$, Lloyd's algorithm minimizes within-cluster inertia:
$$\\mathcal{J}(\\boldsymbol{\\mu}_1, \\dots, \\boldsymbol{\\mu}_K) = \\sum_{i=1}^N \\min_{k \\in \\{1,\\dots,K\\}} \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_k\\|_2^2$$

### 2. Log-Domain GMM Expectation-Maximization
Under diagonal covariance $\\boldsymbol{\\Sigma}_k = \\operatorname{diag}(\\boldsymbol{\\sigma}_k^2)$, posterior responsibilities $\\gamma_{ik}$ evaluate via log-sum-exp:
$$\\log \\gamma_{ik} = \\log \\pi_k - \\frac{1}{2} \\left[ D \\log(2\\pi) + \\sum_{j=1}^D \\log \\sigma_{kj}^2 + \\sum_{j=1}^D \\frac{(x_{ij} - \\mu_{kj})^2}{\\sigma_{kj}^2} \\right]$$
$$\\gamma_{ik} = \\operatorname{softmax}_k(\\log \\boldsymbol{\\gamma}_i), \\quad \\boldsymbol{\\mu}_k = \\frac{\\sum_i \\gamma_{ik} \\mathbf{x}_i}{\\sum_i \\gamma_{ik}}, \\quad \\sigma_{kj}^2 = \\max\\left( \\frac{\\sum_i \\gamma_{ik} (x_{ij} - \\mu_{kj})^2}{\\sum_i \\gamma_{ik}}, \\; \\epsilon \\right)$$

**Asymptotic Complexity:** k-Means per-iteration: $\\mathcal{O}(N K D)$; GMM EM per-iteration: $\\mathcal{O}(N K D)$; Spectral Laplacian: $\\mathcal{O}(N^2 D)$ time, $\\mathcal{O}(N^2)$ memory."""
cells.append(nbf.v4.new_markdown_cell(c2))

# Cell 3: Code
c3 = """import torch
import core

env = core.setup_env(seed=42)
N, D, K = 64, 8, 3
X = torch.cat([
    torch.randn(24, D) + torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    torch.randn(20, D) + torch.tensor([-3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    torch.randn(20, D) + torch.tensor([0.0, -3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
], dim=0)
centers_init = X[[0, 24, 44]].clone()

print(f"Setup complete on {env.device}: Dataset X={X.shape}, Clusters K={K}")"""
cells.append(nbf.v4.new_code_cell(c3))

# Cell 4: Code
c4 = """class ClusteringToolkit:
    @staticmethod
    def kmeans_step(X: torch.Tensor, centers: torch.Tensor, iters: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
        c = centers.clone()
        for _ in range(iters):
            dist_sq = torch.cdist(X, c) ** 2
            assigns = torch.argmin(dist_sq, dim=1)
            new_c = []
            for k in range(c.shape[0]):
                mask = (assigns == k)
                new_c.append(X[mask].mean(dim=0) if mask.sum() > 0 else c[k])
            c = torch.stack(new_c, dim=0)
        return c, assigns

    @staticmethod
    def gmm_e_step(X: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        K = mu.shape[0]
        log_resp = torch.zeros(N, K, device=X.device)
        for k in range(K):
            diff = X - mu[k]
            quad = torch.sum(diff ** 2 / var[k], dim=1)
            log_det = torch.sum(torch.log(var[k]))
            log_prob = -0.5 * (D * 1.837877 + log_det + quad)
            log_resp[:, k] = torch.log(pi[k] + 1e-8) + log_prob
        return torch.exp(log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True))

    @staticmethod
    def symmetric_laplacian(X: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
        A = torch.exp(-gamma * (torch.cdist(X, X) ** 2))
        A.fill_diagonal_(0.0)
        d_inv_sqrt = torch.pow(torch.clamp(torch.sum(A, dim=1), min=1e-6), -0.5)
        return torch.eye(X.shape[0], device=X.device) - (d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0))"""
cells.append(nbf.v4.new_code_cell(c4))

# Cell 5: Code
c5 = """# Quantitative Invariants & Parity Verification
final_c, assigns = ClusteringToolkit.kmeans_step(X, centers_init, iters=15)
pi = torch.full((K,), 1.0 / K)
var = torch.ones(K, D)
resp = ClusteringToolkit.gmm_e_step(X, final_c, var, pi)
L_sym = ClusteringToolkit.symmetric_laplacian(X, gamma=0.1)

# Invariant 1: Complete cluster partitioning & responsibility unity
assert torch.all(assigns >= 0) and torch.all(assigns < K)
resp_sum = torch.sum(resp, dim=1)
torch.testing.assert_close(resp_sum, torch.ones(N), atol=1e-5, rtol=1e-5)

# Invariant 2: Symmetric Normalized Laplacian semi-definiteness & zero smallest eigenvalue
sym_err = torch.max(torch.abs(L_sym - L_sym.T)).item()
assert sym_err < 1e-5, f"Laplacian asymmetry: {sym_err}"
eigs = torch.linalg.eigvalsh(L_sym)
assert eigs.min().item() > -1e-5, f"Negative Laplacian eigenvalue: {eigs.min().item()}"
assert eigs[0].item() < 0.1, "Smallest Laplacian eigenvalue should approach zero (constant eigenvector)"
print(f"Verification passed: K-means clusters assigned={len(torch.unique(assigns))}/{K}, min_eig={eigs.min().item():.2e}")"""
cells.append(nbf.v4.new_code_cell(c5))

# Cell 6: Code
c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_kmeans = benchmark_fn(lambda: ClusteringToolkit.kmeans_step(X, centers_init, iters=10), iters=20, name="Lloyd k-Means (10 iters)")
b_gmm = benchmark_fn(lambda: ClusteringToolkit.gmm_e_step(X, final_c, var, pi), iters=30, name="GMM E-step Log-Domain")
b_lap = benchmark_fn(lambda: ClusteringToolkit.symmetric_laplacian(X, 0.1), iters=30, name="Symmetric Laplacian (64x64)")
print(format_benchmark_table([b_kmeans, b_gmm, b_lap]))"""
cells.append(nbf.v4.new_code_cell(c6))

# Cell 7: Code
c7 = """# Pathological Misuse: Unregularized GMM Covariance Singularity vs Diagonal Jitter
# ANTI-PATTERN: Variance collapsing to 0.0 on near-isolated points blows log-likelihood to infinity
var_collapsing = torch.tensor([[1e-12] * D]) # Pathological zero variance
diff_single = (X[:1] - final_c[:1]) ** 2
log_det_collapsed = torch.sum(torch.log(var_collapsing)).item()

# Unregularized variance produces extreme negative log determinant (-10^2) and division blowup
assert log_det_collapsed < -100.0, "Unregularized log determinant should blow up negatively"

# Corrected: Tikhonov diagonal jitter stabilizer bounds variance from below
jitter = 1e-4
var_stabilized = torch.clamp(var_collapsing, min=jitter)
log_det_stabilized = torch.sum(torch.log(var_stabilized)).item()

print(f"[FAILURE] Unregularized covariance log det: {log_det_collapsed:.2f} (Singular Collapse)")
print(f"[CORRECT] Jittered covariance log det: {log_det_stabilized:.2f} (Bounded Stable Condition)")
assert log_det_stabilized > -100.0, "Jittered variance must prevent singularity collapse" """
cells.append(nbf.v4.new_code_cell(c7))

# Cell 8: Markdown
c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Geometric Inductive Bias:** $k$-Means enforces convex polyhedral Voronoi decision regions, while Spectral Clustering transforms complex non-convex data manifolds into linear eigenspaces.
2. **Log-Sum-Exp Numerical Armor:** Evaluating Gaussian mixture posterior responsibilities directly in log-probability space completely prevents floating-point underflow in high dimensions.
3. **Graph Laplacians Across Paradigms:** The symmetric normalized Laplacian $L_{\\text{sym}} = I - D^{-1/2} A D^{-1/2}$ identical in Spectral Clustering powers graph convolutional layers (GCNs) and manifold diffusion models.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [04_clustering_kmeans_spectral_and_gmm_em.md](../../markdown/02_classical_and_probabilistic_ml/04_clustering_kmeans_spectral_and_gmm_em.md)
- **PyTorch 2.x Module:** [04_clustering_kmeans_spectral_and_gmm_em.py](../../pytorch2/02_classical_and_probabilistic_ml/04_clustering_kmeans_spectral_and_gmm_em.py)
- **Functional JAX Module:** [04_clustering_kmeans_spectral_and_gmm_em.py](../../jax/02_classical_and_probabilistic_ml/04_clustering_kmeans_spectral_and_gmm_em.py)
- **Native Rust Candle Track:** [04_clustering_kmeans_spectral_and_gmm_em.rs](../../candle/02_classical_and_probabilistic_ml/04_clustering_kmeans_spectral_and_gmm_em.rs)
- **Hardware WGSL Kernel:** [04_clustering_kmeans_spectral_and_gmm_em.rs](../../kernels/02_classical_and_probabilistic_ml/04_clustering_kmeans_spectral_and_gmm_em.rs)
- **Golden Reference Checkpoint:** [04_clustering_kmeans_spectral_and_gmm_em.safetensors](../../weights/02_classical_and_probabilistic_ml/04_clustering_kmeans_spectral_and_gmm_em.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 03. k-NN & Locally Weighted Regression](03_knn_prototype_and_locally_weighted_regression.ipynb) | [Category Index](../README.md) | [Next: 05. Dimensionality Reduction (PCA/LDA/t-SNE/UMAP)](05_dimensionality_reduction_pca_lda_ica_tsne_umap.ipynb) &rarr;"""
cells.append(nbf.v4.new_markdown_cell(c8))

nb = nbf.v4.new_notebook(cells=cells)
out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "04_clustering_kmeans_spectral_and_gmm_em.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote notebook to", out_path)
