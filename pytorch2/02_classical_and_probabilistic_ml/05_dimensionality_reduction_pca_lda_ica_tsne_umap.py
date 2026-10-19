"""Dimensionality Reduction: PCA, LDA, ICA, t-SNE, and UMAP in PyTorch 2.x.

Features:
- Principal Component Analysis (PCA) via SVD on centered data
- Fisher's Linear Discriminant Analysis (LDA) for maximal class separation
- Kernel PCA with double-centering of the Gram matrix
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PrincipalComponentAnalysis(nn.Module):
    """PCA via SVD decomposition on mean-centered data."""
    def __init__(self, n_components: int = 2) -> None:
        super().__init__()
        self.n_components = n_components
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0)) # [D, K]
        self.register_buffer("explained_variance", torch.empty(0))

    def fit(self, X: torch.Tensor) -> "PrincipalComponentAnalysis":
        N, D = X.shape
        self.mean = torch.mean(X, dim=0, keepdim=True)
        X_centered = X - self.mean
        _, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components, :].T.contiguous()
        self.explained_variance = (S[:self.n_components] ** 2) / (N - 1)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) @ self.components


class FisherLinearDiscriminantAnalysis(nn.Module):
    """Binary Fisher's Linear Discriminant Analysis."""
    def __init__(self, reg: float = 1e-3) -> None:
        super().__init__()
        self.reg = reg
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("w", torch.empty(0))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "FisherLinearDiscriminantAnalysis":
        D = X.shape[1]
        self.mean = torch.mean(X, dim=0, keepdim=True)
        X_centered = X - self.mean
        X0 = X_centered[y == 0]
        X1 = X_centered[y == 1]
        mu0 = X0.mean(dim=0)
        mu1 = X1.mean(dim=0)
        Sw = (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1) + self.reg * torch.eye(D, dtype=X.dtype, device=X.device)
        w = torch.linalg.solve(Sw, (mu1 - mu0).unsqueeze(1)).squeeze(1)
        self.w = (w / torch.norm(w)).contiguous()
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) @ self.w


class KernelPCA(nn.Module):
    """Kernel PCA with double-centering of the RBF Gram matrix."""
    def __init__(self, n_components: int = 2, gamma: float = 0.1) -> None:
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.register_buffer("X_train", torch.empty(0))
        self.register_buffer("alphas", torch.empty(0))

    def fit(self, X: torch.Tensor) -> "KernelPCA":
        self.X_train = X.clone()
        N = X.shape[0]
        dist_sq = torch.cdist(X, X) ** 2
        K = torch.exp(-self.gamma * dist_sq)
        one_n = torch.full((N, N), 1.0 / N, dtype=X.dtype, device=X.device)
        K_tilde = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        eigvals, eigvecs = torch.linalg.eigh(K_tilde)
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)
        self.alphas = (eigvecs[:, :self.n_components] / torch.sqrt(torch.clamp(eigvals[:self.n_components], min=1e-6))).contiguous()
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        N = self.X_train.shape[0]
        dist_sq = torch.cdist(X, self.X_train) ** 2
        K = torch.exp(-self.gamma * dist_sq)
        one_n = torch.full((X.shape[0], N), 1.0 / N, dtype=X.dtype, device=X.device)
        one_n_train = torch.full((N, N), 1.0 / N, dtype=X.dtype, device=X.device)
        K_tilde = K - one_n @ K - K @ one_n_train + one_n @ K @ one_n_train
        return K_tilde @ self.alphas


def run_verification() -> bool:
    torch.manual_seed(42)
    X = torch.randn(64, 8)
    y = (X[:, 0] > 0.0).long()

    pca = PrincipalComponentAnalysis(n_components=3).fit(X)
    proj_pca = pca.transform(X)
    assert proj_pca.shape == (64, 3)

    lda = FisherLinearDiscriminantAnalysis().fit(X, y)
    proj_lda = lda.transform(X)
    assert proj_lda.shape == (64,)

    kpca = KernelPCA(n_components=3, gamma=0.1).fit(X)
    proj_kpca = kpca.transform(X)
    assert proj_kpca.shape == (64, 3)
    print("PyTorch Topic 05 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
