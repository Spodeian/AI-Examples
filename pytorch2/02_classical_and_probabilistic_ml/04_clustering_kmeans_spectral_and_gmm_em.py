"""Clustering: k-Means, Spectral Clustering, and GMM Expectation-Maximization in PyTorch 2.x.

Features:
- Lloyd's k-Means with distance-based cluster assignment
- Log-domain Gaussian Mixture Model Expectation-Maximization (E-step & M-step) with diagonal variance jitter
- Symmetric Normalized Graph Laplacian for Spectral Clustering: L_sym = I - D^{-1/2} A D^{-1/2}
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KMeans(nn.Module):
    """Vectorized k-Means clustering via Lloyd's algorithm."""
    def __init__(self, n_clusters: int = 3, max_iter: int = 20) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.register_buffer("centers", torch.empty(0))

    def fit(self, X: torch.Tensor, init_centers: torch.Tensor | None = None) -> "KMeans":
        N, D = X.shape
        if init_centers is not None:
            centers = init_centers.clone()
        else:
            centers = X[:self.n_clusters].clone()

        for _ in range(self.max_iter):
            dist_sq = torch.cdist(X, centers) ** 2 # [N, K]
            assignments = torch.argmin(dist_sq, dim=1)
            new_centers = []
            for k in range(self.n_clusters):
                mask = (assignments == k)
                if mask.sum() > 0:
                    new_centers.append(X[mask].mean(dim=0))
                else:
                    new_centers.append(centers[k])
            centers = torch.stack(new_centers, dim=0)

        self.centers = centers
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(X, self.centers) ** 2
        return torch.argmin(dist_sq, dim=1)


class GaussianMixtureModel(nn.Module):
    """Diagonal Covariance Gaussian Mixture Model in Log-Domain."""
    def __init__(self, n_components: int = 3, max_iter: int = 10, jitter: float = 1e-4) -> None:
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.jitter = jitter
        self.register_buffer("mu", torch.empty(0))
        self.register_buffer("var", torch.empty(0))
        self.register_buffer("pi", torch.empty(0))

    def fit(self, X: torch.Tensor, init_mu: torch.Tensor | None = None) -> "GaussianMixtureModel":
        N, D = X.shape
        K = self.n_components

        pi = torch.full((K,), 1.0 / K, dtype=X.dtype, device=X.device)
        mu = init_mu.clone() if init_mu is not None else X[:K].clone()
        var = torch.ones(K, D, dtype=X.dtype, device=X.device)

        for _ in range(self.max_iter):
            # E-step in log space
            log_resp = torch.zeros(N, K, dtype=X.dtype, device=X.device)
            for k in range(K):
                diff = X - mu[k]
                quad = torch.sum(diff ** 2 / var[k], dim=1)
                log_det = torch.sum(torch.log(var[k]))
                log_prob = -0.5 * (D * 1.8378770664 + log_det + quad)
                log_resp[:, k] = torch.log(pi[k] + 1e-8) + log_prob

            log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
            resp = torch.exp(log_resp - log_norm)

            # M-step
            N_k = torch.sum(resp, dim=0) + 1e-6
            pi = N_k / N
            mu = (resp.T @ X) / N_k.unsqueeze(1)
            for k in range(K):
                diff = X - mu[k]
                var[k] = torch.sum(resp[:, k:k+1] * (diff ** 2), dim=0) / N_k[k]
                var[k] = torch.clamp(var[k], min=self.jitter)

        self.mu, self.var, self.pi = mu, var, pi
        return self


def compute_symmetric_laplacian(X: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Computes symmetric normalized graph Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}."""
    N = X.shape[0]
    A = torch.exp(-gamma * (torch.cdist(X, X) ** 2))
    A.fill_diagonal_(0.0)
    degrees = torch.sum(A, dim=1)
    d_inv_sqrt = torch.pow(torch.clamp(degrees, min=1e-6), -0.5)
    return torch.eye(N, dtype=X.dtype, device=X.device) - (d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0))


def run_verification() -> bool:
    torch.manual_seed(42)
    X = torch.cat([
        torch.randn(20, 8) + 3.0,
        torch.randn(20, 8) - 3.0
    ], dim=0)

    km = KMeans(n_clusters=2, max_iter=10).fit(X)
    labels = km.predict(X)
    assert len(torch.unique(labels)) == 2

    gmm = GaussianMixtureModel(n_components=2, max_iter=5).fit(X)
    assert gmm.mu.shape == (2, 8)

    L = compute_symmetric_laplacian(X, gamma=0.1)
    assert L.shape == (40, 40)
    assert torch.all(torch.isfinite(L))
    print("PyTorch Topic 04 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
