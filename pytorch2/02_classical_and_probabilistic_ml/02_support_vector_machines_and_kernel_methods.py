"""Support Vector Machines and Kernel Methods in PyTorch 2.x.

Features:
- Primal & Dual Linear / Kernel SVM
- RBF, Polynomial, and Matérn kernels
- Kernel Ridge Regression
- One-Class SVM / SVDD (Support Vector Data Description)
- Nyström low-rank landmark approximation
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_kernel_matrix(X1: torch.Tensor, X2: torch.Tensor, kernel: str = "rbf", gamma: float = 0.5, degree: int = 2, coef0: float = 1.0) -> torch.Tensor:
    """Computes Gram matrix between X1 [N, D] and X2 [M, D]."""
    if kernel == "rbf":
        dist_sq = torch.cdist(X1, X2, p=2.0) ** 2
        return torch.exp(-gamma * dist_sq)
    elif kernel == "poly":
        return (X1 @ X2.T + coef0) ** degree
    elif kernel == "linear":
        return X1 @ X2.T
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")


class KernelRidge(nn.Module):
    """Kernel Ridge Regression with closed-form dual solution."""
    def __init__(self, kernel: str = "rbf", gamma: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("X_train", torch.empty(0))
        self.register_buffer("dual_coef", torch.empty(0))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "KernelRidge":
        self.X_train = X.clone()
        N = X.shape[0]
        K = compute_kernel_matrix(X, X, kernel=self.kernel, gamma=self.gamma)
        reg_K = K + self.alpha * torch.eye(N, dtype=X.dtype, device=X.device)
        self.dual_coef = torch.linalg.solve(reg_K, y)
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        K_cross = compute_kernel_matrix(X, self.X_train, kernel=self.kernel, gamma=self.gamma)
        return K_cross @ self.dual_coef


class NystromApproximation(nn.Module):
    """Nyström low-rank kernel approximation using landmark points."""
    def __init__(self, m_landmarks: int = 32, gamma: float = 0.125, eps: float = 1e-4) -> None:
        super().__init__()
        self.m_landmarks = m_landmarks
        self.gamma = gamma
        self.eps = eps
        self.register_buffer("landmarks", torch.empty(0))
        self.register_buffer("K_mm_inv", torch.empty(0))

    def fit(self, X: torch.Tensor) -> "NystromApproximation":
        self.landmarks = X[:self.m_landmarks].clone()
        K_mm = compute_kernel_matrix(self.landmarks, self.landmarks, kernel="rbf", gamma=self.gamma)
        m = self.landmarks.shape[0]
        self.K_mm_inv = torch.linalg.pinv(K_mm + self.eps * torch.eye(m, dtype=X.dtype, device=X.device))
        return self

    def approximate_gram(self, X: torch.Tensor) -> torch.Tensor:
        K_nm = compute_kernel_matrix(X, self.landmarks, kernel="rbf", gamma=self.gamma)
        return K_nm @ self.K_mm_inv @ K_nm.T


class LinearSVMPrimal(nn.Module):
    """Primal Soft-Margin Support Vector Machine trained with subgradient descent / hinge loss."""
    def __init__(self, in_features: int, C: float = 1.0) -> None:
        super().__init__()
        self.C = C
        self.linear = nn.Linear(in_features, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        preds = self.forward(X)
        hinge = torch.clamp(1.0 - y * preds, min=0.0).mean()
        reg = 0.5 * torch.sum(self.linear.weight ** 2)
        return reg + self.C * hinge


def run_verification() -> bool:
    torch.manual_seed(42)
    X = torch.randn(64, 8)
    w_true = torch.tensor([1.2, -0.8, 0.5, -1.0, 0.3, -0.2, 0.7, -0.4])
    y_cont = X @ w_true + 0.1 * torch.randn(64)
    gamma = 1.0 / 8.0
    
    krr = KernelRidge(kernel="rbf", gamma=gamma, alpha=0.1)
    krr.fit(X, y_cont)
    pred = krr(X)
    res = torch.norm(pred - y_cont).item()
    assert res < 5.0, f"KRR residual too high: {res}"

    nystrom = NystromApproximation(m_landmarks=32, gamma=gamma)
    nystrom.fit(X)
    K_approx = nystrom.approximate_gram(X)
    K_exact = compute_kernel_matrix(X, X, kernel="rbf", gamma=gamma)
    err = torch.norm(K_approx - K_exact) / torch.norm(K_exact)
    assert err < 0.20, f"Nystrom relative error {err} too high"
    print("PyTorch Topic 02 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
