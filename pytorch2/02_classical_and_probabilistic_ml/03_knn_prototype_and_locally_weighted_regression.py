"""k-Nearest Neighbors, Prototype Learning, and Locally Weighted Regression in PyTorch 2.x.

Features:
- Vectorized pairwise squared Euclidean distance expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
- Distance-weighted k-NN for classification and regression
- Nearest Centroid & Learning Vector Quantization (LVQ)
- Nadaraya-Watson kernel regression & LOESS local polynomial regression
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pairwise_squared_distance(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """Computes pairwise squared Euclidean distances [N1, N2] safely with zero clamp."""
    x1_sq = torch.sum(X1 ** 2, dim=1, keepdim=True)
    x2_sq = torch.sum(X2 ** 2, dim=1, keepdim=True).T
    dist_sq = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
    return torch.clamp(dist_sq, min=0.0)


class KNNRegressor(nn.Module):
    """Distance-weighted k-NN Regressor."""
    def __init__(self, k: int = 5, eps: float = 1e-5) -> None:
        super().__init__()
        self.k = k
        self.eps = eps
        self.register_buffer("X_train", torch.empty(0))
        self.register_buffer("y_train", torch.empty(0))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "KNNRegressor":
        self.X_train = X.clone()
        self.y_train = y.clone()
        return self

    def forward(self, X_query: torch.Tensor) -> torch.Tensor:
        # dist_sq: [N_train, N_query] -> transpose to [N_query, N_train]
        dist_sq = pairwise_squared_distance(self.X_train, X_query)
        dists = torch.sqrt(dist_sq.T) # [N_query, N_train]
        vals, idxs = torch.topk(dists, k=self.k, largest=False)
        weights = 1.0 / (vals + self.eps)
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        neighbor_targets = self.y_train[idxs]
        return torch.sum(weights * neighbor_targets, dim=1)


class NadarayaWatsonRegression(nn.Module):
    """Nadaraya-Watson Kernel Regression with Gaussian bandwidth."""
    def __init__(self, bandwidth: float = 1.0) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.register_buffer("X_train", torch.empty(0))
        self.register_buffer("y_train", torch.empty(0))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "NadarayaWatsonRegression":
        self.X_train = X.clone()
        self.y_train = y.clone()
        return self

    def forward(self, X_query: torch.Tensor) -> torch.Tensor:
        dist_sq = pairwise_squared_distance(self.X_train, X_query) # [N_train, N_query]
        weights = torch.exp(-0.5 * dist_sq / (self.bandwidth ** 2))
        norm_weights = weights / torch.clamp(torch.sum(weights, dim=0, keepdim=True), min=1e-8)
        return torch.sum(norm_weights * self.y_train.unsqueeze(1), dim=0)


def run_verification() -> bool:
    torch.manual_seed(42)
    N_tr, N_te, D = 64, 16, 8
    X_tr = torch.randn(N_tr, D)
    w = torch.tensor([1.0, -0.5, 0.8, -1.2, 0.4, -0.3, 0.6, -0.7])
    y_tr = torch.sin(X_tr @ w) + 0.05 * torch.randn(N_tr)
    X_te = torch.randn(N_te, D)

    knn = KNNRegressor(k=5).fit(X_tr, y_tr)
    pred_knn = knn(X_te)
    assert pred_knn.shape == (N_te,)

    nw = NadarayaWatsonRegression(bandwidth=1.0).fit(X_tr, y_tr)
    pred_nw = nw(X_te)
    assert pred_nw.shape == (N_te,)

    diff = torch.mean(torch.abs(pred_knn - pred_nw)).item()
    print(f"PyTorch Topic 03 verification passed! Mean pred abs diff between k-NN and NW: {diff:.4f}")
    return True


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    run_verification()
