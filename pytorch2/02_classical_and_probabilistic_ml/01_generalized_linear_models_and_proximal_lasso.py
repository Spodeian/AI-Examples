"""PyTorch 2.x implementation for Category 02 Topic 01:
Generalized Linear Models and Proximal Lasso.

Provides mathematically rigorous, production-grade implementations of:
1. Ordinary Least Squares (OLS) via QR decomposition.
2. Ridge Regression with L2 regularization.
3. Lasso Regression via Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
4. Poisson GLM with log link and negative log-likelihood gradient step.
5. Robust Huber loss regression.
6. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import math
import torch
import torch.nn as nn
from safetensors.torch import load_file


def solve_ols_qr(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solves Ordinary Least Squares using QR decomposition without explicit matrix inversion."""
    q, r = torch.linalg.qr(x)
    w = torch.linalg.solve_triangular(r, torch.matmul(q.T, y).unsqueeze(-1), upper=True).squeeze(-1)
    mse = torch.mean((torch.matmul(x, w) - y) ** 2)
    return w, mse


def solve_ridge(x: torch.Tensor, y: torch.Tensor, lambda_reg: float = 0.5) -> torch.Tensor:
    """Solves Ridge regression closed form: (X^T X + lambda I)^{-1} X^T y."""
    xtx = torch.matmul(x.T, x)
    reg_mat = xtx + lambda_reg * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(reg_mat, torch.matmul(x.T, y))


def solve_lasso_fista(
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_reg: float = 0.2,
    steps: int = 10,
) -> torch.Tensor:
    """Solves Lasso regression via FISTA (Beck & Teboulle, 2009)."""
    xtx = torch.matmul(x.T, x)
    eig_max = float(torch.linalg.eigvalsh(xtx)[-1].item())
    step_size = 1.0 / (eig_max + 1e-4)

    w = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    y_k = w.clone()
    t_k = 1.0

    for _ in range(steps):
        grad = torch.matmul(x.T, torch.matmul(x, y_k) - y)
        z = y_k - step_size * grad
        threshold = step_size * lambda_reg
        w_next = torch.sign(z) * torch.clamp(torch.abs(z) - threshold, min=0.0)

        t_next = (1.0 + math.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
        y_k = w_next + ((t_k - 1.0) / t_next) * (w_next - w)
        w = w_next
        t_k = t_next

    return w


def step_poisson_glm(
    x: torch.Tensor,
    y_counts: torch.Tensor,
    w_init: torch.Tensor,
    lr: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Poisson GLM gradient and performs 1 gradient step."""
    mu = torch.exp(torch.matmul(x, w_init))
    grad = torch.matmul(x.T, mu - y_counts)
    w_next = w_init - lr * grad
    return grad, w_next


def compute_huber_mean_loss(residual: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    """Computes mean Huber loss."""
    abs_r = torch.abs(residual)
    quad = torch.clamp(abs_r, max=delta)
    lin = abs_r - quad
    loss = 0.5 * (quad ** 2) + delta * lin
    return torch.mean(loss)


class LinearAnchor(nn.Module):
    """Linear anchor module for cross-track parity verification."""

    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class GLMUnifiedModule(nn.Module):
    """Unified module for Category 02 Topic 01."""

    def __init__(self, weights_path: Path):
        super().__init__()
        weights = load_file(str(weights_path))

        self.x_data = weights["x_data"]
        self.y_data = weights["y_data"]
        self.golden_w_ols = weights["golden_w_ols"]
        self.golden_mse_ols = weights["golden_mse_ols"]
        self.golden_w_ridge = weights["golden_w_ridge"]
        self.golden_w_lasso = weights["golden_w_lasso"]

        self.w_poisson_init = weights["w_poisson_init"]
        self.y_counts = weights["y_counts"]
        self.golden_poisson_grad = weights["golden_poisson_grad"]
        self.golden_w_poisson_next = weights["golden_w_poisson_next"]
        self.golden_mean_huber = weights["golden_mean_huber"]

        self.anchor = LinearAnchor(4, 2)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, tolerance: float = 1e-5) -> dict[str, float]:
        """Verifies parity against golden reference tensors."""
        w_ols, mse_ols = solve_ols_qr(self.x_data, self.y_data)
        ols_err = max(
            float(torch.max(torch.abs(w_ols - self.golden_w_ols)).item()),
            float(torch.abs(mse_ols - self.golden_mse_ols).item()),
        )

        w_ridge = solve_ridge(self.x_data, self.y_data, lambda_reg=0.5)
        ridge_err = float(torch.max(torch.abs(w_ridge - self.golden_w_ridge)).item())

        w_lasso = solve_lasso_fista(self.x_data, self.y_data, lambda_reg=0.2, steps=10)
        lasso_err = float(torch.max(torch.abs(w_lasso - self.golden_w_lasso)).item())

        p_grad, w_p_next = step_poisson_glm(self.x_data, self.y_counts, self.w_poisson_init, lr=0.01)
        poisson_err = max(
            float(torch.max(torch.abs(p_grad - self.golden_poisson_grad)).item()),
            float(torch.max(torch.abs(w_p_next - self.golden_w_poisson_next)).item()),
        )

        res = torch.matmul(self.x_data, w_ols) - self.y_data
        huber = compute_huber_mean_loss(res, delta=0.5)
        huber_err = float(torch.abs(huber - self.golden_mean_huber).item())

        anchor_out = self.anchor(self.anchor_x)
        anchor_err = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())

        assert ols_err < tolerance, f"OLS error {ols_err} exceeds {tolerance}"
        assert ridge_err < tolerance, f"Ridge error {ridge_err} exceeds {tolerance}"
        assert lasso_err < tolerance, f"Lasso error {lasso_err} exceeds {tolerance}"
        assert poisson_err < tolerance, f"Poisson error {poisson_err} exceeds {tolerance}"
        assert huber_err < tolerance, f"Huber error {huber_err} exceeds {tolerance}"
        assert anchor_err < tolerance, f"Anchor error {anchor_err} exceeds {tolerance}"

        return {
            "ols_err": ols_err,
            "ridge_err": ridge_err,
            "lasso_err": lasso_err,
            "poisson_err": poisson_err,
            "huber_err": huber_err,
            "anchor_err": anchor_err,
        }


if __name__ == "__main__":
    ckpt = Path("weights/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.safetensors")
    module = GLMUnifiedModule(ckpt)
    errors = module.verify_invariants()
    print("PyTorch Category 02 Topic 01 Invariants Verified:", errors)
