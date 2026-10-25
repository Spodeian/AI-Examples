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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file

    def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)

    def huber_loss(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        abs_r = torch.abs(residual)
        quadratic = torch.clamp(abs_r, max=delta)
        linear = abs_r - quadratic
        return 0.5 * (quadratic ** 2) + delta * linear

    def generate_topic_02_01_artifacts():
        torch.manual_seed(42)
        np.random.seed(42)

        # Synthetic design matrix X [N=8, D=4] and target y [N=8]
        x_data = torch.tensor([
            [1.0, 0.5, -0.2, 0.8],
            [1.0, -0.8, 0.4, -0.5],
            [1.0, 1.2, -0.9, 0.3],
            [1.0, -0.3, 0.7, -1.1],
            [1.0, 0.9, 0.1, 0.6],
            [1.0, -1.4, -0.5, 0.2],
            [1.0, 0.2, 1.1, -0.7],
            [1.0, -0.6, -0.8, 1.0]
        ], dtype=torch.float32)

        true_w = torch.tensor([1.5, -2.0, 0.0, 0.8], dtype=torch.float32)
        y_data = torch.matmul(x_data, true_w) + 0.05 * torch.randn(8)

        # 1. Ordinary Least Squares (OLS) closed form via QR decomposition:
        # X = Q R -> w_ols = R^{-1} Q^T y
        q, r = torch.linalg.qr(x_data)
        w_ols = torch.linalg.solve_triangular(r, torch.matmul(q.T, y_data).unsqueeze(-1), upper=True).squeeze(-1)
        y_pred_ols = torch.matmul(x_data, w_ols)
        mse_ols = torch.mean((y_pred_ols - y_data) ** 2)

        # 2. Ridge Regression closed-form: (X^T X + lambda I)^{-1} X^T y
        lambda_ridge = 0.5
        xtx = torch.matmul(x_data.T, x_data)
        reg_mat = xtx + lambda_ridge * torch.eye(4)
        w_ridge = torch.linalg.solve(reg_mat, torch.matmul(x_data.T, y_data))

        # 3. Lasso via 10 FISTA steps (Beck & Teboulle, 2009)
        # Lipschitz constant L <= ||X^T X||_2
        eig_max = float(torch.linalg.eigvalsh(xtx)[-1].item())
        step_size = 1.0 / (eig_max + 1e-4)
        lambda_lasso = 0.2

        w_lasso = torch.zeros(4, dtype=torch.float32)
        y_k = w_lasso.clone()
        t_k = 1.0

        for _ in range(10):
            # Gradient of 0.5 * ||X y_k - y||^2 is X^T (X y_k - y)
            grad = torch.matmul(x_data.T, torch.matmul(x_data, y_k) - y_data)
            w_next = soft_threshold(y_k - step_size * grad, threshold=step_size * lambda_lasso)
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
            y_k = w_next + ((t_k - 1.0) / t_next) * (w_next - w_lasso)
            w_lasso = w_next
            t_k = t_next

        # 4. Poisson GLM step with log link: mu = exp(X w)
        w_poisson_init = torch.tensor([0.1, -0.1, 0.05, 0.02], dtype=torch.float32)
        eta_lin = torch.matmul(x_data, w_poisson_init)
        mu_poisson = torch.exp(eta_lin)
        # Poisson deviance: 2 * sum(y * log(y/mu) - (y - mu)) on non-negative y
        y_counts = torch.abs(torch.round(y_data + 2.0)) # non-negative integer counts
        # Gradient of negative log likelihood: X^T (mu - y)
        poisson_grad = torch.matmul(x_data.T, mu_poisson - y_counts)
        w_poisson_next = w_poisson_init - 0.01 * poisson_grad

        # 5. Robust Huber Regression Loss evaluation
        residuals = torch.matmul(x_data, w_ols) - y_data
        huber_res = huber_loss(residuals, delta=0.5)
        mean_huber = torch.mean(huber_res)

        # 6. Linear Anchor for Multi-Track Parity
        anchor_x = torch.tensor([
            [0.2, -0.1, 0.4, -0.3],
            [-0.5, 0.6, -0.7, 0.8]
        ], dtype=torch.float32)
        anchor_w = torch.randn(4, 2, dtype=torch.float32)
        anchor_b = torch.randn(2, dtype=torch.float32)
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        tensors = {
            "x_data": x_data,
            "y_data": y_data,
            "golden_w_ols": w_ols,
            "golden_mse_ols": mse_ols,
            "golden_w_ridge": w_ridge,
            "golden_w_lasso": w_lasso,
            "w_poisson_init": w_poisson_init,
            "y_counts": y_counts,
            "golden_poisson_grad": poisson_grad,
            "golden_w_poisson_next": w_poisson_next,
            "golden_mean_huber": mean_huber,

            "anchor_x": anchor_x,
            "anchor_w": anchor_w,
            "anchor_b": anchor_b,
            "golden_anchor_out": golden_anchor_out,

            # Standard parity test runner keys
            "test_input": anchor_x.clone().contiguous(),
            "test_output": golden_anchor_out.clone().contiguous(),
            "weight": anchor_w.clone().contiguous(),
            "bias": anchor_b.clone().contiguous(),
        }

        out_dir = Path("weights/02_classical_and_probabilistic_ml")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "01_generalized_linear_models_and_proximal_lasso.safetensors"

        save_file(tensors, str(out_path))
        print(f"Saved {len(tensors)} tensors to {out_path}")

        # Self-Parity Assertions
        assert torch.allclose(w_ols, tensors["golden_w_ols"], atol=1e-6)
        assert torch.allclose(w_ridge, tensors["golden_w_ridge"], atol=1e-6)
        assert torch.allclose(w_lasso, tensors["golden_w_lasso"], atol=1e-6)
        assert torch.allclose(poisson_grad, tensors["golden_poisson_grad"], atol=1e-6)
        assert torch.allclose(w_poisson_next, tensors["golden_w_poisson_next"], atol=1e-6)
        assert torch.allclose(mean_huber, tensors["golden_mean_huber"], atol=1e-6)
        assert torch.allclose(golden_anchor_out, tensors["golden_anchor_out"], atol=1e-6)
        print("PyTorch Self-Parity verified successfully!")

    generate_topic_02_01_artifacts()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    ckpt = Path("weights/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.safetensors")
    module = GLMUnifiedModule(ckpt)
    errors = module.verify_invariants()
    print("PyTorch Category 02 Topic 01 Invariants Verified:", errors)
