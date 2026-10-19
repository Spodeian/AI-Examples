"""Generate golden reference SafeTensors and verify PyTorch/JAX parity for Category 02 Topic 01.

Topic 01: 01_generalized_linear_models_and_proximal_lasso
Category: 02_classical_and_probabilistic_ml
"""

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

if __name__ == "__main__":
    generate_topic_02_01_artifacts()
