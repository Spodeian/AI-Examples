"""Generate golden weights and reference outputs for Topic 06:
06_naive_bayes_bayesian_regression_and_gaussian_processes
"""

from __future__ import annotations
from pathlib import Path
import torch
import safetensors.torch

def generate_weights() -> None:
    torch.manual_seed(42)
    N, D, C = 32, 4, 2

    # Synthetic observations X [32, 4]
    X = torch.randn(N, D)
    
    # Target values
    true_w = torch.tensor([1.5, -2.0, 0.8, -0.5], dtype=torch.float32)
    y_reg = X @ true_w + 0.1 * torch.randn(N) # [32]
    y_cls = (X[:, 0] + X[:, 1] > 0.0).long() # [32]

    # --- 1. Gaussian Naive Bayes ---
    X0 = X[y_cls == 0]
    X1 = X[y_cls == 1]
    means = torch.stack([X0.mean(dim=0), X1.mean(dim=0)], dim=0) # [2, 4]
    vars_ = torch.stack([X0.var(dim=0, unbiased=False) + 1e-4, X1.var(dim=0, unbiased=False) + 1e-4], dim=0) # [2, 4]
    priors = torch.tensor([len(X0) / N, len(X1) / N], dtype=torch.float32) # [2]
    log_priors = torch.log(priors) # [2]

    # Vectorized log posterior computation: log p(c) - 0.5 * sum(log(2*pi*var) + (x - mu)^2 / var)
    # Shape: [N, C, D]
    diff_sq = (X.unsqueeze(1) - means.unsqueeze(0)) ** 2
    log_gauss = -0.5 * (torch.log(2.0 * torch.pi * vars_).unsqueeze(0) + diff_sq / vars_.unsqueeze(0)) # [N, C, D]
    nb_log_joint = log_priors.unsqueeze(0) + log_gauss.sum(dim=-1) # [N, C]
    # Log-sum-exp normalization
    nb_log_probs = nb_log_joint - torch.logsumexp(nb_log_joint, dim=-1, keepdim=True) # [N, C]

    # --- 2. Bayesian Linear Regression ---
    alpha = 1.0 # prior precision
    beta = 2.0  # noise precision
    # S_N_inv = alpha * I + beta * X^T X
    S_N_inv = alpha * torch.eye(D) + beta * (X.T @ X)
    S_N = torch.linalg.inv(S_N_inv) # [D, D]
    m_N = beta * (S_N @ (X.T @ y_reg)) # [D]
    
    # In-sample predictive mean and variance: mu_* = X m_N, var_* = 1/beta + diag(X S_N X^T)
    bayes_pred_mean = X @ m_N # [32]
    bayes_pred_var = (1.0 / beta) + torch.sum((X @ S_N) * X, dim=-1) # [32]

    # --- 3. Exact Gaussian Process Regression (RBF Kernel) ---
    sigma_f = 1.0
    lengthscale = 1.0
    noise_var = 1e-4

    # Pairwise squared distances: [N, N]
    dist_sq = torch.cdist(X, X) ** 2
    K = (sigma_f ** 2) * torch.exp(-0.5 * dist_sq / (lengthscale ** 2))
    K_y = K + noise_var * torch.eye(N)
    L = torch.linalg.cholesky(K_y)
    # gp_alpha = K_y^{-1} y_reg via cholesky_solve
    gp_alpha = torch.cholesky_solve(y_reg.unsqueeze(1), L).squeeze(1) # [32]
    gp_pred_mean = K @ gp_alpha # [32]
    
    # Predictive variance: var_* = k(x, x) + noise_var - v^T v where L v = k_*
    # v = L^{-1} K. Since test points are X itself:
    v = torch.linalg.solve_triangular(L, K, upper=False) # [N, N]
    gp_pred_var = (sigma_f ** 2 + noise_var) - torch.sum(v ** 2, dim=0) # [32]
    gp_pred_var = torch.clamp(gp_pred_var, min=1e-8)

    # --- 4. Sparse Inducing Point GP (Nyström SVGP approximation) ---
    M = 8
    Z = X[:M].clone() # [8, 4] inducing landmarks
    dist_zz = torch.cdist(Z, Z) ** 2
    K_zz = (sigma_f ** 2) * torch.exp(-0.5 * dist_zz / (lengthscale ** 2)) + 1e-5 * torch.eye(M) # [8, 8]
    L_z = torch.linalg.cholesky(K_zz)
    dist_xz = torch.cdist(X, Z) ** 2
    K_xz = (sigma_f ** 2) * torch.exp(-0.5 * dist_xz / (lengthscale ** 2)) # [32, 8]
    # A = L_z^{-1} K_zx = L_z^{-1} K_xz^T => [8, 32]
    A = torch.linalg.solve_triangular(L_z, K_xz.T, upper=False)
    # diag(K_xz K_zz^{-1} K_zx) = col_norms_sq(A)
    svgp_approx_diag = torch.sum(A ** 2, dim=0) # [32]

    # --- 5. Linear baseline for cross-track parity contract ---
    weight = torch.tensor([0.4, -0.6, 0.8, -0.2], dtype=torch.float32)
    bias = torch.tensor([0.1], dtype=torch.float32)
    test_out = X @ weight + bias

    # Save dictionary
    state_dict = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": X.clone().contiguous(),
        "test_output": test_out.clone().contiguous(),
        "X": X.clone().contiguous(),
        "y_reg": y_reg.clone().contiguous(),
        "y_cls": y_cls.float().clone().contiguous(),
        "nb_means": means.clone().contiguous(),
        "nb_vars": vars_.clone().contiguous(),
        "nb_priors": priors.clone().contiguous(),
        "nb_log_probs": nb_log_probs.clone().contiguous(),
        "bayes_m_N": m_N.clone().contiguous(),
        "bayes_S_N": S_N.clone().contiguous(),
        "bayes_pred_mean": bayes_pred_mean.clone().contiguous(),
        "bayes_pred_var": bayes_pred_var.clone().contiguous(),
        "gp_K": K.clone().contiguous(),
        "gp_alpha": gp_alpha.clone().contiguous(),
        "gp_pred_mean": gp_pred_mean.clone().contiguous(),
        "gp_pred_var": gp_pred_var.clone().contiguous(),
        "Z": Z.clone().contiguous(),
        "K_zz": K_zz.clone().contiguous(),
        "K_xz": K_xz.clone().contiguous(),
        "svgp_approx_diag": svgp_approx_diag.clone().contiguous(),
    }

    out_dir = Path("weights/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "06_naive_bayes_bayesian_regression_and_gaussian_processes.safetensors"
    safetensors.torch.save_file(state_dict, out_file)
    print(f"Saved {len(state_dict)} tensors to {out_file}")

if __name__ == "__main__":
    generate_weights()
