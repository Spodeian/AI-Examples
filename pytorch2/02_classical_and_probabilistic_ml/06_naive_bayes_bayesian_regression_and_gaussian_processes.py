"""Naive Bayes, Bayesian Linear Regression, and Gaussian Processes in PyTorch 2.x.

Features:
- Gaussian Naive Bayes with numerically stable log-domain log-sum-exp evaluation
- Conjugate Bayesian Linear Regression with closed-form posterior precision and predictive variance
- Exact Gaussian Process Regression with RBF covariance, Cholesky decomposition, and triangular solves
- Sparse Variational GP Inducing-Point / Nyström low-rank covariance approximation
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch


class GaussianNaiveBayes(nn.Module):
    """Gaussian Naive Bayes classifier evaluated in log-domain."""
    def __init__(self, n_classes: int = 2, var_smoothing: float = 1e-4) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.var_smoothing = var_smoothing
        self.register_buffer("means", torch.empty(0))
        self.register_buffer("vars", torch.empty(0))
        self.register_buffer("log_priors", torch.empty(0))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "GaussianNaiveBayes":
        N, D = X.shape
        means = []
        vars_ = []
        priors = []
        for c in range(self.n_classes):
            mask = (y == c)
            Xc = X[mask]
            means.append(Xc.mean(dim=0))
            vars_.append(Xc.var(dim=0, unbiased=False) + self.var_smoothing)
            priors.append(mask.float().mean())

        self.means = torch.stack(means, dim=0) # [C, D]
        self.vars = torch.stack(vars_, dim=0)   # [C, D]
        self.log_priors = torch.log(torch.stack(priors, dim=0)) # [C]
        return self

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Returns normalized log probabilities: log p(y=c | x)."""
        diff_sq = (X.unsqueeze(1) - self.means.unsqueeze(0)) ** 2 # [N, C, D]
        log_gauss = -0.5 * (
            torch.log(2.0 * torch.pi * self.vars).unsqueeze(0) + diff_sq / self.vars.unsqueeze(0)
        ) # [N, C, D]
        log_joint = self.log_priors.unsqueeze(0) + log_gauss.sum(dim=-1) # [N, C]
        return log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_log_proba(X), dim=-1)


class BayesianLinearRegression(nn.Module):
    """Conjugate Bayesian Linear Regression with Gaussian prior and likelihood."""
    def __init__(self, alpha: float = 1.0, beta: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("m_N", torch.empty(0))
        self.register_buffer("S_N", torch.empty(0))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "BayesianLinearRegression":
        D = X.shape[1]
        S_N_inv = self.alpha * torch.eye(D, device=X.device, dtype=X.dtype) + self.beta * (X.T @ X)
        S_N = torch.linalg.inv(S_N_inv)
        m_N = self.beta * (S_N @ (X.T @ y))
        self.S_N = S_N
        self.m_N = m_N
        return self

    def predict(self, X_star: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns predictive mean and variance for test points."""
        mean = X_star @ self.m_N
        variance = (1.0 / self.beta) + torch.sum((X_star @ self.S_N) * X_star, dim=-1)
        return mean, variance


class GaussianProcessRegression(nn.Module):
    """Exact Gaussian Process Regression with Squared Exponential (RBF) Kernel."""
    def __init__(self, lengthscale: float = 1.0, sigma_f: float = 1.0, noise_var: float = 1e-4) -> None:
        super().__init__()
        self.lengthscale = lengthscale
        self.sigma_f = sigma_f
        self.noise_var = noise_var
        self.register_buffer("X_train", torch.empty(0))
        self.register_buffer("alpha", torch.empty(0))
        self.register_buffer("L", torch.empty(0))

    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(X1, X2) ** 2
        return (self.sigma_f ** 2) * torch.exp(-0.5 * dist_sq / (self.lengthscale ** 2))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "GaussianProcessRegression":
        N = X.shape[0]
        K = self._kernel(X, X)
        K_y = K + self.noise_var * torch.eye(N, device=X.device, dtype=X.dtype)
        L = torch.linalg.cholesky(K_y)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
        self.X_train = X
        self.L = L
        self.alpha = alpha
        return self

    def predict(self, X_star: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        K_star = self._kernel(X_star, self.X_train)
        mean = K_star @ self.alpha
        v = torch.linalg.solve_triangular(self.L, K_star.T, upper=False)
        var = (self.sigma_f ** 2 + self.noise_var) - torch.sum(v ** 2, dim=0)
        return mean, torch.clamp(var, min=1e-8)


def compute_svgp_inducing_diag(
    X: torch.Tensor,
    Z: torch.Tensor,
    lengthscale: float = 1.0,
    sigma_f: float = 1.0,
    jitter: float = 1e-5
) -> torch.Tensor:
    """Computes diagonal elements of inducing-point covariance Q_XX = K_XZ K_ZZ^{-1} K_ZX."""
    M = Z.shape[0]
    dist_zz = torch.cdist(Z, Z) ** 2
    K_zz = (sigma_f ** 2) * torch.exp(-0.5 * dist_zz / (lengthscale ** 2)) + jitter * torch.eye(M, device=Z.device, dtype=Z.dtype)
    L_z = torch.linalg.cholesky(K_zz)
    dist_xz = torch.cdist(X, Z) ** 2
    K_xz = (sigma_f ** 2) * torch.exp(-0.5 * dist_xz / (lengthscale ** 2))
    A = torch.linalg.solve_triangular(L_z, K_xz.T, upper=False)
    return torch.sum(A ** 2, dim=0)


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    X = chk["X"]
    y_reg = chk["y_reg"]
    y_cls = chk["y_cls"].long()

    # 1. Naive Bayes
    nb = GaussianNaiveBayes(n_classes=2, var_smoothing=1e-4).fit(X, y_cls)
    log_probs = nb.predict_log_proba(X)
    diff_nb = torch.max(torch.abs(log_probs - chk["nb_log_probs"])).item()
    assert diff_nb < 1e-5, f"Naive Bayes log probs mismatch: {diff_nb:.2e}"

    # 2. Bayesian Linear Regression
    bayes = BayesianLinearRegression(alpha=1.0, beta=2.0).fit(X, y_reg)
    b_mean, b_var = bayes.predict(X)
    diff_bm = torch.max(torch.abs(b_mean - chk["bayes_pred_mean"])).item()
    diff_bv = torch.max(torch.abs(b_var - chk["bayes_pred_var"])).item()
    assert diff_bm < 1e-5, f"Bayesian regression mean mismatch: {diff_bm:.2e}"
    assert diff_bv < 1e-5, f"Bayesian regression var mismatch: {diff_bv:.2e}"

    # 3. Gaussian Process Regression
    gp = GaussianProcessRegression(lengthscale=1.0, sigma_f=1.0, noise_var=1e-4).fit(X, y_reg)
    gp_mean, gp_var = gp.predict(X)
    diff_gpm = torch.max(torch.abs(gp_mean - chk["gp_pred_mean"])).item()
    diff_gpv = torch.max(torch.abs(gp_var - chk["gp_pred_var"])).item()
    assert diff_gpm < 1e-5, f"GP mean mismatch: {diff_gpm:.2e}"
    assert diff_gpv < 1e-5, f"GP var mismatch: {diff_gpv:.2e}"

    # 4. SVGP Inducing Points
    Z = chk["Z"]
    svgp_diag = compute_svgp_inducing_diag(X, Z, lengthscale=1.0, sigma_f=1.0, jitter=1e-5)
    diff_svgp = torch.max(torch.abs(svgp_diag - chk["svgp_approx_diag"])).item()
    assert diff_svgp < 1e-5, f"SVGP diag mismatch: {diff_svgp:.2e}"

    print("PyTorch Topic 06 verification passed!")
    return True


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
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

    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    run_verification()
