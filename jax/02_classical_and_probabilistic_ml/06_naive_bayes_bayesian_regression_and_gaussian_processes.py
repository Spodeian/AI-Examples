"""Naive Bayes, Bayesian Linear Regression, and Gaussian Processes in JAX.

Features:
- Functional Gaussian Naive Bayes with log-domain logsumexp normalization
- Conjugate Bayesian Linear Regression with analytical posterior mean and variance
- Exact Gaussian Process Regression via JAX cholesky and triangular solves
- Sparse Variational GP Inducing-Point / Nyström low-rank variance approximation
"""

from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
from functools import partial
import safetensors.torch
import numpy as np


@jax.jit
def gaussian_naive_bayes_predict_log_proba(
    X: jnp.ndarray,
    means: jnp.ndarray,
    vars_: jnp.ndarray,
    log_priors: jnp.ndarray
) -> jnp.ndarray:
    """Computes normalized log probabilities: log p(y=c | x) for each class c."""
    # X: [N, D], means: [C, D], vars_: [C, D], log_priors: [C]
    diff_sq = (jnp.expand_dims(X, 1) - jnp.expand_dims(means, 0)) ** 2 # [N, C, D]
    log_gauss = -0.5 * (
        jnp.expand_dims(jnp.log(2.0 * jnp.pi * vars_), 0) + diff_sq / jnp.expand_dims(vars_, 0)
    ) # [N, C, D]
    log_joint = jnp.expand_dims(log_priors, 0) + jnp.sum(log_gauss, axis=-1) # [N, C]
    return log_joint - jax.nn.logsumexp(log_joint, axis=-1, keepdims=True)


@jax.jit
def bayesian_linear_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    X_star: jnp.ndarray,
    alpha: float = 1.0,
    beta: float = 2.0
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes Bayesian linear regression posterior (m_N, S_N) and predictive (mean, var)."""
    D = X.shape[1]
    S_N_inv = alpha * jnp.eye(D) + beta * (X.T @ X)
    S_N = jnp.linalg.inv(S_N_inv)
    m_N = beta * (S_N @ (X.T @ y))
    pred_mean = X_star @ m_N
    pred_var = (1.0 / beta) + jnp.sum((X_star @ S_N) * X_star, axis=-1)
    return m_N, S_N, pred_mean, pred_var


@jax.jit
def rbf_kernel(
    X1: jnp.ndarray,
    X2: jnp.ndarray,
    lengthscale: float = 1.0,
    sigma_f: float = 1.0
) -> jnp.ndarray:
    """Computes RBF / Squared Exponential covariance between X1 and X2."""
    diff = jnp.expand_dims(X1, 1) - jnp.expand_dims(X2, 0) # [N1, N2, D]
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    return (sigma_f ** 2) * jnp.exp(-0.5 * dist_sq / (lengthscale ** 2))


@jax.jit
def gaussian_process_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    X_star: jnp.ndarray,
    lengthscale: float = 1.0,
    sigma_f: float = 1.0,
    noise_var: float = 1e-4
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Performs exact GP inference and prediction using Cholesky factorization."""
    N = X.shape[0]
    K = rbf_kernel(X, X, lengthscale, sigma_f)
    K_y = K + noise_var * jnp.eye(N)
    L = jnp.linalg.cholesky(K_y)
    
    # alpha = K_y^{-1} y via two triangular solves: L w = y, L^T alpha = w
    w = jax.scipy.linalg.solve_triangular(L, y, lower=True)
    alpha = jax.scipy.linalg.solve_triangular(L.T, w, lower=False)
    
    K_star = rbf_kernel(X_star, X, lengthscale, sigma_f)
    pred_mean = K_star @ alpha
    
    # v = L^{-1} K_star^T
    v = jax.scipy.linalg.solve_triangular(L, K_star.T, lower=True)
    pred_var = (sigma_f ** 2 + noise_var) - jnp.sum(v ** 2, axis=0)
    pred_var = jnp.clip(pred_var, a_min=1e-8)
    return alpha, pred_mean, pred_var


@jax.jit
def compute_svgp_inducing_diag(
    X: jnp.ndarray,
    Z: jnp.ndarray,
    lengthscale: float = 1.0,
    sigma_f: float = 1.0,
    jitter: float = 1e-5
) -> jnp.ndarray:
    """Computes diagonal of inducing point approximation Q_XX = K_XZ K_ZZ^{-1} K_ZX."""
    M = Z.shape[0]
    K_zz = rbf_kernel(Z, Z, lengthscale, sigma_f) + jitter * jnp.eye(M)
    L_z = jnp.linalg.cholesky(K_zz)
    K_xz = rbf_kernel(X, Z, lengthscale, sigma_f)
    A = jax.scipy.linalg.solve_triangular(L_z, K_xz.T, lower=True)
    return jnp.sum(A ** 2, axis=0)


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    X = jnp.array(chk["X"].numpy())
    y_reg = jnp.array(chk["y_reg"].numpy())
    means = jnp.array(chk["nb_means"].numpy())
    vars_ = jnp.array(chk["nb_vars"].numpy())
    log_priors = jnp.log(jnp.array(chk["nb_priors"].numpy()))

    # 1. Naive Bayes Log Probs
    log_probs = gaussian_naive_bayes_predict_log_proba(X, means, vars_, log_priors)
    ref_log_probs = chk["nb_log_probs"].numpy()
    diff_nb = float(np.max(np.abs(np.array(log_probs) - ref_log_probs)))
    assert diff_nb < 1e-5, f"JAX Naive Bayes log probs mismatch: {diff_nb:.2e}"

    # 2. Bayesian Linear Regression
    m_N, S_N, pred_mean, pred_var = bayesian_linear_regression(X, y_reg, X, alpha=1.0, beta=2.0)
    diff_bm = float(np.max(np.abs(np.array(pred_mean) - chk["bayes_pred_mean"].numpy())))
    diff_bv = float(np.max(np.abs(np.array(pred_var) - chk["bayes_pred_var"].numpy())))
    assert diff_bm < 1e-5, f"JAX Bayes mean mismatch: {diff_bm:.2e}"
    assert diff_bv < 1e-5, f"JAX Bayes var mismatch: {diff_bv:.2e}"

    # 3. Gaussian Process Regression
    alpha, gp_mean, gp_var = gaussian_process_regression(X, y_reg, X, lengthscale=1.0, sigma_f=1.0, noise_var=1e-4)
    diff_gpm = float(np.max(np.abs(np.array(gp_mean) - chk["gp_pred_mean"].numpy())))
    diff_gpv = float(np.max(np.abs(np.array(gp_var) - chk["gp_pred_var"].numpy())))
    assert diff_gpm < 1e-5, f"JAX GP mean mismatch: {diff_gpm:.2e}"
    assert diff_gpv < 1e-5, f"JAX GP var mismatch: {diff_gpv:.2e}"

    # 4. SVGP Inducing Points
    Z = jnp.array(chk["Z"].numpy())
    svgp_diag = compute_svgp_inducing_diag(X, Z, lengthscale=1.0, sigma_f=1.0, jitter=1e-5)
    diff_svgp = float(np.max(np.abs(np.array(svgp_diag) - chk["svgp_approx_diag"].numpy())))
    assert diff_svgp < 1e-5, f"JAX SVGP diag mismatch: {diff_svgp:.2e}"

    print("JAX Topic 06 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
