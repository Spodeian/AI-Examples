"""Support Vector Machines and Kernel Methods in JAX.

Functional implementations:
- RBF, Polynomial, and Linear Kernel Gram Matrix
- Kernel Ridge Regression closed-form dual solver
- Nyström low-rank landmark approximation
- Primal Soft-Margin SVM Hinge Loss
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_kernel_matrix(X1: jnp.ndarray, X2: jnp.ndarray, kernel: str = "rbf", gamma: float = 0.125, degree: int = 2, coef0: float = 1.0) -> jnp.ndarray:
    """Computes Gram matrix between X1 [N, D] and X2 [M, D]."""
    if kernel == "rbf":
        # Pairwise squared Euclidean distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
        x1_sq = jnp.sum(X1 ** 2, axis=1, keepdims=True)
        x2_sq = jnp.sum(X2 ** 2, axis=1, keepdims=True).T
        dist_sq = jnp.maximum(x1_sq + x2_sq - 2.0 * jnp.dot(X1, X2.T), 0.0)
        return jnp.exp(-gamma * dist_sq)
    elif kernel == "poly":
        return (jnp.dot(X1, X2.T) / X1.shape[1] + coef0) ** degree
    elif kernel == "linear":
        return jnp.dot(X1, X2.T)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")


@jax.jit
def solve_kernel_ridge(X: jnp.ndarray, y: jnp.ndarray, gamma: float = 0.125, alpha: float = 0.1) -> jnp.ndarray:
    """Solves (K + alpha * I) c = y for dual coefficients c."""
    N = X.shape[0]
    K = compute_kernel_matrix(X, X, kernel="rbf", gamma=gamma)
    reg_K = K + alpha * jnp.eye(N)
    return jnp.linalg.solve(reg_K, y)


@jax.jit
def predict_kernel_ridge(X_test: jnp.ndarray, X_train: jnp.ndarray, dual_coef: jnp.ndarray, gamma: float = 0.125) -> jnp.ndarray:
    """Evaluates K(X_test, X_train) @ dual_coef."""
    K_cross = compute_kernel_matrix(X_test, X_train, kernel="rbf", gamma=gamma)
    return jnp.dot(K_cross, dual_coef)


@jax.jit
def nystrom_approximate_gram(X: jnp.ndarray, landmarks: jnp.ndarray, gamma: float = 0.125, eps: float = 1e-4) -> jnp.ndarray:
    """Computes Nyström approximation K_nm @ pinv(K_mm) @ K_nm.T."""
    m = landmarks.shape[0]
    K_mm = compute_kernel_matrix(landmarks, landmarks, kernel="rbf", gamma=gamma)
    K_nm = compute_kernel_matrix(X, landmarks, kernel="rbf", gamma=gamma)
    K_mm_inv = jnp.linalg.pinv(K_mm + eps * jnp.eye(m))
    return jnp.dot(K_nm, jnp.dot(K_mm_inv, K_nm.T))


@jax.jit
def primal_svm_loss(w: jnp.ndarray, b: float, X: jnp.ndarray, y: jnp.ndarray, C: float = 1.0) -> jnp.ndarray:
    """Primal Soft-Margin SVM objective: 0.5 * ||w||^2 + C * mean(max(0, 1 - y * (X w + b)))."""
    preds = jnp.dot(X, w) + b
    hinge = jnp.mean(jnp.maximum(1.0 - y * preds, 0.0))
    reg = 0.5 * jnp.sum(w ** 2)
    return reg + C * hinge


def run_verification() -> bool:
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, shape=(64, 8))
    w_true = jnp.array([1.2, -0.8, 0.5, -1.0, 0.3, -0.2, 0.7, -0.4])
    y_cont = jnp.dot(X, w_true) + 0.1 * jax.random.normal(key, shape=(64,))
    gamma = 0.125

    dual_coef = solve_kernel_ridge(X, y_cont, gamma=gamma, alpha=0.1)
    preds = predict_kernel_ridge(X, X, dual_coef, gamma=gamma)
    res = jnp.linalg.norm(preds - y_cont)
    assert res < 5.0, f"JAX KRR residual too high: {res}"

    landmarks = X[:32]
    K_approx = nystrom_approximate_gram(X, landmarks, gamma=gamma)
    K_exact = compute_kernel_matrix(X, X, kernel="rbf", gamma=gamma)
    err = jnp.linalg.norm(K_approx - K_exact) / jnp.linalg.norm(K_exact)
    assert err < 0.20, f"JAX Nystrom relative error {err} too high"
    print("JAX Topic 02 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
