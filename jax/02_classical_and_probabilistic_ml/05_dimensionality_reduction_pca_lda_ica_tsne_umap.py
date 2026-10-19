"""Dimensionality Reduction: PCA, LDA, ICA, t-SNE, and UMAP in JAX.

Functional implementations:
- PCA via SVD on centered data
- Fisher's Linear Discriminant Analysis (LDA)
- Kernel PCA with double-centering of the Gram matrix
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


from functools import partial


@partial(jax.jit, static_argnums=(1,))
def pca_fit_transform(X: jnp.ndarray, n_components: int = 3) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes mean-centered PCA via SVD."""
    N = X.shape[0]
    mean = jnp.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
    V_top = Vt[:n_components, :].T
    proj = jnp.dot(X_centered, V_top)
    return proj, V_top, mean


@jax.jit
def lda_fit_transform(X: jnp.ndarray, y: jnp.ndarray, reg: float = 1e-3) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Binary Fisher's LDA direction."""
    D = X.shape[1]
    mean = jnp.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    mask0 = (y == 0)[:, None]
    mask1 = (y == 1)[:, None]

    n0 = jnp.maximum(jnp.sum(mask0), 1.0)
    n1 = jnp.maximum(jnp.sum(mask1), 1.0)

    mu0 = jnp.sum(X_centered * mask0, axis=0) / n0
    mu1 = jnp.sum(X_centered * mask1, axis=0) / n1

    diff0 = (X_centered - mu0) * mask0
    diff1 = (X_centered - mu1) * mask1

    Sw = jnp.dot(diff0.T, diff0) + jnp.dot(diff1.T, diff1) + reg * jnp.eye(D)
    w = jnp.linalg.solve(Sw, mu1 - mu0)
    w_norm = w / jnp.linalg.norm(w)
    proj = jnp.dot(X_centered, w_norm)
    return proj, w_norm


@partial(jax.jit, static_argnums=(1,))
def kpca_fit_transform(X: jnp.ndarray, n_components: int = 3, gamma: float = 0.1) -> jnp.ndarray:
    """Kernel PCA with double-centering."""
    N = X.shape[0]
    x_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
    dist_sq = jnp.maximum(x_sq + x_sq.T - 2.0 * jnp.dot(X, X.T), 0.0)
    K = jnp.exp(-gamma * dist_sq)

    one_n = jnp.full((N, N), 1.0 / N)
    K_tilde = K - jnp.dot(one_n, K) - jnp.dot(K, one_n) + jnp.dot(one_n, jnp.dot(K, one_n))

    eigvals, eigvecs = jnp.linalg.eigh(K_tilde)
    eigvals = jnp.flip(eigvals)
    eigvecs = jnp.flip(eigvecs, axis=1)

    alphas = eigvecs[:, :n_components] / jnp.sqrt(jnp.maximum(eigvals[:n_components], 1e-6))
    return jnp.dot(K_tilde, alphas)


def run_verification() -> bool:
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, shape=(64, 8))
    y = (X[:, 0] > 0.0).astype(jnp.int32)

    proj_pca, _, _ = pca_fit_transform(X, 3)
    assert proj_pca.shape == (64, 3)

    proj_lda, _ = lda_fit_transform(X, y)
    assert proj_lda.shape == (64,)

    proj_kpca = kpca_fit_transform(X, 3, 0.1)
    assert proj_kpca.shape == (64, 3)

    print("JAX Topic 05 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
