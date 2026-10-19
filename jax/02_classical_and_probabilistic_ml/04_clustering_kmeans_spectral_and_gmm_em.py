"""Clustering: k-Means, Spectral Clustering, and GMM Expectation-Maximization in JAX.

Functional implementations:
- Vectorized k-Means assignment & Lloyd step
- GMM log-domain Expectation-Maximization step
- Symmetric normalized Graph Laplacian
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def kmeans_assign(X: jnp.ndarray, centers: jnp.ndarray) -> jnp.ndarray:
    """Assigns each sample in X to closest cluster center."""
    x_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
    c_sq = jnp.sum(centers ** 2, axis=1, keepdims=True).T
    dist_sq = jnp.maximum(x_sq + c_sq - 2.0 * jnp.dot(X, centers.T), 0.0)
    return jnp.argmin(dist_sq, axis=1)


@jax.jit
def gmm_e_step(X: jnp.ndarray, mu: jnp.ndarray, var: jnp.ndarray, pi: jnp.ndarray) -> jnp.ndarray:
    """Computes posterior responsibilities in log domain: resp = softmax(log pi + log p(x|mu, var))."""
    N, D = X.shape
    K = mu.shape[0]

    def log_prob_k(k: int):
        diff = X - mu[k]
        quad = jnp.sum(diff ** 2 / var[k], axis=1)
        log_det = jnp.sum(jnp.log(var[k]))
        return -0.5 * (D * 1.8378770664 + log_det + quad)

    # Compute log-probs for all K components: [N, K]
    log_probs = jnp.stack([log_prob_k(k) for k in range(K)], axis=1)
    log_joint = jnp.log(pi + 1e-8) + log_probs
    log_norm = jax.nn.logsumexp(log_joint, axis=1, keepdims=True)
    return jnp.exp(log_joint - log_norm)


@jax.jit
def compute_symmetric_laplacian(X: jnp.ndarray, gamma: float = 0.1) -> jnp.ndarray:
    """Computes symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}."""
    N = X.shape[0]
    x_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
    dist_sq = jnp.maximum(x_sq + x_sq.T - 2.0 * jnp.dot(X, X.T), 0.0)
    A = jnp.exp(-gamma * dist_sq)
    A = A * (1.0 - jnp.eye(N)) # zero out diagonal
    degrees = jnp.sum(A, axis=1)
    d_inv_sqrt = jnp.power(jnp.maximum(degrees, 1e-6), -0.5)
    return jnp.eye(N) - (jnp.expand_dims(d_inv_sqrt, 1) * A * jnp.expand_dims(d_inv_sqrt, 0))


def run_verification() -> bool:
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, shape=(40, 8))
    centers = X[:2]

    assigns = kmeans_assign(X, centers)
    assert assigns.shape == (40,)

    pi = jnp.array([0.5, 0.5])
    mu = centers
    var = jnp.ones((2, 8))
    resp = gmm_e_step(X, mu, var, pi)
    assert resp.shape == (40, 2)
    assert jnp.allclose(jnp.sum(resp, axis=1), 1.0, atol=1e-5)

    L = compute_symmetric_laplacian(X, gamma=0.1)
    assert L.shape == (40, 40)
    print("JAX Topic 04 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
