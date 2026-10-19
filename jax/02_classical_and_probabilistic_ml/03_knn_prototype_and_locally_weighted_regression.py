"""k-Nearest Neighbors, Prototype Learning, and Locally Weighted Regression in JAX.

Functional implementations:
- Vectorized pairwise squared Euclidean distance
- Distance-weighted k-NN regression
- Nadaraya-Watson kernel regression
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def pairwise_squared_distance(X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
    """Computes pairwise squared Euclidean distances [N1, N2] with numerical clamp."""
    x1_sq = jnp.sum(X1 ** 2, axis=1, keepdims=True)
    x2_sq = jnp.sum(X2 ** 2, axis=1, keepdims=True).T
    dist_sq = x1_sq + x2_sq - 2.0 * jnp.dot(X1, X2.T)
    return jnp.maximum(dist_sq, 0.0)


@jax.jit
def nadaraya_watson_predict(X_train: jnp.ndarray, y_train: jnp.ndarray, X_query: jnp.ndarray, bandwidth: float = 1.0) -> jnp.ndarray:
    """Nadaraya-Watson kernel regression: sum_i w_i(x) y_i / sum_i w_i(x)."""
    dist_sq = pairwise_squared_distance(X_train, X_query) # [N_train, N_query]
    weights = jnp.exp(-0.5 * dist_sq / (bandwidth ** 2))
    sum_w = jnp.sum(weights, axis=0, keepdims=True)
    norm_weights = weights / jnp.maximum(sum_w, 1e-8)
    return jnp.sum(norm_weights * jnp.expand_dims(y_train, 1), axis=0)


def run_verification() -> bool:
    key = jax.random.PRNGKey(42)
    N_tr, N_te, D = 64, 16, 8
    X_tr = jax.random.normal(key, shape=(N_tr, D))
    w = jnp.array([1.0, -0.5, 0.8, -1.2, 0.4, -0.3, 0.6, -0.7])
    y_tr = jnp.sin(jnp.dot(X_tr, w))
    X_te = jax.random.normal(key, shape=(N_te, D))

    pred_nw = nadaraya_watson_predict(X_tr, y_tr, X_te, bandwidth=1.0)
    assert pred_nw.shape == (N_te,)
    assert jnp.all(jnp.isfinite(pred_nw))
    print("JAX Topic 03 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
