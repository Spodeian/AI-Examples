"""Functional JAX/Equinox implementation for Category 01 Topic 01:
01_generative_vs_discriminative_classifiers
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp

    import jax
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
    _ = jax
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


def fit_gda_parameters(
    x: Any,
    y: Any,
    num_classes: int,
    reg_lambda: float = 1e-4,
) -> tuple[Any, Any, Any, Any, Any]:
    """Fits Gaussian Discriminant Analysis (GDA) with shared covariance and derives equivalent linear parameters."""
    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.int64)
    num_samples, dim = x_np.shape

    means = np.zeros((num_classes, dim), dtype=np.float32)
    priors = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        mask = y_np == c
        count = float(np.sum(mask))
        priors[c] = count / float(num_samples)
        means[c] = np.mean(x_np[mask], axis=0)

    centered_x = x_np.copy()
    for c in range(num_classes):
        mask = y_np == c
        centered_x[mask] = x_np[mask] - means[c]

    shared_cov = np.dot(centered_x.T, centered_x) / float(num_samples)
    reg_cov = shared_cov + reg_lambda * np.eye(dim, dtype=np.float32)
    cov_inv = np.linalg.inv(reg_cov)

    w_derived = np.zeros((dim, num_classes), dtype=np.float32)
    b_derived = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        mu_c = means[c]
        sigma_inv_mu = np.dot(cov_inv, mu_c)
        w_derived[:, c] = sigma_inv_mu
        b_derived[c] = -0.5 * float(np.dot(mu_c, sigma_inv_mu)) + np.log(priors[c] + 1e-12)

    if HAS_JAX and isinstance(x, Array):
        return (
            jnp.asarray(means),
            jnp.asarray(reg_cov),
            jnp.asarray(priors),
            jnp.asarray(w_derived),
            jnp.asarray(b_derived),
        )
    return means, reg_cov, priors, w_derived, b_derived


def predict_logits_gda(x: Any, w_derived: Any, b_derived: Any) -> Any:
    """Evaluates generative linear discriminant logits."""
    x_arr = jnp.asarray(x) if HAS_JAX else np.asarray(x)
    w_arr = jnp.asarray(w_derived) if HAS_JAX else np.asarray(w_derived)
    b_arr = jnp.asarray(b_derived) if HAS_JAX else np.asarray(b_derived)
    return x_arr @ w_arr + b_arr


def predict_proba_softmax(logits: Any) -> Any:
    """Evaluates Softmax probabilities along the last dimension."""
    if HAS_JAX:
        return jax.nn.softmax(logits, axis=-1)
    m = np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - m)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


class FunctionalDiscriminativeClassifier:
    """Stateless functional discriminative linear classifier."""

    def __init__(self, weight: Any, bias: Any) -> None:
        self.weight = weight
        self.bias = bias

    def __call__(self, x: Any) -> Any:
        return x @ self.weight + self.bias
