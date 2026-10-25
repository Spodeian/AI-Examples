"""Surrogate Models Bayesian Optimization & Differentiable Proxies — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies. Run ``python jax/06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def rbf_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    lengthscale: jnp.ndarray,
    variance: jnp.ndarray,
) -> jnp.ndarray:
    diff = (jnp.expand_dims(x1, 1) - jnp.expand_dims(x2, 0)) / lengthscale
    dist_sq = jnp.sum(diff**2, axis=-1)
    return variance * jnp.exp(-0.5 * dist_sq)


def gp_posterior(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_test: jnp.ndarray,
    lengthscale: jnp.ndarray,
    variance: jnp.ndarray,
    noise_var: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    N = x_train.shape[0]
    K_yy = rbf_kernel(x_train, x_train, lengthscale, variance) + noise_var * jnp.eye(N, dtype=x_train.dtype)
    K_sy = rbf_kernel(x_test, x_train, lengthscale, variance)
    K_ss = rbf_kernel(x_test, x_test, lengthscale, variance)
    
    alpha = jnp.linalg.solve(K_yy, jnp.expand_dims(y_train, -1)).squeeze(-1)
    mu_s = K_sy @ alpha
    
    V = jnp.linalg.solve(K_yy, K_sy.T)
    cov_s = K_ss - K_sy @ V
    var_s = jnp.clip(jnp.diag(cov_s), a_min=1e-6)
    return mu_s, var_s


def expected_improvement(
    mu: jnp.ndarray,
    var: jnp.ndarray,
    best_y: jnp.ndarray,
    xi: float = 0.01,
) -> jnp.ndarray:
    sigma = jnp.sqrt(var)
    improvement = mu - best_y - xi
    Z = improvement / sigma
    phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * jnp.exp(-0.5 * Z**2)
    phi_cdf = 0.5 * (1.0 + jax.lax.erf(Z / math.sqrt(2.0)))
    ei = improvement * phi_cdf + sigma * phi_z
    return jnp.clip(ei, a_min=0.0)


def ucb_acquisition(
    mu: jnp.ndarray,
    var: jnp.ndarray,
    beta: float = 2.0,
) -> jnp.ndarray:
    sigma = jnp.sqrt(var)
    return mu + beta * sigma


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"
    
    k_sy = rbf_kernel(g["x_test"], g["x_train"], g["lengthscale"], g["variance"])
    assert float(jnp.max(jnp.abs(k_sy - g["k_sy"]))) < 1e-5
    
    mu_s, var_s = gp_posterior(g["x_train"], g["y_train"], g["x_test"], g["lengthscale"], g["variance"], g["noise_var"])
    assert float(jnp.max(jnp.abs(mu_s - g["mu_s"]))) < 1e-5
    assert float(jnp.max(jnp.abs(var_s - g["var_s"]))) < 1e-5
    
    ei = expected_improvement(g["mu_s"], g["var_s"], g["best_y"], xi=0.01)
    assert float(jnp.max(jnp.abs(ei - g["ei_acq"]))) < 1e-5
    
    ucb = ucb_acquisition(g["mu_s"], g["var_s"], beta=2.0)
    assert float(jnp.max(jnp.abs(ucb - g["ucb_acq"]))) < 1e-5
    
    print("06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
