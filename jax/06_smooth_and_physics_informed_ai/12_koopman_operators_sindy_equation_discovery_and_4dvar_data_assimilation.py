"""Koopman Operators SINDy Equation Discovery & 4D-Var Data Assimilation — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation. Run ``python jax/06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def koopman_forward(
    x: jnp.ndarray,
    w_enc: jnp.ndarray,
    b_enc: jnp.ndarray,
    k_mat: jnp.ndarray,
    w_dec: jnp.ndarray,
    b_dec: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    psi = jnp.tanh(x @ w_enc + b_enc)
    psi_next = psi @ k_mat
    x_next = psi_next @ w_dec + b_dec
    return psi, psi_next, x_next


def sindy_library(x: jnp.ndarray) -> jnp.ndarray:
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    ones = jnp.ones_like(x1)
    return jnp.concatenate([
        ones,
        x1,
        x2,
        x1**2,
        x1 * x2,
        x2**2,
        jnp.sin(x1),
        jnp.sin(x2),
    ], axis=-1)


def sindy_step(
    theta_x: jnp.ndarray,
    xi: jnp.ndarray,
    threshold: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_dot_pred = theta_x @ xi
    mask = jnp.abs(xi) >= threshold
    xi_sparse = xi * mask.astype(jnp.float32)
    return x_dot_pred, xi_sparse


def four_d_var_step(
    x_b: jnp.ndarray,
    b_cov_inv: jnp.ndarray,
    y_obs: jnp.ndarray,
    r_cov_inv: jnp.ndarray,
    h_obs: jnp.ndarray,
    m_dyn: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    h_m = m_dyn @ h_obs.T
    d_innov = y_obs - x_b @ h_m
    hessian = b_cov_inv + h_m @ r_cov_inv @ h_m.T
    rhs = d_innov @ r_cov_inv @ h_m.T
    delta_x = jnp.linalg.solve(hessian, rhs.T).T
    x_analysis = x_b + delta_x
    return x_analysis, delta_x


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"
    
    psi, psi_next, x_next = koopman_forward(g["x_state"], g["w_enc"], g["b_enc"], g["k_mat"], g["w_dec"], g["b_dec"])
    assert float(jnp.max(jnp.abs(psi - g["psi"]))) < 1e-5
    assert float(jnp.max(jnp.abs(psi_next - g["psi_next"]))) < 1e-5
    assert float(jnp.max(jnp.abs(x_next - g["x_next"]))) < 1e-5
    
    theta_x = sindy_library(g["x_state"])
    assert float(jnp.max(jnp.abs(theta_x - g["theta_x"]))) < 1e-5
    
    x_dot_pred, xi_sparse = sindy_step(g["theta_x"], g["xi_noisy"], threshold=0.05)
    assert float(jnp.max(jnp.abs(x_dot_pred - g["x_dot_pred"]))) < 1e-5
    assert float(jnp.max(jnp.abs(xi_sparse - g["xi_sparse"]))) < 1e-5
    
    x_analysis, delta_x = four_d_var_step(g["x_b"], g["b_cov_inv"], g["y_obs"], g["r_cov_inv"], g["h_obs"], g["m_dyn"])
    assert float(jnp.max(jnp.abs(x_analysis - g["x_analysis"]))) < 1e-5
    assert float(jnp.max(jnp.abs(delta_x - g["delta_x"]))) < 1e-5
    
    print("06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
