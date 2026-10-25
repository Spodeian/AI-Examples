"""Causal Discovery NOTEARS SCM and Instrumental Variables — functional JAX implementation.

Topic: 09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables. Run ``python jax/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.scipy.linalg
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def notears_acyclicity_constraint(w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Zheng et al. (2018) NOTEARS continuous characterization of acyclicity."""
    d = w.shape[0]
    m = w * w
    exp_m = jax.scipy.linalg.expm(m)
    h = (jnp.trace(exp_m) - float(d))[None]
    grad_h = 2.0 * (exp_m.T * w)
    return h, grad_h


def scm_generate_and_counterfactual(
    scm_w: jnp.ndarray,
    scm_u: jnp.ndarray,
    do_var_idx: int = 1,
    do_val: float = 3.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pearl's SCM forward computation and counterfactual intervention."""
    d = scm_w.shape[0]
    eye = jnp.eye(d)
    inv_w = jnp.linalg.inv(eye - scm_w)
    x_factual = scm_u @ inv_w

    w_int = scm_w.at[:, do_var_idx].set(0.0)
    u_int = scm_u.at[0, do_var_idx].set(do_val)
    inv_w_int = jnp.linalg.inv(eye - w_int)
    x_counterfactual = u_int @ inv_w_int

    return x_factual, x_counterfactual


def two_stage_least_squares(
    z_inst: jnp.ndarray,
    t_treat: jnp.ndarray,
    y_outcome: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Two-Stage Least Squares (2SLS) Instrumental Variable Estimator."""
    zt_z = z_inst.T @ z_inst
    zt_t = z_inst.T @ t_treat
    gamma = jnp.linalg.inv(zt_z) @ zt_t
    t_hat = z_inst @ gamma

    th_th = t_hat.T @ t_hat
    th_y = t_hat.T @ y_outcome
    beta = jnp.linalg.inv(th_th) @ th_y

    return t_hat, beta


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # NOTEARS
    h, grad_h = notears_acyclicity_constraint(g["w_dag"])
    err_h = float(jnp.max(jnp.abs(h - g["h_dag_ref"])))
    assert err_h < 1e-5, f"NOTEARS h error {err_h:.2e}"
    err_grad = float(jnp.max(jnp.abs(grad_h - g["grad_h_ref"])))
    assert err_grad < 1e-5, f"NOTEARS grad_h error {err_grad:.2e}"

    # SCM
    x_fact, x_cf = scm_generate_and_counterfactual(g["scm_w"], g["scm_u"], do_var_idx=1, do_val=3.0)
    err_fact = float(jnp.max(jnp.abs(x_fact - g["scm_x_factual_ref"])))
    assert err_fact < 1e-5, f"SCM factual error {err_fact:.2e}"
    err_cf = float(jnp.max(jnp.abs(x_cf - g["counterfactual_x_ref"])))
    assert err_cf < 1e-5, f"SCM counterfactual error {err_cf:.2e}"

    # 2SLS
    t_hat, beta = two_stage_least_squares(g["z_inst"], g["t_treat"], g["y_outcome"])
    err_that = float(jnp.max(jnp.abs(t_hat - g["t_hat_ref"])))
    assert err_that < 1e-5, f"2SLS T_hat error {err_that:.2e}"
    err_beta = float(jnp.max(jnp.abs(beta - g["beta_2sls_ref"])))
    assert err_beta < 1e-5, f"2SLS beta error {err_beta:.2e}"

    print("09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
