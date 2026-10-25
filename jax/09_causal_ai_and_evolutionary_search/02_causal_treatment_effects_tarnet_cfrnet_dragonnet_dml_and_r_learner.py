"""Causal Treatment Effects TARNet CFRNet Dragonnet DML and R Learner — functional JAX implementation.

Topic: 09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner. Run ``python jax/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def tarnet_forward(
    x: jnp.ndarray,
    phi_w: jnp.ndarray,
    phi_b: jnp.ndarray,
    h0_w: jnp.ndarray,
    h0_b: jnp.ndarray,
    h1_w: jnp.ndarray,
    h1_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Treatment-Agnostic Representation Network (TARNet)."""
    phi = jax.nn.relu(x @ phi_w + phi_b)
    y0 = phi @ h0_w + h0_b
    y1 = phi @ h1_w + h1_b
    tau = y1 - y0
    return phi, y0, y1, tau


def cfrnet_mmd_loss(phi: jnp.ndarray, t: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """CFRNet linear MMD representation balance penalty."""
    t_mask = jnp.squeeze(t, -1)
    n_treated = jnp.maximum(jnp.sum(t_mask), 1.0)
    n_control = jnp.maximum(jnp.sum(1.0 - t_mask), 1.0)

    mu_1 = jnp.sum(phi * t_mask[:, None], axis=0) / n_treated
    mu_0 = jnp.sum(phi * (1.0 - t_mask)[:, None], axis=0) / n_control

    diff = mu_1 - mu_0
    return jnp.sum(diff * diff)[None]


def dragonnet_propensity_loss(
    phi: jnp.ndarray,
    t: jnp.ndarray,
    w_prop: jnp.ndarray,
    b_prop: jnp.ndarray,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dragonnet propensity score head and BCE loss."""
    logits = phi @ w_prop + b_prop
    propensity = jax.nn.sigmoid(logits)
    p_c = jnp.clip(propensity, a_min=eps, a_max=1.0 - eps)
    bce = -jnp.mean(t * jnp.log(p_c) + (1.0 - t) * jnp.log(1.0 - p_c))[None]
    return propensity, bce


def dml_cate_estimate(
    y: jnp.ndarray,
    t: jnp.ndarray,
    m_hat: jnp.ndarray,
    e_hat: jnp.ndarray,
) -> jnp.ndarray:
    """Double Machine Learning (DML) Robinson estimator."""
    res_t = t - e_hat
    res_y = y - m_hat
    num = jnp.sum(res_t * res_y)
    den = jnp.maximum(jnp.sum(res_t * res_t), 1e-8)
    return (num / den)[None]


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # TARNet
    phi, y0, y1, tau = tarnet_forward(
        g["test_input"], g["phi_w"], g["phi_b"], g["h0_w"], g["h0_b"], g["h1_w"], g["h1_b"]
    )
    err_y0 = float(jnp.max(jnp.abs(y0 - g["y0_ref"])))
    assert err_y0 < 1e-5, f"TARNet y0 error {err_y0:.2e}"
    err_y1 = float(jnp.max(jnp.abs(y1 - g["y1_ref"])))
    assert err_y1 < 1e-5, f"TARNet y1 error {err_y1:.2e}"
    err_tau = float(jnp.max(jnp.abs(tau - g["tau_ref"])))
    assert err_tau < 1e-5, f"TARNet tau error {err_tau:.2e}"

    # CFRNet MMD
    mmd = cfrnet_mmd_loss(phi, g["t_treatment"])
    err_mmd = float(jnp.max(jnp.abs(mmd - g["mmd_loss_ref"])))
    assert err_mmd < 1e-5, f"CFRNet MMD error {err_mmd:.2e}"

    # Dragonnet
    prop, bce = dragonnet_propensity_loss(phi, g["t_treatment"], g["w_prop"], g["b_prop"])
    err_prop = float(jnp.max(jnp.abs(prop - g["propensity_ref"])))
    assert err_prop < 1e-5, f"Dragonnet propensity error {err_prop:.2e}"
    err_bce = float(jnp.max(jnp.abs(bce - g["prop_loss_ref"])))
    assert err_bce < 1e-5, f"Dragonnet BCE error {err_bce:.2e}"

    # DML
    dml_tau = dml_cate_estimate(g["y_outcome"], g["t_treatment"], g["m_hat"], g["e_hat"])
    err_dml = float(jnp.max(jnp.abs(dml_tau - g["dml_tau_ref"])))
    assert err_dml < 1e-5, f"DML tau error {err_dml:.2e}"

    print("09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
