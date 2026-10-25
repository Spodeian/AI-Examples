"""Causal Representation Identifiability iVAE Synthetic Control and Symbolic Sr — functional JAX implementation.

Topic: 09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr. Run ``python jax/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ivae_kl_divergence(
    mu_z: jnp.ndarray,
    logvar_z: jnp.ndarray,
    mu_u: jnp.ndarray,
    logvar_u: jnp.ndarray,
) -> jnp.ndarray:
    var_z = jnp.exp(logvar_z)
    var_u = jnp.exp(logvar_u)
    kl_per_dim = 0.5 * (
        (var_z + (mu_z - mu_u) ** 2) / var_u - 1.0 - (logvar_z - logvar_u)
    )
    return jnp.sum(kl_per_dim, axis=-1)


def project_simplex(v: jnp.ndarray) -> jnp.ndarray:
    n_features = v.shape[0]
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u) - 1.0
    ind = jnp.arange(1, n_features + 1, dtype=v.dtype)
    cond = u - cssv / ind > 0.0
    rho = jnp.max(jnp.where(cond, ind, 0)).astype(jnp.int32)
    theta = cssv[rho - 1] / rho.astype(v.dtype)
    w = jnp.clip(v - theta, 0.0)
    return w


def synthetic_control_effect(
    y1: jnp.ndarray,
    y0: jnp.ndarray,
    w: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    y1_synth = y0 @ w
    tau = y1 - y1_synth
    return y1_synth, tau


def symbolic_pareto_loss(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    complexity: jnp.ndarray,
    lambda_reg: jnp.ndarray,
) -> jnp.ndarray:
    mse = jnp.mean((y_true - y_pred) ** 2)
    return mse + lambda_reg * complexity


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # iVAE KL
    kl = ivae_kl_divergence(g["mu_z"], g["logvar_z"], g["mu_u"], g["logvar_u"])
    assert float(jnp.max(jnp.abs(kl - g["ivae_kl_ref"]))) < 1e-5, "iVAE KL mismatch"

    # Synthetic Control
    w_simp = project_simplex(g["sc_v"])
    assert float(jnp.max(jnp.abs(w_simp - g["sc_w_simplex_ref"]))) < 1e-5, "Simplex mismatch"

    synth, tau = synthetic_control_effect(g["sc_y1_post"], g["sc_y0_post"], w_simp)
    assert float(jnp.max(jnp.abs(synth - g["sc_y1_synth_ref"]))) < 1e-5, "Synthetic control mismatch"
    assert float(jnp.max(jnp.abs(tau - g["sc_tau_ref"]))) < 1e-5, "Treatment effect mismatch"

    # Symbolic Pareto
    pareto = symbolic_pareto_loss(g["sr_y_true"], g["sr_y_pred"], g["sr_complexity"], g["sr_lambda"])
    assert float(jnp.max(jnp.abs(pareto - g["sr_pareto_loss_ref"]))) < 1e-5, "Pareto loss mismatch"

    print("09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
