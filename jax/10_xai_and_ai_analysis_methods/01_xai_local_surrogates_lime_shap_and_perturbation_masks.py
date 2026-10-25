"""XAI Local Surrogates LIME SHAP and Perturbation Masks — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks. Run ``python jax/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def lime_weighted_regression(
    z_prime: jnp.ndarray,
    y: jnp.ndarray,
    pi: jnp.ndarray,
    reg: float = 1e-4,
) -> jnp.ndarray:
    sqrt_pi = jnp.expand_dims(jnp.sqrt(pi), -1)
    z_w = z_prime * sqrt_pi
    y_w = y * jnp.sqrt(pi)
    d = z_prime.shape[1]
    lhs = z_w.T @ z_w + reg * jnp.eye(d)
    rhs = z_w.T @ y_w
    return jnp.linalg.solve(lhs, rhs)


def shapley_kernel_weights(
    coalition_sizes: jnp.ndarray,
    m_features: int = 4,
) -> jnp.ndarray:
    weights = []
    for s_val in coalition_sizes:
        s = int(s_val)
        if 0 < s < m_features:
            binom = math.comb(m_features, s)
            w = float(m_features - 1) / (float(binom) * float(s) * float(m_features - s))
        else:
            w = 1000.0
        weights.append(w)
    return jnp.asarray(weights, dtype=jnp.float32)


def rise_feature_importance(
    masks: jnp.ndarray,
    model_outputs: jnp.ndarray,
) -> jnp.ndarray:
    n = float(masks.shape[0])
    return (jnp.expand_dims(model_outputs, 0) @ masks).squeeze(0) / n


def faithfulness_degradation(
    y_orig: jnp.ndarray,
    degraded_y: jnp.ndarray,
) -> jnp.ndarray:
    return y_orig - degraded_y


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # LIME
    lime_w = lime_weighted_regression(g["lime_z"], g["lime_y"], g["lime_pi"], reg=1e-4)
    assert float(jnp.max(jnp.abs(lime_w - g["lime_w_ref"]))) < 1e-5, "LIME weights mismatch"

    # KernelSHAP
    shap_w = shapley_kernel_weights(g["shap_sizes"], m_features=4)
    assert float(jnp.max(jnp.abs(shap_w - g["shap_weights_ref"]))) < 1e-5, "SHAP weights mismatch"

    # RISE
    rise_imp = rise_feature_importance(g["rise_masks"], g["rise_outputs"])
    assert float(jnp.max(jnp.abs(rise_imp - g["rise_importance_ref"]))) < 1e-5, "RISE importance mismatch"

    # ROAR
    roar_deg = faithfulness_degradation(g["roar_orig"], g["roar_degraded"])
    assert float(jnp.max(jnp.abs(roar_deg - g["roar_degradation_ref"]))) < 1e-5, "ROAR degradation mismatch"

    print("10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
