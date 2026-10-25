"""XAI Concepts TCAV CBM Counterfactuals DiCE and Global ALE — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale. Run ``python jax/10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def tcav_directional_derivatives(
    activation_grads: jnp.ndarray,
    cav: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    cav_norm = cav / jnp.sqrt(jnp.sum(cav * cav) + 1e-8)
    dir_derivs = activation_grads @ cav_norm
    positive_mask = jnp.where(dir_derivs > 0.0, 1.0, 0.0)
    tcav_score = jnp.mean(positive_mask)
    return dir_derivs, tcav_score


def cbm_test_time_intervention(
    pred_concepts: jnp.ndarray,
    true_concepts: jnp.ndarray,
    intervention_mask: jnp.ndarray,
    task_weights: jnp.ndarray,
    task_bias: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    intervened_concepts = (1.0 - intervention_mask) * pred_concepts + intervention_mask * true_concepts
    task_preds = intervened_concepts @ task_weights + task_bias
    return intervened_concepts, task_preds


def dice_dpp_diversity(counterfactuals: jnp.ndarray) -> jnp.ndarray:
    diff = jnp.expand_dims(counterfactuals, 1) - jnp.expand_dims(counterfactuals, 0)
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-8)
    kernel_mat = 1.0 / (1.0 + dist)
    return jnp.linalg.det(kernel_mat)


def ale_accumulated_local_effects(interval_diffs: jnp.ndarray) -> jnp.ndarray:
    cumsum_ale = jnp.cumsum(interval_diffs, axis=0)
    centered_ale = cumsum_ale - jnp.mean(cumsum_ale)
    return centered_ale


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # TCAV
    derivs, score = tcav_directional_derivatives(g["tcav_grads"], g["tcav_cav"])
    assert float(jnp.max(jnp.abs(derivs - g["tcav_derivs_ref"]))) < 1e-5, "TCAV derivs mismatch"
    assert float(jnp.max(jnp.abs(score - g["tcav_score_ref"]))) < 1e-5, "TCAV score mismatch"

    # CBM
    c_int, y_pred = cbm_test_time_intervention(g["cbm_pred_c"], g["cbm_true_c"], g["cbm_mask"], g["cbm_w"], g["cbm_b"])
    assert float(jnp.max(jnp.abs(c_int - g["cbm_c_int_ref"]))) < 1e-5, "CBM intervened concepts mismatch"
    assert float(jnp.max(jnp.abs(y_pred - g["cbm_y_ref"]))) < 1e-5, "CBM task preds mismatch"

    # DiCE
    div = dice_dpp_diversity(g["dice_cfs"])
    assert float(jnp.max(jnp.abs(div - g["dice_div_ref"]))) < 1e-5, "DiCE diversity mismatch"

    # ALE
    ale = ale_accumulated_local_effects(g["ale_diffs"])
    assert float(jnp.max(jnp.abs(ale - g["ale_ref"]))) < 1e-5, "ALE mismatch"

    print("10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
