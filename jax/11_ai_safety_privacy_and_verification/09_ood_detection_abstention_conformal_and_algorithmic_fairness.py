"""OOD Detection Abstention Conformal and Algorithmic Fairness — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness. Run ``python jax/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def energy_ood_score(logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """Helmholtz free energy OOD score (Liu et al. 2020): E(x) = -T * logsumexp(logits / T)."""
    return -float(temperature) * jax.nn.logsumexp(logits / float(temperature), axis=-1)


def split_conformal_quantile(cal_scores: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Split Conformal Prediction (Vovk et al. 2005; Angelopoulos & Bates 2021) 1 - alpha quantile."""
    sorted_scores = jnp.sort(cal_scores)
    n = sorted_scores.shape[0]
    idx = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - float(alpha))) - 1))
    return sorted_scores[idx]


def selective_net_loss(
    ce_losses: jnp.ndarray,
    selection_probs: jnp.ndarray,
    target_coverage: float,
    penalty_weight: float,
) -> jnp.ndarray:
    """SelectiveNet risk with coverage penalty constraint (Geifman & El-Yaniv 2019)."""
    phi_hat = jnp.mean(selection_probs)
    empirical_risk = jnp.sum(ce_losses * selection_probs) / (jnp.sum(selection_probs) + 1e-8)
    coverage_deficit = jnp.maximum(0.0, float(target_coverage) - phi_hat)
    penalty = float(penalty_weight) * (coverage_deficit ** 2)
    return empirical_risk + penalty


def equalized_odds_discrepancy(
    y_pred: jnp.ndarray,
    y_true: jnp.ndarray,
    sensitive_attr: jnp.ndarray,
) -> jnp.ndarray:
    """Equalized Odds discrepancy (Hardt et al. 2016): |TPR_0 - TPR_1| + |FPR_0 - FPR_1|."""
    mask_a0 = sensitive_attr == 0.0
    mask_a1 = sensitive_attr == 1.0
    mask_y1 = y_true == 1.0
    mask_y0 = y_true == 0.0

    tpr_0 = jnp.sum(y_pred * mask_a0 * mask_y1) / (jnp.sum(mask_a0 * mask_y1) + 1e-8)
    tpr_1 = jnp.sum(y_pred * mask_a1 * mask_y1) / (jnp.sum(mask_a1 * mask_y1) + 1e-8)
    fpr_0 = jnp.sum(y_pred * mask_a0 * mask_y0) / (jnp.sum(mask_a0 * mask_y0) + 1e-8)
    fpr_1 = jnp.sum(y_pred * mask_a1 * mask_y0) / (jnp.sum(mask_a1 * mask_y0) + 1e-8)

    return jnp.abs(tpr_0 - tpr_1) + jnp.abs(fpr_0 - fpr_1)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Energy OOD
    energy = energy_ood_score(g["logits"], temperature=1.0)
    err = float(jnp.max(jnp.abs(energy - g["energy_ref"])))
    assert err < 1e-5, f"energy error {err:.2e}"

    # 3. Conformal quantile
    q_hat = split_conformal_quantile(g["cal_scores"], alpha=0.1)
    err = float(jnp.max(jnp.abs(q_hat - g["q_hat_ref"])))
    assert err < 1e-5, f"q_hat error {err:.2e}"

    # 4. SelectiveNet loss
    sel_loss = selective_net_loss(g["ce_losses"], g["selection_probs"], target_coverage=0.7, penalty_weight=5.0)
    err = float(jnp.max(jnp.abs(sel_loss - g["selective_loss_ref"])))
    assert err < 1e-5, f"sel_loss error {err:.2e}"

    # 5. Equalized Odds
    eo_disc = equalized_odds_discrepancy(g["y_pred"], g["y_true"], g["sensitive_attr"])
    err = float(jnp.max(jnp.abs(eo_disc - g["eo_discrepancy_ref"])))
    assert err < 1e-5, f"eo_disc error {err:.2e}"

    print("11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
