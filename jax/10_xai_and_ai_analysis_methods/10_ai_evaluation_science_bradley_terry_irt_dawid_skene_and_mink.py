"""AI Evaluation Science Bradley Terry IRT Dawid Skene and Min-K% — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink. Run ``python jax/10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def btl_win_probability(skills: jnp.ndarray) -> jnp.ndarray:
    diff = skills[:, None] - skills[None, :]
    return 1.0 / (1.0 + jnp.exp(-diff))


def irt_2pl_probability(
    abilities: jnp.ndarray,
    difficulties: jnp.ndarray,
    discriminations: jnp.ndarray,
) -> jnp.ndarray:
    diff = abilities[:, None] - difficulties[None, :]
    return 1.0 / (1.0 + jnp.exp(-discriminations[None, :] * diff))


def dawid_skene_e_step(
    annotations: jnp.ndarray,
    class_priors: jnp.ndarray,
    confusion_matrices: jnp.ndarray,
) -> jnp.ndarray:
    n, j_count = annotations.shape
    k_count = class_priors.shape[0]

    log_rows = []
    for i in range(n):
        k_lps = []
        for k in range(k_count):
            lp = jnp.log(class_priors[k] + 1e-8)
            for j in range(j_count):
                obs = int(annotations[i, j])
                lp = lp + jnp.log(confusion_matrices[j, k, obs] + 1e-8)
            k_lps.append(lp)
        log_rows.append(jnp.stack(k_lps))

    log_post = jnp.stack(log_rows)
    exp_lp = jnp.exp(log_post - jnp.max(log_post, axis=-1, keepdims=True))
    return exp_lp / jnp.sum(exp_lp, axis=-1, keepdims=True)


def min_k_percent_prob(token_log_probs: jnp.ndarray, k_fraction: float = 0.25) -> jnp.ndarray:
    sorted_lp = jnp.sort(token_log_probs)
    k_count = max(1, int(round(k_fraction * token_log_probs.shape[0])))
    return jnp.mean(sorted_lp[:k_count])


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. BTL win probabilities
    btl_p = btl_win_probability(g["btl_skills"])
    assert float(jnp.max(jnp.abs(btl_p - g["btl_win_prob_ref"]))) < 1e-5, "BTL mismatch"

    # 3. 2PL IRT
    irt_p = irt_2pl_probability(g["irt_abilities"], g["irt_difficulties"], g["irt_discriminations"])
    assert float(jnp.max(jnp.abs(irt_p - g["irt_prob_ref"]))) < 1e-5, "IRT mismatch"

    # 4. Dawid-Skene E-step
    ds_post = dawid_skene_e_step(g["ds_annotations"], g["ds_priors"], g["ds_confusion"])
    assert float(jnp.max(jnp.abs(ds_post - g["ds_posterior_ref"]))) < 1e-5, "Dawid-Skene mismatch"

    # 5. Min-K% Prob
    mink = min_k_percent_prob(g["mink_log_probs"], 0.25)
    assert float(jnp.max(jnp.abs(mink - g["mink_score_ref"]))) < 1e-5, "Min-K% mismatch"

    print("10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
