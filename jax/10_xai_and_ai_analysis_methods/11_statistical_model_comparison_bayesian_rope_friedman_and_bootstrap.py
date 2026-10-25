"""Statistical Model Comparison Bayesian RoPE Friedman and Bootstrap — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap. Run ``python jax/10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.scipy.special as jsp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def mcnemar_test_statistic(b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    diff = jnp.abs(b - c) - 1.0
    return (diff * diff) / (b + c + 1e-8)


def nadeau_bengio_corrected_t(
    diffs: jnp.ndarray,
    n_train: float,
    n_test: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    k = float(diffs.shape[0])
    d_bar = jnp.mean(diffs)
    s_sq = jnp.sum((diffs - d_bar) ** 2) / (k - 1.0)
    c_factor = 1.0 / k + float(n_test) / float(n_train)
    sigma = jnp.sqrt(c_factor * s_sq + 1e-8)
    t_stat = d_bar / sigma
    return t_stat, sigma


def friedman_rank_statistic(ranks: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = float(ranks.shape[0])
    k = float(ranks.shape[1])
    avg_ranks = jnp.mean(ranks, axis=0)
    sum_sq = jnp.sum(avg_ranks * avg_ranks)
    chi_f = (12.0 * n / (k * (k + 1.0))) * (sum_sq - (k * (k + 1.0) ** 2) / 4.0)
    return chi_f, avg_ranks


def bayesian_rope_probabilities(
    diffs: jnp.ndarray,
    n_train: float,
    n_test: float,
    rope_radius: float = 0.01,
) -> jnp.ndarray:
    k = float(diffs.shape[0])
    d_bar = jnp.mean(diffs)
    s_sq = jnp.sum((diffs - d_bar) ** 2) / (k - 1.0)
    c_factor = 1.0 / k + float(n_test) / float(n_train)
    sigma = jnp.sqrt(c_factor * s_sq + 1e-8)

    r = float(rope_radius)
    z1 = (r - d_bar) / sigma
    z2 = (-r - d_bar) / sigma

    sqrt2 = 1.4142135623730951
    phi1 = 0.5 * (1.0 + jsp.erf(z1 / sqrt2))
    phi2 = 0.5 * (1.0 + jsp.erf(z2 / sqrt2))

    p_rope = phi1 - phi2
    p_better = 1.0 - phi1
    p_worse = phi2

    return jnp.stack([p_better, p_rope, p_worse])


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. McNemar
    mcn = mcnemar_test_statistic(g["mcnemar_b"], g["mcnemar_c"])
    assert float(jnp.max(jnp.abs(mcn - g["mcnemar_stat_ref"]))) < 1e-5, "McNemar mismatch"

    # 3. Nadeau-Bengio
    t_stat, sigma = nadeau_bengio_corrected_t(g["cv_diffs"], 800.0, 200.0)
    assert float(jnp.max(jnp.abs(t_stat - g["nb_t_ref"]))) < 1e-5, "NB t mismatch"
    assert float(jnp.max(jnp.abs(sigma - g["nb_sigma_ref"]))) < 1e-5, "NB sigma mismatch"

    # 4. Friedman
    chi_f, avg_r = friedman_rank_statistic(g["ranks"])
    assert float(jnp.max(jnp.abs(chi_f - g["friedman_stat_ref"]))) < 1e-5, "Friedman stat mismatch"
    assert float(jnp.max(jnp.abs(avg_r - g["avg_ranks_ref"]))) < 1e-5, "Avg ranks mismatch"

    # 5. Bayesian ROPE
    rope_probs = bayesian_rope_probabilities(g["cv_diffs"], 800.0, 200.0, 0.01)
    assert float(jnp.max(jnp.abs(rope_probs - g["rope_probs_ref"]))) < 1e-5, "ROPE mismatch"

    print("10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
