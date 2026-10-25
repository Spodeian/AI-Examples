"""AI Analysis Neural Collapse Superposition and WeightWatcher RMT — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt. Run ``python jax/10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def neural_collapse_metrics(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n, d = features.shape
    c = num_classes
    
    means = []
    for k in range(c):
        mask = jnp.expand_dims(labels == float(k), -1)
        count = jnp.sum(mask)
        mean_k = jnp.sum(features * mask, axis=0) / jnp.maximum(count, 1.0)
        means.append(mean_k)
    class_means = jnp.stack(means, axis=0)  # [C, D]
    
    # NC1
    assigned_means = class_means[labels.astype(int)]
    diff = features - assigned_means
    nc1_var = jnp.sum(diff * diff) / float(n)
    
    # NC2 Simplex ETF
    global_mean = jnp.mean(class_means, axis=0, keepdims=True)
    centered = class_means - global_mean
    norms = jnp.sqrt(jnp.sum(centered * centered, axis=-1, keepdims=True) + 1e-8)
    norm_means = centered / norms
    
    cos_mat = norm_means @ norm_means.T
    ideal_cos = -1.0 / float(c - 1)
    
    off_diag_mask = 1.0 - jnp.eye(c)
    err = (cos_mat - ideal_cos) * off_diag_mask
    nc2_etf_err = jnp.sum(err * err) / float(c * (c - 1))
    
    return nc1_var, nc2_etf_err


def superposition_interference(w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    gram = w.T @ w
    m = gram.shape[0]
    off_diag = gram * (1.0 - jnp.eye(m))
    interf = jnp.sum(off_diag * off_diag)
    return gram, interf


def weightwatcher_alpha_hill(
    eigenvalues: jnp.ndarray,
    lambda_min: float = 0.5,
) -> jnp.ndarray:
    mask = eigenvalues >= lambda_min
    tail = eigenvalues * mask
    k_tail = jnp.sum(mask)
    log_ratios = jnp.log(jnp.maximum(tail, lambda_min) / lambda_min) * mask
    alpha = 1.0 + k_tail / (jnp.sum(log_ratios) + 1e-8)
    return alpha


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Neural Collapse
    nc1, nc2 = neural_collapse_metrics(g["nc_feats"], g["nc_labels"], num_classes=3)
    assert float(jnp.max(jnp.abs(nc1 - g["nc1_ref"]))) < 1e-5, "NC1 mismatch"
    assert float(jnp.max(jnp.abs(nc2 - g["nc2_ref"]))) < 1e-5, "NC2 mismatch"

    # Superposition
    sp_gram, sp_interf = superposition_interference(g["sp_w"])
    assert float(jnp.max(jnp.abs(sp_gram - g["sp_gram_ref"]))) < 1e-5, "Gram mismatch"
    assert float(jnp.max(jnp.abs(sp_interf - g["sp_interf_ref"]))) < 1e-5, "Interference mismatch"

    # WeightWatcher
    ww_alpha = weightwatcher_alpha_hill(g["ww_evals"], lambda_min=0.5)
    assert float(jnp.max(jnp.abs(ww_alpha - g["ww_alpha_ref"]))) < 1e-5, "WeightWatcher alpha mismatch"

    print("10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
