"""AI Brain Surgery Knowledge Editing ROME MEMIT and Stitching — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching. Run ``python jax/10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def rome_rank_one_edit(
    weight: jnp.ndarray,
    key_star: jnp.ndarray,
    val_star: jnp.ndarray,
    cov_inv: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    v_curr = weight @ key_star
    residual = val_star - v_curr
    u = cov_inv @ key_star
    denom = jnp.sum(key_star * u) + 1e-8
    delta_w = (jnp.expand_dims(residual, 1) @ jnp.expand_dims(u, 0)) / denom
    w_new = weight + delta_w
    return delta_w, w_new


def memit_mass_edit(
    weight: jnp.ndarray,
    keys: jnp.ndarray,
    vals: jnp.ndarray,
    cov_inv: jnp.ndarray,
    reg: float = 1e-4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    residuals = vals - weight @ keys
    u_mat = cov_inv @ keys
    m_dim = keys.shape[1]
    gram = keys.T @ u_mat + reg * jnp.eye(m_dim)
    coeff = jnp.linalg.solve(gram.T, residuals.T).T
    delta_w = coeff @ u_mat.T
    w_new = weight + delta_w
    return delta_w, w_new


def model_stitching_layer(
    acts_a: jnp.ndarray,
    acts_b: jnp.ndarray,
    reg: float = 1e-4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_a = acts_a.shape[1]
    lhs = acts_a.T @ acts_a + reg * jnp.eye(d_a)
    rhs = acts_a.T @ acts_b
    s_matrix = jnp.linalg.solve(lhs, rhs)
    b_hat = acts_a @ s_matrix
    return s_matrix, b_hat


def neuron_knockout(
    weight: jnp.ndarray,
    knockout_mask: jnp.ndarray,
) -> jnp.ndarray:
    keep_mask = 1.0 - jnp.expand_dims(knockout_mask, 0)
    return weight * keep_mask


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # ROME
    rome_delta, rome_wnew = rome_rank_one_edit(g["rome_w"], g["rome_k_star"], g["rome_v_star"], g["rome_cov_inv"])
    assert float(jnp.max(jnp.abs(rome_delta - g["rome_delta_ref"]))) < 1e-5, "ROME delta mismatch"
    assert float(jnp.max(jnp.abs(rome_wnew - g["rome_wnew_ref"]))) < 1e-5, "ROME w_new mismatch"

    # MEMIT
    memit_delta, memit_wnew = memit_mass_edit(g["rome_w"], g["memit_keys"], g["memit_vals"], g["rome_cov_inv"], reg=1e-4)
    assert float(jnp.max(jnp.abs(memit_delta - g["memit_delta_ref"]))) < 1e-5, "MEMIT delta mismatch"
    assert float(jnp.max(jnp.abs(memit_wnew - g["memit_wnew_ref"]))) < 1e-5, "MEMIT w_new mismatch"

    # Model Stitching
    stitch_s, stitch_bhat = model_stitching_layer(g["stitch_a"], g["stitch_b"], reg=1e-4)
    assert float(jnp.max(jnp.abs(stitch_s - g["stitch_s_ref"]))) < 1e-5, "Stitching matrix mismatch"
    assert float(jnp.max(jnp.abs(stitch_bhat - g["stitch_bhat_ref"]))) < 1e-5, "Stitched activations mismatch"

    # Neuron Knockout
    ko_pruned = neuron_knockout(g["rome_w"], g["ko_mask"])
    assert float(jnp.max(jnp.abs(ko_pruned - g["ko_pruned_ref"]))) < 1e-5, "Neuron knockout mismatch"

    print("10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
