"""AI Analysis Representation Similarity CKA SVCCA RSA and Intrinsic Dim — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim. Run ``python jax/10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def center_gram_matrix(gram: jnp.ndarray) -> jnp.ndarray:
    mean_row = jnp.mean(gram, axis=0, keepdims=True)
    mean_col = jnp.mean(gram, axis=1, keepdims=True)
    mean_total = jnp.mean(gram)
    return gram - mean_row - mean_col + mean_total


def linear_cka(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    k = x @ x.T
    l_mat = y @ y.T
    k_c = center_gram_matrix(k)
    l_c = center_gram_matrix(l_mat)
    hsic_xy = jnp.sum(k_c * l_c)
    hsic_xx = jnp.sum(k_c * k_c)
    hsic_yy = jnp.sum(l_c * l_c)
    denom = jnp.sqrt(hsic_xx * hsic_yy) + 1e-8
    return hsic_xy / denom


def rsa_correlation(rdm_a: jnp.ndarray, rdm_b: jnp.ndarray) -> jnp.ndarray:
    n = rdm_a.shape[0]
    idx_i, idx_j = jnp.triu_indices(n, k=1)
    vec_a = rdm_a[idx_i, idx_j]
    vec_b = rdm_b[idx_i, idx_j]
    va_c = vec_a - jnp.mean(vec_a)
    vb_c = vec_b - jnp.mean(vec_b)
    num = jnp.sum(va_c * vb_c)
    denom = jnp.sqrt(jnp.sum(va_c * va_c) * jnp.sum(vb_c * vb_c)) + 1e-8
    return num / denom


def twonn_intrinsic_dimension(r1: jnp.ndarray, r2: jnp.ndarray) -> jnp.ndarray:
    mu = r2 / (r1 + 1e-8)
    log_mu = jnp.log(mu)
    n = float(r1.shape[0])
    return n / (jnp.sum(log_mu) + 1e-8)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # CKA
    cka = linear_cka(g["cka_x"], g["cka_y"])
    assert float(jnp.max(jnp.abs(cka - g["cka_score_ref"]))) < 1e-5, "CKA mismatch"

    # RSA
    rsa = rsa_correlation(g["rdm_a"], g["rdm_b"])
    assert float(jnp.max(jnp.abs(rsa - g["rsa_corr_ref"]))) < 1e-5, "RSA correlation mismatch"

    # TwoNN
    twonn = twonn_intrinsic_dimension(g["twonn_r1"], g["twonn_r2"])
    assert float(jnp.max(jnp.abs(twonn - g["twonn_id_ref"]))) < 1e-5, "TwoNN mismatch"

    print("10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
