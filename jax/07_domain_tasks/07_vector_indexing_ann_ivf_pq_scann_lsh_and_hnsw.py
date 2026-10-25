"""Vector Indexing Ann IVF PQ Scann Lsh and HNSW — functional JAX implementation.

Topic: 07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw. Run ``python jax/07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def pq_distance_table(
    query_subvectors: jnp.ndarray,
    codebooks: jnp.ndarray,
) -> jnp.ndarray:
    diff = jnp.expand_dims(query_subvectors, 2) - jnp.expand_dims(codebooks, 0)
    return jnp.sum(diff**2, axis=-1)


def pq_asymmetric_distance(
    dist_table: jnp.ndarray,
    codes: jnp.ndarray,
) -> jnp.ndarray:
    codes_int = codes.astype(jnp.int32)
    b, n_docs, m = codes.shape
    out = jnp.zeros((b, n_docs))
    for mi in range(m):
        c_m = codes_int[:, :, mi]
        t_m = dist_table[:, mi, :]
        d_m = jnp.take_along_axis(t_m, c_m, axis=1)
        out = out + d_m
    return out


def scann_anisotropic_loss(
    x: jnp.ndarray,
    x_tilde: jnp.ndarray,
    omega: float = 2.0,
    eps: float = 1e-6,
) -> jnp.ndarray:
    e = x - x_tilde
    x_norm_sq = jnp.sum(x**2, axis=-1, keepdims=True) + eps
    dot_ex = jnp.sum(e * x, axis=-1, keepdims=True)
    e_parallel = (dot_ex / x_norm_sq) * x
    e_perp = e - e_parallel
    loss_perp = jnp.sum(e_perp**2, axis=-1)
    loss_parallel = jnp.sum(e_parallel**2, axis=-1)
    return loss_perp + (1.0 + omega) * loss_parallel


def lsh_hash_hamming_distance(
    q: jnp.ndarray,
    d: jnp.ndarray,
    hyperplanes: jnp.ndarray,
) -> jnp.ndarray:
    bits_q = jnp.where(q @ hyperplanes >= 0, 1.0, -1.0)
    bits_d = jnp.where(d @ hyperplanes >= 0, 1.0, -1.0)
    return jnp.mean(bits_q * bits_d, axis=-1)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Product Quantization
    table_calc = pq_distance_table(g["query_subvectors"], g["codebooks"])
    err_table = float(jnp.max(jnp.abs(table_calc - g["dist_table"])))
    assert err_table < 1e-5, f"dist table error {err_table:.2e}"

    adc_calc = pq_asymmetric_distance(g["dist_table"], g["codes"])
    err_adc = float(jnp.max(jnp.abs(adc_calc - g["adc_distances"])))
    assert err_adc < 1e-5, f"adc distance error {err_adc:.2e}"

    # 2. ScaNN Loss
    scann_calc = scann_anisotropic_loss(g["x"], g["x_tilde"], omega=2.0)
    err_scann = float(jnp.max(jnp.abs(scann_calc - g["scann_loss"])))
    assert err_scann < 1e-5, f"scann loss error {err_scann:.2e}"

    # 3. LSH
    lsh_calc = lsh_hash_hamming_distance(g["q"], g["d"], g["hyperplanes"])
    err_lsh = float(jnp.max(jnp.abs(lsh_calc - g["lsh_sim"])))
    assert err_lsh < 1e-5, f"lsh sim error {err_lsh:.2e}"

    print("07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
