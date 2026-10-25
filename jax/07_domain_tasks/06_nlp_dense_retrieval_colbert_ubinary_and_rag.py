"""Nlp Dense Retrieval Colbert Ubinary and RAG — functional JAX implementation.

Topic: 07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag. Run ``python jax/07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def colbert_maxsim(
    query_embeds: jnp.ndarray,
    doc_embeds: jnp.ndarray,
) -> jnp.ndarray:
    sim_matrix = query_embeds @ jnp.swapaxes(doc_embeds, -2, -1)
    max_per_query = jnp.max(sim_matrix, axis=-1)
    return jnp.sum(max_per_query, axis=-1)


def ubinary_quantize_and_dot(
    q_float: jnp.ndarray,
    d_float: jnp.ndarray,
) -> jnp.ndarray:
    q_b = jnp.where(q_float >= 0, 1.0, -1.0)
    d_b = jnp.where(d_float >= 0, 1.0, -1.0)
    return jnp.sum(q_b * d_b, axis=-1)


def reciprocal_rank_fusion(
    ranks1: jnp.ndarray,
    ranks2: jnp.ndarray,
    k: float = 60.0,
) -> jnp.ndarray:
    return (1.0 / (k + ranks1)) + (1.0 / (k + ranks2))


def dense_infonce_loss(
    query_vecs: jnp.ndarray,
    pos_vecs: jnp.ndarray,
    tau: float = 0.05,
) -> jnp.ndarray:
    b = query_vecs.shape[0]
    logits = (query_vecs @ pos_vecs.T) / tau
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    targets = jnp.arange(b)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[:, None], axis=-1))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. ColBERT MaxSim
    maxsim_calc = colbert_maxsim(g["query_embeds"], g["doc_embeds"])
    err_maxsim = float(jnp.max(jnp.abs(maxsim_calc - g["maxsim_ref"])))
    assert err_maxsim < 1e-5, f"maxsim error {err_maxsim:.2e}"

    # 2. UBinary
    ub_calc = ubinary_quantize_and_dot(g["q_float"], g["d_float"])
    err_ub = float(jnp.max(jnp.abs(ub_calc - g["ubinary_ref"])))
    assert err_ub < 1e-5, f"ubinary error {err_ub:.2e}"

    # 3. RRF
    rrf_calc = reciprocal_rank_fusion(g["ranks1"], g["ranks2"], k=60.0)
    err_rrf = float(jnp.max(jnp.abs(rrf_calc - g["rrf_ref"])))
    assert err_rrf < 1e-5, f"rrf error {err_rrf:.2e}"

    # 4. Dense InfoNCE Loss
    loss_calc = dense_infonce_loss(g["q_vecs"], g["p_vecs"], tau=0.05)
    err_loss = float(jnp.max(jnp.abs(loss_calc - g["infonce_ref"])))
    assert err_loss < 1e-5, f"infonce loss error {err_loss:.2e}"

    print("07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
