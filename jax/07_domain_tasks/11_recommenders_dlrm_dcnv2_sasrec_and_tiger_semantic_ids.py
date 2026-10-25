"""Recommenders Dlrm Dcnv2 Sasrec and Tiger Semantic Ids — JAX reference implementation.

Topic: 07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids
Golden weights: weights/07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dlrm_feature_interaction(v_dense: jnp.ndarray, v_sparse: jnp.ndarray) -> jnp.ndarray:
    """DLRM Explicit Pairwise Dot-Product Feature Interaction in JAX."""
    s = v_sparse.shape[1]
    v_all = jnp.concatenate([jnp.expand_dims(v_dense, axis=1), v_sparse], axis=1)
    interaction_mat = jnp.einsum("bik, bjk -> bij", v_all, v_all)
    tril_idx = jnp.tril_indices(s + 1, -1)
    interactions = interaction_mat[:, tril_idx[0], tril_idx[1]]
    return jnp.concatenate([v_dense, interactions], axis=1)


def dcn_v2_cross_layer(
    x0: jnp.ndarray,
    xl: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """DCN-v2 (Deep & Cross Network v2) Cross Layer in JAX."""
    proj = xl @ weight + bias
    return x0 * proj + xl


def sasrec_causal_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """SASRec Causally-Masked Self-Attention in JAX."""
    t = q.shape[1]
    d = q.shape[-1]
    scores = jnp.einsum("bid, bjd -> bij", q, k) / jnp.sqrt(d)
    mask = jnp.triu(jnp.ones((t, t)), k=1)
    scores = jnp.where(mask[None, :, :], -1e9, scores)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bij, bjd -> bid", weights, v)


def rq_vae_quantize_step(
    residual: jnp.ndarray,
    codebook: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """TIGER / RQ-VAE Residual Vector Quantization Step in JAX."""
    r_sq = jnp.sum(residual ** 2, axis=-1, keepdims=True)
    c_sq = jnp.sum(codebook ** 2, axis=-1, keepdims=True).T
    dist = r_sq - 2.0 * (residual @ codebook.T) + c_sq
    indices = jnp.argmin(dist, axis=-1)
    quantized = codebook[indices]
    next_residual = residual - quantized
    return quantized, indices, next_residual


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    raw = load_file(str(path))
    tensors = {k: jnp.array(v) for k, v in raw.items()}

    # Anchor check
    out = linear_anchor(tensors["test_input"], tensors["weight"], tensors["bias"])
    err = float(jnp.max(jnp.abs(out - tensors["test_output"])))
    assert err < 1e-5, f"test_output parity error: {err}"

    # 1. DLRM
    dlrm = dlrm_feature_interaction(tensors["v_dense"], tensors["v_sparse"])
    err_dlrm = float(jnp.max(jnp.abs(dlrm - tensors["dlrm_ref"])))
    assert err_dlrm < 1e-5, f"dlrm parity error: {err_dlrm}"

    # 2. DCN-v2
    dcn = dcn_v2_cross_layer(tensors["x0"], tensors["xl"], tensors["w_cross"], tensors["b_cross"])
    err_dcn = float(jnp.max(jnp.abs(dcn - tensors["dcn_ref"])))
    assert err_dcn < 1e-5, f"dcn parity error: {err_dcn}"

    # 3. SASRec
    sasrec = sasrec_causal_attention(tensors["q"], tensors["k"], tensors["v"])
    err_sasrec = float(jnp.max(jnp.abs(sasrec - tensors["sasrec_ref"])))
    assert err_sasrec < 1e-5, f"sasrec parity error: {err_sasrec}"

    # 4. RQ-VAE
    quantized, indices, next_res = rq_vae_quantize_step(tensors["residual"], tensors["codebook"])
    err_quant = float(jnp.max(jnp.abs(quantized - tensors["quantized_ref"])))
    err_idx = int(jnp.max(jnp.abs(indices - tensors["indices_ref"])))
    err_res = float(jnp.max(jnp.abs(next_res - tensors["next_res_ref"])))
    assert err_quant < 1e-5, f"rq-vae quantized parity error: {err_quant}"
    assert err_idx == 0, f"rq-vae indices parity error: {err_idx}"
    assert err_res < 1e-5, f"rq-vae next_res parity error: {err_res}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
