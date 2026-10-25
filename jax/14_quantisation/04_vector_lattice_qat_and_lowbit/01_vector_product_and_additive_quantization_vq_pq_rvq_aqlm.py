"""Vector Product and Additive Quantization VQ PQ RVQ AQLM — functional JAX implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm. Run ``python jax/14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def vector_quantize(vectors: jnp.ndarray, codebook: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-codebook Vector Quantization (VQ): nearest Euclidean centroid."""
    dist = (
        jnp.sum(vectors ** 2, axis=1, keepdims=True)
        - 2.0 * (vectors @ codebook.T)
        + jnp.sum(codebook ** 2, axis=1, keepdims=True).T
    )
    indices = jnp.argmin(dist, axis=1)
    recon = codebook[indices]
    return recon, indices


def product_quantize(vectors: jnp.ndarray, codebooks: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Product Quantization (PQ): splits D-dim vector into M subvectors."""
    n, d = vectors.shape
    m, k, d_sub = codebooks.shape
    assert d == m * d_sub, f"Dimension mismatch: {d} != {m} * {d_sub}"

    recon_subs = []
    indices_list = []
    for i in range(m):
        sub_vec = vectors[:, i * d_sub : (i + 1) * d_sub]
        sub_cb = codebooks[i]
        sub_recon, sub_idx = vector_quantize(sub_vec, sub_cb)
        recon_subs.append(sub_recon)
        indices_list.append(jnp.expand_dims(sub_idx, axis=1))

    recon = jnp.concatenate(recon_subs, axis=1)
    indices = jnp.concatenate(indices_list, axis=1)
    return recon, indices


def residual_vector_quantize(vectors: jnp.ndarray, codebooks: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Residual Vector Quantization (RVQ): sequential multi-stage codebook refinement."""
    residual = vectors
    recon = jnp.zeros_like(vectors)
    indices_list = []

    m = codebooks.shape[0]
    for i in range(m):
        cb = codebooks[i]
        stage_recon, stage_idx = vector_quantize(residual, cb)
        recon = recon + stage_recon
        residual = residual - stage_recon
        indices_list.append(jnp.expand_dims(stage_idx, axis=1))

    indices = jnp.concatenate(indices_list, axis=1)
    return recon, indices


def aqlm_dequantize(indices: jnp.ndarray, codebooks: jnp.ndarray) -> jnp.ndarray:
    """Additive Multi-Codebook Quantization (AQLM) dequantization."""
    n, m = indices.shape
    _, _, d = codebooks.shape
    recon = jnp.zeros((n, d), dtype=codebooks.dtype)
    int_indices = indices.astype(jnp.int32)
    for i in range(m):
        idx = int_indices[:, i]
        recon = recon + codebooks[i][idx]
    return recon


def aqlm_forward(
    x: jnp.ndarray,
    indices: jnp.ndarray,
    codebooks: jnp.ndarray,
    d_in: int,
    d_out: int,
    bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """AQLM forward pass: reconstructs [d_in, d_out] weight matrix and computes Y = X @ W + bias."""
    w_recon_vecs = aqlm_dequantize(indices, codebooks)
    w_deq = w_recon_vecs.reshape((d_in, d_out))
    y = x @ w_deq
    if bias is not None:
        y = y + bias
    return y, w_deq


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor parity
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. VQ parity
    vq_recon, vq_idx = vector_quantize(g["w_vecs"], g["vq_codebook"])
    err_vq_recon = float(jnp.max(jnp.abs(vq_recon - g["vq_recon"])))
    err_vq_idx = float(jnp.max(jnp.abs(vq_idx.astype(jnp.float32) - g["vq_indices"])))
    assert err_vq_recon < 1e-5, f"VQ recon error {err_vq_recon:.2e}"
    assert err_vq_idx < 1e-5, f"VQ indices error {err_vq_idx:.2e}"

    # 3. PQ parity
    pq_recon, pq_idx = product_quantize(g["w_vecs"], g["pq_codebooks"])
    err_pq_recon = float(jnp.max(jnp.abs(pq_recon - g["pq_recon"])))
    err_pq_idx = float(jnp.max(jnp.abs(pq_idx.astype(jnp.float32) - g["pq_indices"])))
    assert err_pq_recon < 1e-5, f"PQ recon error {err_pq_recon:.2e}"
    assert err_pq_idx < 1e-5, f"PQ indices error {err_pq_idx:.2e}"

    # 4. RVQ parity
    rvq_recon, rvq_idx = residual_vector_quantize(g["w_vecs"], g["rvq_codebooks"])
    err_rvq_recon = float(jnp.max(jnp.abs(rvq_recon - g["rvq_recon"])))
    err_rvq_idx = float(jnp.max(jnp.abs(rvq_idx.astype(jnp.float32) - g["rvq_indices"])))
    assert err_rvq_recon < 1e-5, f"RVQ recon error {err_rvq_recon:.2e}"
    assert err_rvq_idx < 1e-5, f"RVQ indices error {err_rvq_idx:.2e}"

    # 5. AQLM forward parity
    out_aqlm, w_aqlm_deq = aqlm_forward(
        g["x_calib"], g["aqlm_indices"], g["aqlm_codebooks"], d_in=16, d_out=8, bias=g["b_orig"]
    )
    err_aqlm_w = float(jnp.max(jnp.abs(w_aqlm_deq - g["w_aqlm_deq"])))
    err_aqlm_out = float(jnp.max(jnp.abs(out_aqlm - g["out_aqlm"])))
    assert err_aqlm_w < 1e-5, f"AQLM dequant error {err_aqlm_w:.2e}"
    assert err_aqlm_out < 1e-5, f"AQLM output error {err_aqlm_out:.2e}"

    print("14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
