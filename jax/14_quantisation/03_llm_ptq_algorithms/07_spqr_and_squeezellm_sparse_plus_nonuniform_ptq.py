"""SpQR and Squeezellm Sparse Plus Nonuniform PTQ — functional JAX implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq. Run ``python jax/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def compute_hessian_sensitivity(
    x_calib: jnp.ndarray,
    weight: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_samples = x_calib.shape[0]
    h_diag = 2.0 * jnp.sum(x_calib ** 2, axis=0) / float(n_samples)
    sensitivity = jnp.expand_dims(h_diag, 1) * jnp.abs(weight)
    return h_diag, sensitivity


def extract_sparse_outliers(
    weight: jnp.ndarray,
    sensitivity: jnp.ndarray,
    top_k_count: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    flat_sens = sensitivity.flatten()
    sorted_sens = jnp.sort(flat_sens)
    threshold = sorted_sens[-top_k_count]
    sparse_mask = sensitivity >= threshold

    w_sparse = jnp.where(sparse_mask, weight, jnp.zeros_like(weight))
    w_dense = jnp.where(sparse_mask, jnp.zeros_like(weight), weight)
    return w_sparse, w_dense, sparse_mask.astype(jnp.float32)


def quantize_dense_int3(
    w_dense: jnp.ndarray,
    sparse_mask: jnp.ndarray,
    qmin: float = -4.0,
    qmax: float = 3.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = jnp.max(jnp.abs(w_dense), axis=0, keepdims=True) / qmax
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)

    q = jnp.clip(jnp.round(w_dense / scale), qmin, qmax)
    q = jnp.where(sparse_mask > 0.0, jnp.zeros_like(q), q)
    w_dense_deq = q * scale
    return w_dense_deq, scale


def spqr_forward(
    x: jnp.ndarray,
    w_dense_deq: jnp.ndarray,
    w_sparse: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    y_dense = x @ w_dense_deq
    y_sparse = x @ w_sparse
    y_total = y_dense + y_sparse
    if bias is not None:
        y_total = y_total + bias
    return y_total, y_dense, y_sparse


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Hessian and sensitivity
    h_diag, sensitivity = compute_hessian_sensitivity(g["x_calib"], g["w_orig"])
    assert float(jnp.max(jnp.abs(h_diag - g["h_diag"]))) < 1e-5
    assert float(jnp.max(jnp.abs(sensitivity - g["sensitivity"]))) < 1e-5

    # 3. Sparse extraction
    w_sparse, w_dense, sparse_mask = extract_sparse_outliers(g["w_orig"], sensitivity, top_k_count=8)
    assert float(jnp.max(jnp.abs(w_sparse - g["w_sparse"]))) < 1e-5
    assert float(jnp.max(jnp.abs(w_dense - g["w_dense"]))) < 1e-5
    assert float(jnp.max(jnp.abs(sparse_mask - g["sparse_mask"]))) < 1e-5

    # 4. Dense 3-bit quantization
    w_dense_deq, dense_scale = quantize_dense_int3(w_dense, sparse_mask, qmin=-4.0, qmax=3.0)
    assert float(jnp.max(jnp.abs(w_dense_deq - g["w_dense_deq"]))) < 1e-5
    assert float(jnp.max(jnp.abs(dense_scale - g["dense_scale"]))) < 1e-5

    # 5. SpQR forward
    out_spqr, y_dense, y_sparse = spqr_forward(g["x_calib"], w_dense_deq, w_sparse, g["b_orig"])
    assert float(jnp.max(jnp.abs(out_spqr - g["out_spqr"]))) < 1e-5
    assert float(jnp.max(jnp.abs(y_dense - g["y_dense"]))) < 1e-5
    assert float(jnp.max(jnp.abs(y_sparse - g["y_sparse"]))) < 1e-5

    print("14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
