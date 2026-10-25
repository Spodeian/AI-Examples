"""Rtn vs GPTQ Hessian Error Compensation — functional JAX implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation. Run ``python jax/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def rtn_quantize(
    weight: jnp.ndarray,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> jnp.ndarray:
    """Standard Round-To-Nearest (RTN) per-column symmetric quantization."""
    scale = jnp.max(jnp.abs(weight), axis=0, keepdims=True) / qmax
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    return jnp.clip(jnp.round(weight / scale), qmin, qmax) * scale


def compute_damped_hessian_inv(
    x: jnp.ndarray,
    damp_factor: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes empirical activation Hessian H = 2 * X^T X / N and damped inverse."""
    n_samples = x.shape[0]
    d_in = x.shape[1]
    H = 2.0 * (x.T @ x) / float(n_samples)
    diag_mean = float(jnp.mean(jnp.diag(H)))
    damp = damp_factor * diag_mean if diag_mean > 0.0 else 1e-4
    H_damped = H + damp * jnp.eye(d_in, dtype=H.dtype)
    H_inv = jnp.linalg.inv(H_damped)
    return H, H_inv


def gptq_quantize(
    weight: jnp.ndarray,
    h_inv: jnp.ndarray,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> jnp.ndarray:
    """Second-order error compensation GPTQ column-by-column update."""
    w = weight
    d_in, _ = w.shape
    scale = jnp.max(jnp.abs(weight), axis=0, keepdims=True) / qmax
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)

    for c in range(d_in):
        w_c = w[c, :]
        s = scale[0, :]
        q_c = jnp.clip(jnp.round(w_c / s), qmin, qmax) * s
        err_c = w_c - q_c
        h_inv_cc = h_inv[c, c]
        if c + 1 < d_in:
            step = jnp.expand_dims(err_c, 0) / h_inv_cc
            delta = step * jnp.expand_dims(h_inv[c + 1 :, c], 1)
            w = w.at[c + 1 :, :].add(-delta)
        w = w.at[c, :].set(q_c)

    return w


def gptq_with_act_order(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    damp_factor: float = 0.01,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPTQ with act-order (descending Hessian diagonal activation sorting)."""
    n_samples = x.shape[0]
    H = 2.0 * (x.T @ x) / float(n_samples)
    diag_h = jnp.diag(H)
    perm = jnp.argsort(-diag_h)
    inv_perm = jnp.argsort(perm)

    x_perm = x[:, perm]
    w_perm = weight[perm, :]

    H_perm, H_inv_perm = compute_damped_hessian_inv(x_perm, damp_factor=damp_factor)
    w_perm_quant = gptq_quantize(w_perm, H_inv_perm, qmin=qmin, qmax=qmax)
    w_final = w_perm_quant[inv_perm, :]

    return w_final, perm.astype(jnp.float32), H_inv_perm


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Hessian and inverse
    H, H_inv = compute_damped_hessian_inv(g["x_calib"], damp_factor=0.01)
    err_h = float(jnp.max(jnp.abs(H - g["h_matrix"])))
    err_h_inv = float(jnp.max(jnp.abs(H_inv - g["h_inv"])))
    assert err_h < 1e-5, f"H error {err_h:.2e}"
    assert err_h_inv < 1e-5, f"H_inv error {err_h_inv:.2e}"

    # 3. RTN
    w_rtn = rtn_quantize(g["w_orig"], qmin=-8.0, qmax=7.0)
    err_rtn = float(jnp.max(jnp.abs(w_rtn - g["w_rtn"])))
    assert err_rtn < 1e-5, f"RTN error {err_rtn:.2e}"

    # 4. GPTQ
    w_gptq = gptq_quantize(g["w_orig"], g["h_inv"], qmin=-8.0, qmax=7.0)
    err_gptq = float(jnp.max(jnp.abs(w_gptq - g["w_gptq"])))
    assert err_gptq < 1e-5, f"GPTQ error {err_gptq:.2e}"

    # 5. GPTQ with Act-Order
    w_act_gptq, perm, _ = gptq_with_act_order(g["x_calib"], g["w_orig"], damp_factor=0.01, qmin=-8.0, qmax=7.0)
    err_act = float(jnp.max(jnp.abs(w_act_gptq - g["w_act_gptq"])))
    err_perm = float(jnp.max(jnp.abs(perm - g["perm_indices"])))
    assert err_act < 1e-5, f"Act-order GPTQ error {err_act:.2e}"
    assert err_perm < 1e-5, f"Perm indices error {err_perm:.2e}"

    print("14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
