"""Binary Sub1bit and Ternary BNN XNOR BiLLM OneBit TWN TTQ — functional JAX implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq. Run ``python jax/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def bnn_xnor_quantize(w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Binary Neural Network (BNN / XNOR-Net) 1-bit quantization."""
    signs = jnp.where(w >= 0.0, 1.0, -1.0)
    alpha = jnp.mean(jnp.abs(w), axis=0, keepdims=True)
    w_bnn = signs * alpha
    return w_bnn, signs, alpha


def twn_quantize(w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Ternary Weight Networks (TWN) 1.58-bit {-1, 0, +1} quantization."""
    delta = 0.7 * jnp.mean(jnp.abs(w), axis=0, keepdims=True)
    t = jnp.where(w > delta, 1.0, jnp.where(w < -delta, -1.0, 0.0))
    nonzeros = (jnp.abs(t) > 0.0).astype(jnp.float32)
    nz_counts = jnp.sum(nonzeros, axis=0, keepdims=True)
    nz_counts = jnp.clip(nz_counts, a_min=1.0)
    alpha = jnp.sum(jnp.abs(w) * nonzeros, axis=0, keepdims=True) / nz_counts
    w_twn = t * alpha
    return w_twn, t, alpha


def ttq_quantize(w: jnp.ndarray, delta_factor: float = 0.35) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Trained Ternary Quantization (TTQ) with asymmetric positive and negative scales."""
    delta = delta_factor * jnp.max(jnp.abs(w), axis=0, keepdims=True)
    pos_mask = (w > delta).astype(jnp.float32)
    neg_mask = (w < -delta).astype(jnp.float32)
    t = pos_mask - neg_mask

    pos_count = jnp.clip(jnp.sum(pos_mask, axis=0, keepdims=True), a_min=1.0)
    neg_count = jnp.clip(jnp.sum(neg_mask, axis=0, keepdims=True), a_min=1.0)

    w_pos = jnp.sum(w * pos_mask, axis=0, keepdims=True) / pos_count
    w_neg = jnp.sum(jnp.abs(w) * neg_mask, axis=0, keepdims=True) / neg_count

    w_ttq = pos_mask * w_pos - neg_mask * w_neg
    return w_ttq, t, w_pos, w_neg


def onebit_quantize(w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""OneBit: 1-bit value-sign rank-1 scaling decomposition: W \approx S \odot (g h^T)."""
    signs = jnp.where(w >= 0.0, 1.0, -1.0)
    abs_w = jnp.abs(w)
    g = jnp.mean(abs_w, axis=1, keepdims=True)
    total_abs = jnp.clip(jnp.sum(abs_w), a_min=1e-5)
    h = (jnp.sum(abs_w, axis=0, keepdims=True) / total_abs) * float(w.shape[0])
    scale_matrix = g @ h
    w_onebit = signs * scale_matrix
    return w_onebit, signs, g, h


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor parity
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. BNN parity
    w_bnn, b_signs, b_alpha = bnn_xnor_quantize(g["w_orig"])
    err_bnn = float(jnp.max(jnp.abs(w_bnn - g["w_bnn"])))
    err_b_signs = float(jnp.max(jnp.abs(b_signs - g["b_signs"])))
    err_b_alpha = float(jnp.max(jnp.abs(b_alpha - g["b_alpha"])))
    assert err_bnn < 1e-5, f"BNN error {err_bnn:.2e}"
    assert err_b_signs < 1e-5, f"BNN signs error {err_b_signs:.2e}"
    assert err_b_alpha < 1e-5, f"BNN alpha error {err_b_alpha:.2e}"

    # 3. TWN parity
    w_twn, t_ternary, t_alpha = twn_quantize(g["w_orig"])
    err_twn = float(jnp.max(jnp.abs(w_twn - g["w_twn"])))
    err_t_ternary = float(jnp.max(jnp.abs(t_ternary - g["t_ternary"])))
    err_t_alpha = float(jnp.max(jnp.abs(t_alpha - g["t_alpha"])))
    assert err_twn < 1e-5, f"TWN error {err_twn:.2e}"
    assert err_t_ternary < 1e-5, f"TWN ternary error {err_t_ternary:.2e}"
    assert err_t_alpha < 1e-5, f"TWN alpha error {err_t_alpha:.2e}"

    # 4. TTQ parity
    w_ttq, ttq_t, w_pos, w_neg = ttq_quantize(g["w_orig"])
    err_ttq = float(jnp.max(jnp.abs(w_ttq - g["w_ttq"])))
    err_ttq_t = float(jnp.max(jnp.abs(ttq_t - g["ttq_t"])))
    err_w_pos = float(jnp.max(jnp.abs(w_pos - g["w_pos"])))
    err_w_neg = float(jnp.max(jnp.abs(w_neg - g["w_neg"])))
    assert err_ttq < 1e-5, f"TTQ error {err_ttq:.2e}"
    assert err_ttq_t < 1e-5, f"TTQ t error {err_ttq_t:.2e}"
    assert err_w_pos < 1e-5, f"TTQ w_pos error {err_w_pos:.2e}"
    assert err_w_neg < 1e-5, f"TTQ w_neg error {err_w_neg:.2e}"

    # 5. OneBit parity
    w_onebit, onebit_signs, onebit_g, onebit_h = onebit_quantize(g["w_orig"])
    err_onebit = float(jnp.max(jnp.abs(w_onebit - g["w_onebit"])))
    err_onebit_signs = float(jnp.max(jnp.abs(onebit_signs - g["onebit_signs"])))
    err_onebit_g = float(jnp.max(jnp.abs(onebit_g - g["onebit_g"])))
    err_onebit_h = float(jnp.max(jnp.abs(onebit_h - g["onebit_h"])))
    assert err_onebit < 1e-5, f"OneBit error {err_onebit:.2e}"
    assert err_onebit_signs < 1e-5, f"OneBit signs error {err_onebit_signs:.2e}"
    assert err_onebit_g < 1e-5, f"OneBit g error {err_onebit_g:.2e}"
    assert err_onebit_h < 1e-5, f"OneBit h error {err_onebit_h:.2e}"

    print("14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
