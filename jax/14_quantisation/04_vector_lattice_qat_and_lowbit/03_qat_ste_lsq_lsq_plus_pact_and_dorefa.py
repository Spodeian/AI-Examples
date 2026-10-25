"""QAT STE LSQ LSQ+ PACT and DoReFa — functional JAX implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa. Run ``python jax/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ste_quantize(
    w: jnp.ndarray,
    scale: jnp.ndarray,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Straight-Through Estimator (STE) uniform quantization."""
    q = jnp.clip(jnp.round(w / scale), qmin, qmax)
    w_deq = q * scale
    ste_mask = ((w >= qmin * scale) & (w <= qmax * scale)).astype(jnp.float32)
    return w_deq, ste_mask


def pact_activation(
    x: jnp.ndarray,
    alpha: jnp.ndarray,
    nbits: int = 4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Parameterized Activation Clipping (PACT) with learnable clipping threshold alpha."""
    qmax = float(2 ** nbits - 1)
    x_clipped = jnp.clip(x, 0.0, float(alpha))
    step = alpha / qmax
    q = jnp.round(x_clipped / step)
    x_deq = q * step
    grad_alpha_mask = (x >= alpha).astype(jnp.float32)
    return x_deq, grad_alpha_mask


def lsq_quantize(
    w: jnp.ndarray,
    scale: jnp.ndarray,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Learned Step Size Quantization (LSQ) with analytical gradient scaling."""
    v = w / scale
    q = jnp.clip(jnp.round(v), qmin, qmax)
    w_deq = q * scale

    d_w_bar_ds = jnp.where(
        v < qmin,
        qmin,
        jnp.where(
            v > qmax,
            qmax,
            jnp.round(v) - v,
        ),
    )
    g = 1.0 / (float(w.size * qmax) ** 0.5)
    return w_deq, d_w_bar_ds * g


def lsq_plus_quantize(
    w: jnp.ndarray,
    scale: jnp.ndarray,
    beta: jnp.ndarray,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> jnp.ndarray:
    """LSQ+ quantization with learnable scale and asymmetric offset beta."""
    v = (w - beta) / scale
    q = jnp.clip(jnp.round(v), qmin, qmax)
    return q * scale + beta


def qat_forward(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    scale_w: jnp.ndarray,
    alpha_x: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """QAT end-to-end forward pass combining PACT activations and LSQ weights."""
    x_q, _ = pact_activation(x, alpha_x, nbits=4)
    w_q, _ = lsq_quantize(weight, scale_w, qmin=-8.0, qmax=7.0)
    y = x_q @ w_q
    if bias is not None:
        y = y + bias
    return y, x_q, w_q


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor parity
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. STE parity
    w_ste_deq, ste_mask = ste_quantize(g["w_orig"], g["scale_w"])
    err_ste_deq = float(jnp.max(jnp.abs(w_ste_deq - g["w_ste_deq"])))
    err_ste_mask = float(jnp.max(jnp.abs(ste_mask - g["ste_mask"])))
    assert err_ste_deq < 1e-5, f"STE dequant error {err_ste_deq:.2e}"
    assert err_ste_mask < 1e-5, f"STE mask error {err_ste_mask:.2e}"

    # 3. PACT parity
    x_pact_deq, pact_grad_mask = pact_activation(g["x_calib"], g["alpha_x"], nbits=4)
    err_pact_deq = float(jnp.max(jnp.abs(x_pact_deq - g["x_pact_deq"])))
    err_pact_mask = float(jnp.max(jnp.abs(pact_grad_mask - g["pact_grad_mask"])))
    assert err_pact_deq < 1e-5, f"PACT dequant error {err_pact_deq:.2e}"
    assert err_pact_mask < 1e-5, f"PACT mask error {err_pact_mask:.2e}"

    # 4. LSQ parity
    w_lsq_deq, lsq_grad_scale = lsq_quantize(g["w_orig"], g["scale_w"])
    err_lsq_deq = float(jnp.max(jnp.abs(w_lsq_deq - g["w_lsq_deq"])))
    err_lsq_grad = float(jnp.max(jnp.abs(lsq_grad_scale - g["lsq_grad_scale"])))
    assert err_lsq_deq < 1e-5, f"LSQ dequant error {err_lsq_deq:.2e}"
    assert err_lsq_grad < 1e-5, f"LSQ grad scale error {err_lsq_grad:.2e}"

    # 5. LSQ+ parity
    w_lsq_plus = lsq_plus_quantize(g["w_orig"], g["scale_w"], g["beta_w"])
    err_lsq_plus = float(jnp.max(jnp.abs(w_lsq_plus - g["w_lsq_plus"])))
    assert err_lsq_plus < 1e-5, f"LSQ+ error {err_lsq_plus:.2e}"

    # 6. QAT forward pass parity
    out_qat, _, _ = qat_forward(g["x_calib"], g["w_orig"], g["scale_w"], g["alpha_x"], g["b_orig"])
    err_qat = float(jnp.max(jnp.abs(out_qat - g["out_qat"])))
    assert err_qat < 1e-5, f"QAT output error {err_qat:.2e}"

    print("14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
