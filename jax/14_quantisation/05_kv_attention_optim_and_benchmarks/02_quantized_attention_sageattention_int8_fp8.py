"""Quantized Attention: SageAttention INT8/FP8 — functional JAX implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.
Run ``python jax/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.py``
to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def key_mean_smoothing(k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Key-mean smoothing: subtract sequence mean from Key matrix."""
    k_mean = jnp.mean(k, axis=0, keepdims=True)
    k_smoothed = k - k_mean
    return k_smoothed, k_mean


def quantize_symmetric_int8(
    x: jnp.ndarray,
    eps: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Symmetric per-tensor INT8 quantization."""
    amax = jnp.max(jnp.abs(x))
    scale = jnp.clip(amax / 127.0, a_min=eps)
    x_int8 = jnp.clip(jnp.round(x / scale), -127.0, 127.0)
    x_deq = x_int8 * scale
    return x_deq, x_int8, jnp.reshape(scale, (1,))


def sage_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """SageAttention forward pass: Key-mean smoothing -> INT8 QK^T -> Softmax -> INT8/FP8 PV."""
    d_k = q.shape[1]
    k_smoothed, k_mean = key_mean_smoothing(k)

    q_deq, q_int8, s_q = quantize_symmetric_int8(q)
    k_deq, k_int8, s_k = quantize_symmetric_int8(k_smoothed)

    int_acc = q_int8 @ k_int8.T
    scores = int_acc * ((s_q * s_k) / math.sqrt(float(d_k)))

    probs = jax.nn.softmax(scores, axis=-1)

    v_deq, v_int8, s_v = quantize_symmetric_int8(v)
    out = probs @ v_deq

    return out, scores, probs, int_acc, k_smoothed


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor parity
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Key-mean smoothing parity
    k_smoothed, k_mean = key_mean_smoothing(g["k"])
    err = float(jnp.max(jnp.abs(k_smoothed - g["k_smoothed"])))
    assert err < 1e-5, f"k_smoothed error {err:.2e}"
    err = float(jnp.max(jnp.abs(k_mean - g["k_mean"])))
    assert err < 1e-5, f"k_mean error {err:.2e}"

    # 3. INT8 quantization parity
    q_deq, q_int8, s_q = quantize_symmetric_int8(g["q"])
    err = float(jnp.max(jnp.abs(q_deq - g["q_deq"])))
    assert err < 1e-5, f"q_deq error {err:.2e}"
    err = float(jnp.max(jnp.abs(q_int8 - g["q_int8"])))
    assert err < 1e-5, f"q_int8 error {err:.2e}"
    err = float(jnp.max(jnp.abs(s_q - g["s_q"])))
    assert err < 1e-5, f"s_q error {err:.2e}"

    k_deq, k_int8, s_k = quantize_symmetric_int8(g["k_smoothed"])
    err = float(jnp.max(jnp.abs(k_deq - g["k_deq"])))
    assert err < 1e-5, f"k_deq error {err:.2e}"
    err = float(jnp.max(jnp.abs(k_int8 - g["k_int8"])))
    assert err < 1e-5, f"k_int8 error {err:.2e}"
    err = float(jnp.max(jnp.abs(s_k - g["s_k"])))
    assert err < 1e-5, f"s_k error {err:.2e}"

    v_deq, v_int8, s_v = quantize_symmetric_int8(g["v"])
    err = float(jnp.max(jnp.abs(v_deq - g["v_deq"])))
    assert err < 1e-5, f"v_deq error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_int8 - g["v_int8"])))
    assert err < 1e-5, f"v_int8 error {err:.2e}"
    err = float(jnp.max(jnp.abs(s_v - g["s_v"])))
    assert err < 1e-5, f"s_v error {err:.2e}"

    # 4. SageAttention forward pass parity
    out_sage, scores_sage, probs_sage, int_acc, _ = sage_attention(g["q"], g["k"], g["v"])
    err = float(jnp.max(jnp.abs(int_acc - g["int_acc"])))
    assert err < 1e-5, f"int_acc error {err:.2e}"
    err = float(jnp.max(jnp.abs(scores_sage - g["scores_sage"])))
    assert err < 1e-5, f"scores_sage error {err:.2e}"
    err = float(jnp.max(jnp.abs(probs_sage - g["probs_sage"])))
    assert err < 1e-5, f"probs_sage error {err:.2e}"
    err = float(jnp.max(jnp.abs(out_sage - g["out_sage"])))
    assert err < 1e-5, f"out_sage error {err:.2e}"

    # 5. Softmax invariance theorem verification
    d_k = g["q"].shape[1]
    scores_fp32_smooth = (g["q"] @ g["k_smoothed"].T) / math.sqrt(float(d_k))
    probs_fp32_smooth = jax.nn.softmax(scores_fp32_smooth, axis=-1)
    err = float(jnp.max(jnp.abs(probs_fp32_smooth - g["probs_fp32_smooth"])))
    assert err < 1e-5, f"probs_fp32_smooth error {err:.2e}"

    scores_fp32_raw = (g["q"] @ g["k"].T) / math.sqrt(float(d_k))
    probs_fp32_raw = jax.nn.softmax(scores_fp32_raw, axis=-1)
    err = float(jnp.max(jnp.abs(probs_fp32_raw - g["probs_fp32_raw"])))
    assert err < 1e-5, f"probs_fp32_raw error {err:.2e}"

    # Mathematically exact equivalence of smoothed vs raw softmax probabilities
    err = float(jnp.max(jnp.abs(probs_fp32_smooth - probs_fp32_raw)))
    assert err < 1e-5, f"Softmax invariance violation: {err:.2e}"

    print("14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
