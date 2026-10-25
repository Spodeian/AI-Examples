"""KV Cache Quantization: KIVI and KVQuant — functional JAX implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant.
Run ``python jax/14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant.py``
to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def quantize_k_per_channel(
    k: jnp.ndarray,
    bits: int = 2,
    eps: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """KIVI Key Quantization: per-channel asymmetric quantization along sequence dimension (axis=0)."""
    qmax = float((1 << bits) - 1)
    k_min = jnp.min(k, axis=0, keepdims=True)
    k_max = jnp.max(k, axis=0, keepdims=True)

    scale = jnp.clip((k_max - k_min) / qmax, a_min=eps)
    zero = jnp.round(jnp.clip(-k_min / scale, 0.0, qmax))

    k_int = jnp.clip(jnp.round(k / scale + zero), 0.0, qmax)
    k_deq = (k_int - zero) * scale
    return k_deq, k_int, scale, zero


def quantize_v_per_token(
    v: jnp.ndarray,
    bits: int = 2,
    eps: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """KIVI Value Quantization: per-token asymmetric quantization along channel dimension (axis=1)."""
    qmax = float((1 << bits) - 1)
    v_min = jnp.min(v, axis=1, keepdims=True)
    v_max = jnp.max(v, axis=1, keepdims=True)

    scale = jnp.clip((v_max - v_min) / qmax, a_min=eps)
    zero = jnp.round(jnp.clip(-v_min / scale, 0.0, qmax))

    v_int = jnp.clip(jnp.round(v / scale + zero), 0.0, qmax)
    v_deq = (v_int - zero) * scale
    return v_deq, v_int, scale, zero


def kivi_attention_decode(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    window_size: int = 4,
    bits: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """KIVI Decoding Attention: quantize past tokens while keeping recent window in FP32."""
    seq_len, d_k = k.shape
    if seq_len > window_size:
        k_past = k[: seq_len - window_size]
        v_past = v[: seq_len - window_size]
        k_past_deq, _, _, _ = quantize_k_per_channel(k_past, bits=bits)
        v_past_deq, _, _, _ = quantize_v_per_token(v_past, bits=bits)

        k_eff = jnp.concatenate([k_past_deq, k[seq_len - window_size :]], axis=0)
        v_eff = jnp.concatenate([v_past_deq, v[seq_len - window_size :]], axis=0)
    else:
        k_eff = k
        v_eff = v

    scores = (q @ k_eff.T) * (1.0 / math.sqrt(float(d_k)))
    probs = jax.nn.softmax(scores, axis=-1)
    out = probs @ v_eff
    return out, k_eff, v_eff


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor parity
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Key per-channel quantization
    k_deq, k_int, k_scale, k_zero = quantize_k_per_channel(g["k"], bits=2)
    err = float(jnp.max(jnp.abs(k_deq - g["k_deq"])))
    assert err < 1e-5, f"k_deq error {err:.2e}"
    err = float(jnp.max(jnp.abs(k_int - g["k_int"])))
    assert err < 1e-5, f"k_int error {err:.2e}"
    err = float(jnp.max(jnp.abs(k_scale - g["k_scale"])))
    assert err < 1e-5, f"k_scale error {err:.2e}"
    err = float(jnp.max(jnp.abs(k_zero - g["k_zero"])))
    assert err < 1e-5, f"k_zero error {err:.2e}"

    # 3. Value per-token quantization
    v_deq, v_int, v_scale, v_zero = quantize_v_per_token(g["v"], bits=2)
    err = float(jnp.max(jnp.abs(v_deq - g["v_deq"])))
    assert err < 1e-5, f"v_deq error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_int - g["v_int"])))
    assert err < 1e-5, f"v_int error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_scale - g["v_scale"])))
    assert err < 1e-5, f"v_scale error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_zero - g["v_zero"])))
    assert err < 1e-5, f"v_zero error {err:.2e}"

    # 4. KIVI Attention Decoding with sliding window
    out_kivi, k_eff, v_eff = kivi_attention_decode(g["q"], g["k"], g["v"], window_size=4, bits=2)
    err = float(jnp.max(jnp.abs(out_kivi - g["out_kivi"])))
    assert err < 1e-5, f"out_kivi error {err:.2e}"
    err = float(jnp.max(jnp.abs(k_eff - g["k_eff"])))
    assert err < 1e-5, f"k_eff error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_eff - g["v_eff"])))
    assert err < 1e-5, f"v_eff error {err:.2e}"

    # 5. Channel peaks
    channel_peaks = jnp.max(jnp.abs(g["k"]), axis=0)
    err = float(jnp.max(jnp.abs(channel_peaks - g["channel_peaks"])))
    assert err < 1e-5, f"channel_peaks error {err:.2e}"

    # 6. Unquantized baseline
    d_k = g["k"].shape[1]
    scores_fp32 = (g["q"] @ g["k"].T) * (1.0 / math.sqrt(float(d_k)))
    probs_fp32 = jax.nn.softmax(scores_fp32, axis=-1)
    out_fp32 = probs_fp32 @ g["v"]
    err = float(jnp.max(jnp.abs(out_fp32 - g["out_fp32"])))
    assert err < 1e-5, f"out_fp32 error {err:.2e}"

    print("14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
