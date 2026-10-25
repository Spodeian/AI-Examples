"""Quantization Method Selection Matrix and Head to Head Benchmark — functional JAX implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark. Run ``python jax/14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark.safetensors"

NF4_LEVELS = jnp.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=jnp.float32)


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def rtn_quantize(w: jnp.ndarray, bits: int) -> jnp.ndarray:
    """Round-to-Nearest uniform symmetric per-row quantization in JAX."""
    levels = float((2 ** (bits - 1)) - 1)
    scale = jnp.max(jnp.abs(w), axis=-1, keepdims=True) / levels
    q = jnp.clip(jnp.round(w / (scale + 1e-8)), -levels, levels)
    return q * scale


def awq_quantize(w: jnp.ndarray, x: jnp.ndarray, bits: int = 4) -> jnp.ndarray:
    """Activation-Aware Weight Quantization in JAX."""
    levels = float((2 ** (bits - 1)) - 1)
    act_scale = jnp.mean(jnp.abs(x), axis=0, keepdims=True)
    s_awq = jnp.clip(act_scale ** 0.5, a_min=1e-3)
    w_scaled = w * s_awq
    s_w = jnp.max(jnp.abs(w_scaled), axis=-1, keepdims=True) / levels
    q = jnp.clip(jnp.round(w_scaled / (s_w + 1e-8)), -levels, levels)
    return (q * s_w) / s_awq


def bitnet_quantize(w: jnp.ndarray) -> jnp.ndarray:
    """BitNet b1.58 native ternary quantization in JAX."""
    scale = jnp.mean(jnp.abs(w))
    q = jnp.clip(jnp.round(w / (scale + 1e-8)), -1.0, 1.0)
    return q * scale


def nf4_quantize(w: jnp.ndarray) -> jnp.ndarray:
    """NormalFloat 4-bit (NF4) quantile codebook quantization in JAX."""
    scale = jnp.max(jnp.abs(w), axis=-1, keepdims=True)
    norm_w = w / (scale + 1e-8)
    diffs = jnp.abs(jnp.expand_dims(norm_w, -1) - NF4_LEVELS)
    idx = jnp.argmin(diffs, axis=-1)
    return NF4_LEVELS[idx] * scale


def compute_sqnr(signal: jnp.ndarray, noise: jnp.ndarray) -> float:
    sig_pwr = float(jnp.mean(signal ** 2))
    noise_pwr = float(jnp.mean(noise ** 2))
    return float(10.0 * jnp.log10(sig_pwr / (noise_pwr + 1e-12)))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Weights
    w = g["w"]
    x = g["x"]

    w_rtn8 = rtn_quantize(w, bits=8)
    assert float(jnp.max(jnp.abs(w_rtn8 - g["w_rtn8"]))) < 1e-5

    w_rtn4 = rtn_quantize(w, bits=4)
    assert float(jnp.max(jnp.abs(w_rtn4 - g["w_rtn4"]))) < 1e-5

    w_awq4 = awq_quantize(w, x, bits=4)
    assert float(jnp.max(jnp.abs(w_awq4 - g["w_awq4"]))) < 1e-5

    w_bitnet = bitnet_quantize(w)
    assert float(jnp.max(jnp.abs(w_bitnet - g["w_bitnet"]))) < 1e-5

    w_nf4 = nf4_quantize(w)
    assert float(jnp.max(jnp.abs(w_nf4 - g["w_nf4"]))) < 1e-5

    # Activations
    y_true = x @ w
    assert float(jnp.max(jnp.abs(y_true - g["y_true"]))) < 1e-5

    y_rtn8 = x @ w_rtn8
    assert float(jnp.max(jnp.abs(y_rtn8 - g["y_rtn8"]))) < 1e-5

    y_rtn4 = x @ w_rtn4
    assert float(jnp.max(jnp.abs(y_rtn4 - g["y_rtn4"]))) < 1e-5

    y_awq4 = x @ w_awq4
    assert float(jnp.max(jnp.abs(y_awq4 - g["y_awq4"]))) < 1e-5

    y_bitnet = x @ w_bitnet
    assert float(jnp.max(jnp.abs(y_bitnet - g["y_bitnet"]))) < 1e-5

    y_nf4 = x @ w_nf4
    assert float(jnp.max(jnp.abs(y_nf4 - g["y_nf4"]))) < 1e-5

    # SQNR metrics
    methods = [
        (w_rtn8, y_rtn8),
        (w_rtn4, y_rtn4),
        (w_awq4, y_awq4),
        (w_bitnet, y_bitnet),
        (w_nf4, y_nf4),
    ]

    for i, (w_q, y_q) in enumerate(methods):
        w_sqnr = compute_sqnr(w, w - w_q)
        act_sqnr = compute_sqnr(y_true, y_true - y_q)
        assert abs(w_sqnr - float(g["weight_sqnrs"][i])) < 1e-3
        assert abs(act_sqnr - float(g["act_sqnrs"][i])) < 1e-3

    print("14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
