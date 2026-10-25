"""HAWQ Mixed Precision Bit Allocation and Numeric Debugging — functional JAX implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging. Run ``python jax/14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def quantize_weight_bits(w: jnp.ndarray, bits: int) -> jnp.ndarray:
    """Symmetric per-row uniform quantization in JAX."""
    levels = float((2 ** (bits - 1)) - 1)
    scale = jnp.max(jnp.abs(w), axis=-1, keepdims=True) / levels
    q = jnp.clip(jnp.round(w / (scale + 1e-8)), -levels, levels)
    return q * scale


def compute_layer_sensitivity(
    w: jnp.ndarray, trace: float, candidate_bits: list[int] = [2, 4, 8]
) -> dict[int, float]:
    """Computes second-order perturbation sensitivity Omega(b) in JAX."""
    sensitivities: dict[int, float] = {}
    for b in candidate_bits:
        w_q = quantize_weight_bits(w, b)
        err = float(jnp.sum((w - w_q) ** 2))
        sensitivities[b] = 0.5 * trace * err
    return sensitivities


def greedy_hawq_bit_allocation(
    layer_weights: list[jnp.ndarray],
    layer_traces: list[float],
    target_avg_bits: float = 4.0,
    candidate_bits: list[int] = [2, 4, 8],
) -> list[int]:
    """Greedy Pareto / ILP bit allocation under parameter bit budget in JAX."""
    numels = [w.size for w in layer_weights]
    total_params = sum(numels)
    max_total_bits = target_avg_bits * total_params

    sensitivities = [
        compute_layer_sensitivity(w, tr, candidate_bits)
        for w, tr in zip(layer_weights, layer_traces)
    ]

    alloc = [min(candidate_bits) for _ in layer_weights]

    def current_total_bits(curr_alloc: list[int]) -> float:
        return sum(b * n for b, n in zip(curr_alloc, numels))

    while True:
        best_layer = None
        best_gain_ratio = -1.0
        best_next_bit = None

        for l in range(len(alloc)):
            cur_b = alloc[l]
            idx = candidate_bits.index(cur_b)
            if idx + 1 < len(candidate_bits):
                next_b = candidate_bits[idx + 1]
                delta_cost = (next_b - cur_b) * numels[l]
                if current_total_bits(alloc) + delta_cost <= max_total_bits:
                    gain = sensitivities[l][cur_b] - sensitivities[l][next_b]
                    ratio = gain / delta_cost
                    if ratio > best_gain_ratio:
                        best_gain_ratio = ratio
                        best_layer = l
                        best_next_bit = next_b

        if best_layer is None:
            break
        alloc[best_layer] = best_next_bit

    return alloc


def layerwise_sqnr_diagnostic(
    x: jnp.ndarray, weights: list[jnp.ndarray], bit_allocation: list[int]
) -> tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray]]:
    """Runs dual FP32 and quantized forward passes and reports layerwise SQNR in JAX."""
    a_fp = x
    a_quant = x
    fp_activations = []
    quant_activations = []
    quant_weights = []
    sqnrs = []

    for w, b in zip(weights, bit_allocation):
        w_q = quantize_weight_bits(w, b)
        quant_weights.append(w_q)

        a_fp = a_fp @ w
        a_quant = a_quant @ w_q

        fp_activations.append(a_fp)
        quant_activations.append(a_quant)

        sig = jnp.mean(a_fp ** 2)
        noise = jnp.mean((a_fp - a_quant) ** 2)
        sqnr = 10.0 * jnp.log10(sig / (noise + 1e-12))
        sqnrs.append(sqnr)

    return jnp.array(sqnrs), quant_weights, fp_activations, quant_activations


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Weights
    weights = [g["w0"], g["w1"], g["w2"], g["w3"]]
    traces = [float(t) for t in g["traces"]]

    # Bit allocation parity
    alloc_3_5 = greedy_hawq_bit_allocation(weights, traces, target_avg_bits=3.5)
    assert jnp.allclose(jnp.array(alloc_3_5, dtype=jnp.float32), g["alloc_target_3_5"])

    alloc_5_0 = greedy_hawq_bit_allocation(weights, traces, target_avg_bits=5.0)
    assert jnp.allclose(jnp.array(alloc_5_0, dtype=jnp.float32), g["alloc_target_5_0"])

    # Diagnostics under target 5.0
    sqnrs, q_weights, fp_acts, q_acts = layerwise_sqnr_diagnostic(g["x"], weights, alloc_5_0)

    for i in range(4):
        assert float(jnp.max(jnp.abs(q_weights[i] - g[f"w{i}_q"]))) < 1e-5
        assert float(jnp.max(jnp.abs(fp_acts[i] - g[f"a{i}"]))) < 1e-5
        assert float(jnp.max(jnp.abs(q_acts[i] - g[f"a{i}_q"]))) < 1e-5

    assert float(jnp.max(jnp.abs(sqnrs - g["sqnrs"]))) < 1e-4

    print("14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
