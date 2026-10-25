"""HAWQ Mixed Precision Bit Allocation and Numeric Debugging — PyTorch 2.x reference implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging
Golden weights: weights/14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging.safetensors

``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def quantize_weight_bits(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-row uniform quantization to specified integer bitwidth.

    Args:
        w: Weight tensor [rows, cols].
        bits: Bitwidth in {2, 4, 8}.

    Returns:
        w_hat: Dequantized reconstructed weight tensor [rows, cols].
    """
    levels = float((2 ** (bits - 1)) - 1)
    scale = torch.amax(torch.abs(w), dim=-1, keepdim=True) / levels
    q = torch.clamp(torch.round(w / (scale + 1e-8)), -levels, levels)
    return q * scale


def compute_layer_sensitivity(
    w: torch.Tensor, trace: float, candidate_bits: list[int] = [2, 4, 8]
) -> dict[int, float]:
    """Computes second-order perturbation sensitivity Omega(b) = 0.5 * Tr(H) * ||W - W_b||^2."""
    sensitivities: dict[int, float] = {}
    for b in candidate_bits:
        w_q = quantize_weight_bits(w, b)
        err = float(torch.sum((w - w_q) ** 2))
        sensitivities[b] = 0.5 * trace * err
    return sensitivities


def greedy_hawq_bit_allocation(
    layer_weights: list[torch.Tensor],
    layer_traces: list[float],
    target_avg_bits: float = 4.0,
    candidate_bits: list[int] = [2, 4, 8],
) -> list[int]:
    """Greedy Pareto / ILP bit allocation under total parameter bit budget."""
    numels = [w.numel() for w in layer_weights]
    total_params = sum(numels)
    max_total_bits = target_avg_bits * total_params

    # Precompute sensitivities
    sensitivities = [
        compute_layer_sensitivity(w, tr, candidate_bits)
        for w, tr in zip(layer_weights, layer_traces)
    ]

    # Initialize all layers with minimum bitwidth
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
    x: torch.Tensor, weights: list[torch.Tensor], bit_allocation: list[int]
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Runs dual FP32 and quantized forward passes and reports layerwise SQNR in dB."""
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

        sig = torch.mean(a_fp ** 2)
        noise = torch.mean((a_fp - a_quant) ** 2)
        sqnr = 10.0 * torch.log10(sig / (noise + 1e-12))
        sqnrs.append(sqnr)

    return torch.tensor(sqnrs), quant_weights, fp_activations, quant_activations


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    shapes = [(16, 16), (16, 24), (24, 24), (24, 16)]
    w0 = torch.randn(*shapes[0]) * 0.1
    w1 = torch.randn(*shapes[1]) * 0.1
    w2 = torch.randn(*shapes[2]) * 0.1
    w3 = torch.randn(*shapes[3]) * 0.1
    weights = [w0, w1, w2, w3]
    x = torch.randn(8, 16)

    traces_list = [12.5, 1.2, 0.8, 8.4]
    traces = torch.tensor(traces_list, dtype=torch.float32)

    # Allocations
    alloc_3_5 = greedy_hawq_bit_allocation(weights, traces_list, target_avg_bits=3.5)
    alloc_5_0 = greedy_hawq_bit_allocation(weights, traces_list, target_avg_bits=5.0)

    t_alloc_3_5 = torch.tensor(alloc_3_5, dtype=torch.float32)
    t_alloc_5_0 = torch.tensor(alloc_5_0, dtype=torch.float32)

    # Diagnostics under target 5.0
    sqnrs, q_weights, fp_acts, q_acts = layerwise_sqnr_diagnostic(x, weights, alloc_5_0)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "traces": traces,
        "alloc_target_3_5": t_alloc_3_5,
        "alloc_target_5_0": t_alloc_5_0,
        "w0_q": q_weights[0],
        "w1_q": q_weights[1],
        "w2_q": q_weights[2],
        "w3_q": q_weights[3],
        "a0": fp_acts[0],
        "a1": fp_acts[1],
        "a2": fp_acts[2],
        "a3": fp_acts[3],
        "a0_q": q_acts[0],
        "a1_q": q_acts[1],
        "a2_q": q_acts[2],
        "a3_q": q_acts[3],
        "sqnrs": sqnrs,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
