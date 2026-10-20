"""JAX / NumPy implementation for Category 05 Topic 04:
Soup of Experts, SMEAR, Rewarded Soups, BTX, and Sparse Upcycling
in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def gelu_np(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation function using math.erf."""
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def soup_of_experts_smear_np(
    x: np.ndarray,
    experts_w1: np.ndarray,
    experts_b1: np.ndarray,
    experts_w2: np.ndarray,
    experts_b2: np.ndarray,
    router_w: np.ndarray,
    router_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Soup of Experts (SMEAR) dynamic parameter merging forward pass."""
    b_sz = x.shape[0]
    num_experts = experts_w1.shape[0]

    # Pool sequence tokens to compute sequence-level routing
    x_pool = np.mean(x, axis=1)  # (B, D)
    routing_logits = np.matmul(x_pool, router_w) + router_b  # (B, E)
    routing_weights = softmax_np(routing_logits, axis=-1)  # (B, E)

    out = np.zeros_like(x)
    for b in range(b_sz):
        g = routing_weights[b]  # (E,)
        # Dynamically merge parameters in weight space
        merged_w1 = np.sum(g[:, None, None] * experts_w1, axis=0)
        merged_b1 = np.sum(g[:, None] * experts_b1, axis=0)
        merged_w2 = np.sum(g[:, None, None] * experts_w2, axis=0)
        merged_b2 = np.sum(g[:, None] * experts_b2, axis=0)

        # Single dense forward pass
        h = gelu_np(np.matmul(x[b], merged_w1) + merged_b1)
        out[b] = np.matmul(h, merged_w2) + merged_b2

    return out, routing_weights


def activation_moe_np(
    x: np.ndarray,
    experts_w1: np.ndarray,
    experts_b1: np.ndarray,
    experts_w2: np.ndarray,
    experts_b2: np.ndarray,
    router_w: np.ndarray,
    router_b: np.ndarray,
) -> np.ndarray:
    """Conventional Activation-Space MoE forward pass."""
    b_sz = x.shape[0]
    num_experts = experts_w1.shape[0]

    x_pool = np.mean(x, axis=1)
    routing_logits = np.matmul(x_pool, router_w) + router_b
    routing_weights = softmax_np(routing_logits, axis=-1)

    out = np.zeros_like(x)
    for b in range(b_sz):
        g = routing_weights[b]
        for e in range(num_experts):
            h_e = gelu_np(np.matmul(x[b], experts_w1[e]) + experts_b1[e])
            out_e = np.matmul(h_e, experts_w2[e]) + experts_b2[e]
            out[b] += g[e] * out_e

    return out


def evaluate_pareto_soup_np(
    x: np.ndarray,
    experts_w1: np.ndarray,
    experts_b1: np.ndarray,
    experts_w2: np.ndarray,
    experts_b2: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Evaluates static weight interpolation (Rewarded Pareto Soup) between expert 0 and 1."""
    w1 = alpha * experts_w1[0] + (1.0 - alpha) * experts_w1[1]
    b1 = alpha * experts_b1[0] + (1.0 - alpha) * experts_b1[1]
    w2 = alpha * experts_w2[0] + (1.0 - alpha) * experts_w2[1]
    b2 = alpha * experts_b2[0] + (1.0 - alpha) * experts_b2[1]

    h = gelu_np(np.matmul(x, w1) + b1)
    return np.matmul(h, w2) + b2


def verify_against_checkpoint(checkpoint_path: Path | str | None = None) -> None:
    """Verifies JAX / NumPy forward passes against golden safetensors checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors")

    weights = load_file(str(checkpoint_path))
    x = weights["input"]
    experts_w1 = weights["experts_w1"]
    experts_b1 = weights["experts_b1"]
    experts_w2 = weights["experts_w2"]
    experts_b2 = weights["experts_b2"]
    router_w = weights["router_weight"]
    router_b = weights["router_bias"]

    # 1. SMEAR Verification
    out_smear, _ = soup_of_experts_smear_np(
        x, experts_w1, experts_b1, experts_w2, experts_b2, router_w, router_b
    )
    err_smear = float(np.max(np.abs(out_smear - weights["smear_output"])))
    assert err_smear < 1e-5, f"SMEAR parity error {err_smear} >= 1e-5"

    # 2. Activation MoE Verification
    out_act = activation_moe_np(
        x, experts_w1, experts_b1, experts_w2, experts_b2, router_w, router_b
    )
    err_act = float(np.max(np.abs(out_act - weights["act_output"])))
    assert err_act < 1e-5, f"Activation MoE parity error {err_act} >= 1e-5"

    # 3. Pareto Soup Verification
    out_pareto = evaluate_pareto_soup_np(
        x, experts_w1, experts_b1, experts_w2, experts_b2, alpha=0.6
    )
    err_pareto = float(np.max(np.abs(out_pareto - weights["pareto_soup_output"])))
    assert err_pareto < 1e-5, f"Pareto Soup parity error {err_pareto} >= 1e-5"

    print("All Category 05 Topic 04 JAX / NumPy models verified against checkpoint with < 1e-5 max error!")


if __name__ == "__main__":
    verify_against_checkpoint()
