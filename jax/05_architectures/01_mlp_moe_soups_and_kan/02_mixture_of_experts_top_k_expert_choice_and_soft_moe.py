"""JAX / NumPy implementation for Category 05 Topic 02:
Mixture of Experts (Top-K, Expert Choice, and Soft MoE) in 05_architectures/01_mlp_moe_soups_and_kan.
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


def eval_expert_np(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Evaluates 2-layer MLP expert."""
    h = gelu_np(np.matmul(x, w1) + b1)
    return np.matmul(h, w2) + b2


def topk_moe_np(
    x: np.ndarray,
    router_w: np.ndarray,
    router_b: np.ndarray,
    experts_w1: np.ndarray,
    experts_b1: np.ndarray,
    experts_w2: np.ndarray,
    experts_b2: np.ndarray,
    top_k: int = 2,
) -> tuple[np.ndarray, float, float]:
    """Sparse Top-K Mixture of Experts with Aux Load-Balancing & Router Z-loss."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_flat = x.reshape(-1, d_model)
    num_tokens = x_flat.shape[0]
    num_experts = router_w.shape[1]

    # Router logits and probabilities: (T, E)
    router_logits = np.matmul(x_flat, router_w) + router_b
    router_probs = softmax_np(router_logits, axis=-1)

    # Top-k selection
    out_flat = np.zeros_like(x_flat)
    f_count = np.zeros(num_experts, dtype=np.float32)

    for t in range(num_tokens):
        logits_t = router_logits[t]
        # Top-k indices
        topk_idx = np.argsort(logits_t)[-top_k:][::-1]
        topk_vals = logits_t[topk_idx]
        topk_weights = softmax_np(topk_vals)

        for k_idx in range(top_k):
            e = int(topk_idx[k_idx])
            w = topk_weights[k_idx]
            f_count[e] += 1.0
            expert_out = eval_expert_np(
                x_flat[t : t + 1],
                experts_w1[e],
                experts_b1[e],
                experts_w2[e],
                experts_b2[e],
            ).squeeze(0)
            out_flat[t] += w * expert_out

    # Load balancing auxiliary loss: E * sum_i(f_i * P_i)
    f_i = f_count / num_tokens
    P_i = np.mean(router_probs, axis=0)
    aux_loss = float(num_experts * np.sum(f_i * P_i))

    # Router Z-loss: (1/T) sum_t (log sum_i exp(H_ti))^2
    lse = np.log(np.sum(np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True)), axis=-1)) + np.max(router_logits, axis=-1)
    z_loss = float(np.mean(lse**2))

    return out_flat.reshape(orig_shape), aux_loss, z_loss


def expert_choice_moe_np(
    x: np.ndarray,
    router_w: np.ndarray,
    router_b: np.ndarray,
    experts_w1: np.ndarray,
    experts_b1: np.ndarray,
    experts_w2: np.ndarray,
    experts_b2: np.ndarray,
    capacity: int = 2,
) -> np.ndarray:
    """Expert Choice Mixture of Experts with inverted routing."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_flat = x.reshape(-1, d_model)
    num_experts = router_w.shape[1]

    router_logits = np.matmul(x_flat, router_w) + router_b
    router_probs = softmax_np(router_logits, axis=-1)

    # Transposed routing matrix: (E, T)
    G = router_probs.T
    out_flat = np.zeros_like(x_flat)

    for e in range(num_experts):
        expert_scores = G[e]
        top_tokens = np.argsort(expert_scores)[-capacity:][::-1]
        for t in top_tokens:
            score = expert_scores[t]
            expert_out = eval_expert_np(
                x_flat[t : t + 1],
                experts_w1[e],
                experts_b1[e],
                experts_w2[e],
                experts_b2[e],
            ).squeeze(0)
            out_flat[t] += score * expert_out

    return out_flat.reshape(orig_shape)


def soft_moe_np(
    x: np.ndarray,
    soft_disp_w: np.ndarray,
    soft_disp_b: np.ndarray,
    soft_comb_w: np.ndarray,
    soft_comb_b: np.ndarray,
    experts_w1: np.ndarray,
    experts_b1: np.ndarray,
    experts_w2: np.ndarray,
    experts_b2: np.ndarray,
    slots_per_expert: int = 1,
) -> np.ndarray:
    """Soft Mixture of Experts with soft dispatch and combination."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_flat = x.reshape(-1, d_model)
    total_slots = soft_disp_w.shape[1]

    # Dispatch: softmax across tokens (axis=0)
    disp_logits = np.matmul(x_flat, soft_disp_w) + soft_disp_b
    Phi = softmax_np(disp_logits, axis=0)

    # Combine: softmax across slots (axis=1)
    comb_logits = np.matmul(x_flat, soft_comb_w) + soft_comb_b
    Psi = softmax_np(comb_logits, axis=1)

    # Slot inputs: (total_slots, d_model) = Phi^T @ x_flat
    slot_inputs = np.matmul(Phi.T, x_flat)

    slot_outputs = np.zeros_like(slot_inputs)
    for j in range(total_slots):
        e = j // slots_per_expert
        slot_outputs[j] = eval_expert_np(
            slot_inputs[j : j + 1],
            experts_w1[e],
            experts_b1[e],
            experts_w2[e],
            experts_b2[e],
        ).squeeze(0)

    # Combine back to tokens
    out_flat = np.matmul(Psi, slot_outputs)
    return out_flat.reshape(orig_shape)


def verify_against_checkpoint(checkpoint_path: Path | str | None = None) -> None:
    """Verifies JAX / NumPy forward passes against golden safetensors checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors")

    weights = load_file(str(checkpoint_path))
    x = weights["input"]
    router_w = weights["router_weight"]
    router_b = weights["router_bias"]
    experts_w1 = weights["experts_w1"]
    experts_b1 = weights["experts_b1"]
    experts_w2 = weights["experts_w2"]
    experts_b2 = weights["experts_b2"]
    soft_disp_w = weights["soft_disp_weight"]
    soft_disp_b = weights["soft_disp_bias"]
    soft_comb_w = weights["soft_comb_weight"]
    soft_comb_b = weights["soft_comb_bias"]

    # 1. Top-K MoE
    out_topk, aux_loss, z_loss = topk_moe_np(
        x, router_w, router_b, experts_w1, experts_b1, experts_w2, experts_b2, top_k=2
    )
    err_topk = float(np.max(np.abs(out_topk - weights["topk_output"])))
    err_aux = abs(aux_loss - float(weights["aux_loss"][0]))
    err_z = abs(z_loss - float(weights["z_loss"][0]))
    assert err_topk < 1e-5, f"JAX TopK MoE parity error {err_topk} >= 1e-5"
    assert err_aux < 1e-5, f"JAX Aux loss parity error {err_aux} >= 1e-5"
    assert err_z < 1e-5, f"JAX Z-loss parity error {err_z} >= 1e-5"

    # 2. Expert Choice MoE
    out_ec = expert_choice_moe_np(
        x, router_w, router_b, experts_w1, experts_b1, experts_w2, experts_b2, capacity=2
    )
    err_ec = float(np.max(np.abs(out_ec - weights["expert_choice_output"])))
    assert err_ec < 1e-5, f"JAX Expert Choice MoE parity error {err_ec} >= 1e-5"

    # 3. Soft MoE
    out_soft = soft_moe_np(
        x, soft_disp_w, soft_disp_b, soft_comb_w, soft_comb_b,
        experts_w1, experts_b1, experts_w2, experts_b2, slots_per_expert=1
    )
    err_soft = float(np.max(np.abs(out_soft - weights["soft_moe_output"])))
    assert err_soft < 1e-5, f"JAX Soft MoE parity error {err_soft} >= 1e-5"

    print("All Category 05 Topic 02 JAX / NumPy models verified against checkpoint with < 1e-5 max error!")


if __name__ == "__main__":
    verify_against_checkpoint()
