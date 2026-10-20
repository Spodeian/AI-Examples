"""JAX / NumPy implementation for Category 05 Topic 03:
DeepSeekMoE, Auxiliary-Loss-Free Load Balancing, and Mixture of Depths (MoD)
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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


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


def deepseek_moe_np(
    x: np.ndarray,
    router_w: np.ndarray,
    router_b: np.ndarray,
    b_dyn: np.ndarray,
    shared_w1: np.ndarray,
    shared_b1: np.ndarray,
    shared_w2: np.ndarray,
    shared_b2: np.ndarray,
    routed_w1: np.ndarray,
    routed_b1: np.ndarray,
    routed_w2: np.ndarray,
    routed_b2: np.ndarray,
    top_k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """DeepSeekMoE forward pass with shared experts and aux-loss-free load balancing."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_flat = x.reshape(-1, d_model)
    num_tokens = x_flat.shape[0]
    num_shared = shared_w1.shape[0]
    num_routed = routed_w1.shape[0]

    out_flat = np.zeros_like(x_flat)

    # 1. Unconditionally evaluate shared experts
    for s in range(num_shared):
        out_flat += eval_expert_np(
            x_flat,
            shared_w1[s],
            shared_b1[s],
            shared_w2[s],
            shared_b2[s],
        )

    # 2. Router logits and aux-loss-free top-k selection
    router_logits = np.matmul(x_flat, router_w) + router_b  # (T, N_r)
    biased_logits = router_logits + b_dyn  # used ONLY for index selection

    expert_counts = np.zeros(num_routed, dtype=np.float32)

    for t in range(num_tokens):
        biased_t = biased_logits[t]
        raw_t = router_logits[t]

        topk_idx = np.argsort(biased_t)[-top_k:][::-1]
        # Invariant: weights use UNBIASED raw logits!
        raw_topk_vals = raw_t[topk_idx]
        weights_t = softmax_np(raw_topk_vals)

        for k_idx in range(top_k):
            e = int(topk_idx[k_idx])
            w = weights_t[k_idx]
            expert_counts[e] += 1.0
            expert_out = eval_expert_np(
                x_flat[t : t + 1],
                routed_w1[e],
                routed_b1[e],
                routed_w2[e],
                routed_b2[e],
            ).squeeze(0)
            out_flat[t] += w * expert_out

    return out_flat.reshape(orig_shape), expert_counts


def update_dynamic_bias_np(b_dyn: np.ndarray, counts: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """Updates dynamic bias outside autograd."""
    avg_count = np.mean(counts)
    return b_dyn + gamma * np.sign(avg_count - counts)


def mixture_of_depths_np(
    x: np.ndarray,
    mod_w: np.ndarray,
    mod_b: np.ndarray,
    block_w1: np.ndarray,
    block_b1: np.ndarray,
    block_w2: np.ndarray,
    block_b2: np.ndarray,
    capacity: int = 2,
) -> np.ndarray:
    """Mixture of Depths (MoD) forward pass."""
    b_sz, s_len, d_model = x.shape
    mod_logits = np.squeeze(np.matmul(x, mod_w) + mod_b, axis=-1)  # (B, S)
    mod_weights = sigmoid_np(mod_logits)  # (B, S)

    out = np.copy(x)
    for b in range(b_sz):
        row_weights = mod_weights[b]
        top_tokens = np.argsort(row_weights)[-capacity:][::-1]
        for s in top_tokens:
            token_in = x[b, s : s + 1]
            gate = row_weights[s]
            block_out = eval_expert_np(token_in, block_w1, block_b1, block_w2, block_b2)
            out[b, s] = token_in.squeeze(0) + gate * block_out.squeeze(0)

    return out


def verify_against_checkpoint(checkpoint_path: Path | str | None = None) -> None:
    """Verifies JAX / NumPy forward passes against golden safetensors checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors")

    weights = load_file(str(checkpoint_path))
    x = weights["input"]
    router_w = weights["router_weight"]
    router_b = weights["router_bias"]
    b_dyn = weights["b_dyn"]
    shared_w1 = weights["shared_w1"]
    shared_b1 = weights["shared_b1"]
    shared_w2 = weights["shared_w2"]
    shared_b2 = weights["shared_b2"]
    routed_w1 = weights["routed_w1"]
    routed_b1 = weights["routed_b1"]
    routed_w2 = weights["routed_w2"]
    routed_b2 = weights["routed_b2"]
    mod_w = weights["mod_weight"]
    mod_b = weights["mod_bias"]

    # 1. DeepSeekMoE Verification
    out_ds, counts = deepseek_moe_np(
        x, router_w, router_b, b_dyn,
        shared_w1, shared_b1, shared_w2, shared_b2,
        routed_w1, routed_b1, routed_w2, routed_b2,
        top_k=2,
    )
    err_ds = float(np.max(np.abs(out_ds - weights["deepseek_moe_output"])))
    assert err_ds < 1e-5, f"DeepSeekMoE output error {err_ds} >= 1e-5"

    b_updated = update_dynamic_bias_np(b_dyn, counts, gamma=0.1)
    err_b = float(np.max(np.abs(b_updated - weights["b_dyn_updated"])))
    assert err_b < 1e-5, f"Dynamic bias update error {err_b} >= 1e-5"

    # 2. Mixture of Depths Verification
    out_mod = mixture_of_depths_np(
        x, mod_w, mod_b,
        shared_w1[0], shared_b1[0], shared_w2[0], shared_b2[0],
        capacity=2,
    )
    err_mod = float(np.max(np.abs(out_mod - weights["mod_output"])))
    assert err_mod < 1e-5, f"Mixture of Depths error {err_mod} >= 1e-5"

    print("All Category 05 Topic 03 JAX / NumPy models verified against checkpoint with < 1e-5 max error!")


if __name__ == "__main__":
    verify_against_checkpoint()
