"""Adaptive Compute Early Exit Token Merging and Frugalgpt Cascades — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades.
Run ``python jax/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def early_exit_predict(
    hidden: jnp.ndarray, head_weight: jnp.ndarray, head_bias: jnp.ndarray, threshold: float = 0.6
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    logits = hidden @ head_weight + head_bias
    probs = jax.nn.softmax(logits, axis=-1)
    confidence = jnp.max(probs, axis=-1)
    exit_mask = (confidence >= threshold).astype(jnp.float32)
    return probs, confidence, exit_mask


def tome_bipartite_merge(
    x: jnp.ndarray, r_merge: int = 2, token_sizes: jnp.ndarray | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch, n, dim = x.shape
    if token_sizes is None:
        token_sizes = jnp.ones((batch, n), dtype=jnp.float32)

    x_np = np.array(x, copy=True)
    sizes_np = np.array(token_sizes, copy=True)

    idx_a = np.arange(0, n, 2)
    idx_b = np.arange(1, n, 2)
    a = x_np[:, idx_a, :]
    b = x_np[:, idx_b, :]
    sizes_a = sizes_np[:, idx_a]
    sizes_b = sizes_np[:, idx_b]

    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    sim = np.matmul(a_norm, np.swapaxes(b_norm, -1, -2))

    best_sim = np.max(sim, axis=-1)
    best_b = np.argmax(sim, axis=-1)

    top_a_sub = np.argsort(-best_sim, axis=-1)[:, :r_merge]

    x_out_list = []
    sizes_out_list = []
    for b_idx in range(batch):
        cur_b = np.copy(b[b_idx])
        cur_sizes_b = np.copy(sizes_b[b_idx])
        a_unmerged = []
        sizes_a_unmerged = []

        merged_a_set = set(top_a_sub[b_idx])
        for a_idx in range(len(idx_a)):
            if a_idx in merged_a_set:
                target_b = best_b[b_idx, a_idx]
                s_a = sizes_a[b_idx, a_idx]
                s_b = cur_sizes_b[target_b]
                cur_b[target_b] = (s_b * cur_b[target_b] + s_a * a[b_idx, a_idx]) / (s_a + s_b)
                cur_sizes_b[target_b] = s_a + s_b
            else:
                a_unmerged.append(a[b_idx, a_idx])
                sizes_a_unmerged.append(sizes_a[b_idx, a_idx])

        if a_unmerged:
            combined_x = np.concatenate([cur_b, np.array(a_unmerged)], axis=0)
            combined_sizes = np.concatenate([cur_sizes_b, np.array(sizes_a_unmerged)], axis=0)
        else:
            combined_x = cur_b
            combined_sizes = cur_sizes_b

        x_out_list.append(combined_x)
        sizes_out_list.append(combined_sizes)

    return jnp.asarray(x_out_list, dtype=jnp.float32), jnp.asarray(sizes_out_list, dtype=jnp.float32)


def frugal_cascade_route(
    confidences: jnp.ndarray,
    costs: jnp.ndarray,
    thresholds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch, num_models = confidences.shape
    conf_np = np.array(confidences)
    costs_np = np.array(costs)
    thresh_np = np.array(thresholds)

    routed_idx = np.full(batch, num_models - 1, dtype=np.float32)
    query_cost = np.zeros(batch, dtype=np.float32)
    resolved = np.zeros(batch, dtype=bool)

    for m in range(num_models):
        active = ~resolved
        query_cost[active] += costs_np[m]
        if m < num_models - 1:
            accepted = active & (conf_np[:, m] >= thresh_np[m])
            routed_idx[accepted] = float(m)
            resolved[accepted] = True
        else:
            routed_idx[active] = float(m)
            resolved[active] = True

    return jnp.asarray(routed_idx), jnp.asarray(query_cost)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Early-Exit
    probs, conf, exit_mask = early_exit_predict(g["hidden"], g["head_w"], g["head_b"], threshold=0.6)
    assert float(jnp.max(jnp.abs(probs - g["probs"]))) < 1e-5
    assert float(jnp.max(jnp.abs(conf - g["conf"]))) < 1e-5
    assert float(jnp.max(jnp.abs(exit_mask - g["exit_mask"]))) < 1e-5

    # 3. ToMe
    x_merged, merged_sizes = tome_bipartite_merge(g["x_tokens"], r_merge=2, token_sizes=g["init_sizes"])
    assert float(jnp.max(jnp.abs(x_merged - g["x_merged"]))) < 1e-5
    assert float(jnp.max(jnp.abs(merged_sizes - g["merged_sizes"]))) < 1e-5

    # 4. FrugalGPT
    routed_idx, total_cost = frugal_cascade_route(g["confidences"], g["costs"], g["thresholds"])
    assert float(jnp.max(jnp.abs(routed_idx - g["routed_idx"]))) < 1e-5
    assert float(jnp.max(jnp.abs(total_cost - g["total_cost"]))) < 1e-5

    print("15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
