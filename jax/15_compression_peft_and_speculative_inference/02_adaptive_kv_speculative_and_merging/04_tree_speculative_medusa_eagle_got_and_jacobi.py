"""Tree Speculative Medusa Eagle GoT and Jacobi — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi. Run ``python jax/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def medusa_heads_predict(
    hidden: jnp.ndarray,
    head_weights: jnp.ndarray,
    head_biases: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    num_heads, _, vocab = head_weights.shape

    logits_list = []
    top_tokens_list = []

    for k in range(num_heads):
        head_logits = hidden @ head_weights[k] + head_biases[k]
        top_tok = jnp.argmax(head_logits, axis=-1)
        logits_list.append(head_logits)
        top_tokens_list.append(top_tok)

    logits = jnp.stack(logits_list, axis=1)
    top_tokens = jnp.stack(top_tokens_list, axis=1)
    return logits, top_tokens


def build_tree_attention_mask(
    parent_indices: jnp.ndarray,
) -> jnp.ndarray:
    num_nodes = parent_indices.shape[0]
    mask = jnp.full((num_nodes, num_nodes), -1e9, dtype=jnp.float32)

    for i in range(num_nodes):
        curr = i
        visited = set()
        while curr not in visited:
            mask = mask.at[i, curr].set(0.0)
            visited.add(curr)
            if curr == 0:
                break
            curr = int(parent_indices[curr])

    return mask


def tree_longest_accepted_path(
    candidate_tokens: jnp.ndarray,
    target_tokens: jnp.ndarray,
    parent_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    num_nodes = candidate_tokens.shape[0]
    is_match = (candidate_tokens == target_tokens)

    accepted = [False] * num_nodes
    accepted[0] = True

    for i in range(1, num_nodes):
        parent = int(parent_indices[i])
        if accepted[parent] and bool(is_match[i]):
            accepted[i] = True

    best_len = 0
    best_node = 0
    depths = [0] * num_nodes
    for i in range(num_nodes):
        if accepted[i]:
            if i == 0:
                depths[i] = 1
            else:
                depths[i] = depths[int(parent_indices[i])] + 1
            if depths[i] > best_len:
                best_len = depths[i]
                best_node = i

    path_mask = jnp.zeros(num_nodes, dtype=jnp.float32)
    curr = best_node
    while True:
        path_mask = path_mask.at[curr].set(1.0)
        if curr == 0:
            break
        curr = int(parent_indices[curr])

    return path_mask, jnp.array(float(best_len), dtype=jnp.float32)


def jacobi_lookahead_step(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_next = jnp.tanh(x @ weight + bias)
    batch = x.shape[0]
    residual = jnp.max(jnp.abs(x_next - x).reshape((batch, -1)), axis=-1)
    return x_next, residual


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Medusa heads
    m_logits, t_tokens = medusa_heads_predict(g["hidden"], g["head_weights"], g["head_biases"])
    err = float(jnp.max(jnp.abs(m_logits - g["medusa_logits"])))
    assert err < 1e-5, f"medusa_logits error {err:.2e}"
    err = float(jnp.max(jnp.abs(t_tokens - g["top_tokens"])))
    assert err < 1e-5, f"top_tokens error {err:.2e}"

    # 3. Tree attention mask
    t_mask = build_tree_attention_mask(g["parent_indices"])
    err = float(jnp.max(jnp.abs(t_mask - g["tree_mask"])))
    assert err < 1e-5, f"tree_mask error {err:.2e}"

    # 4. Tree longest accepted path
    p_mask, p_len = tree_longest_accepted_path(g["candidate_tokens"], g["target_tokens"], g["parent_indices"])
    err = float(jnp.max(jnp.abs(p_mask - g["path_mask"])))
    assert err < 1e-5, f"path_mask error {err:.2e}"
    err = float(jnp.max(jnp.abs(p_len - g["path_length"])))
    assert err < 1e-5, f"path_length error {err:.2e}"

    # 5. Jacobi step
    x_nxt, j_res = jacobi_lookahead_step(g["x_jacobi"], g["w_jacobi"], g["b_jacobi"])
    err = float(jnp.max(jnp.abs(x_nxt - g["x_next"])))
    assert err < 1e-5, f"x_next error {err:.2e}"
    err = float(jnp.max(jnp.abs(j_res - g["jacobi_res"])))
    assert err < 1e-5, f"jacobi_res error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
