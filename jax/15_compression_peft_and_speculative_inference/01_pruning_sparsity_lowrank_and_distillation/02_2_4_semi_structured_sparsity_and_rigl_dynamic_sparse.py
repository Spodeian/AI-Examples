"""2 4 Semi Structured Sparsity and Rigl Dynamic Sparse — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse.
Run ``python jax/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def prune_2_to_4(w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    rows, cols = w.shape
    w_blocks = w.reshape(rows, cols // 4, 4)
    abs_blocks = jnp.abs(w_blocks)
    order = jnp.argsort(-abs_blocks, axis=-1)
    rank = jnp.argsort(order, axis=-1)
    mask_blocks = (rank < 2).astype(jnp.float32)
    mask = mask_blocks.reshape(rows, cols)
    w_sparse = w * mask
    return w_sparse, mask


def compress_2_to_4(w: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    rows, cols = w.shape
    num_blocks = cols // 4
    w_blocks = np.asarray(w).reshape(rows, num_blocks, 4)
    mask_blocks = np.asarray(mask).reshape(rows, num_blocks, 4)

    vals_list = []
    idxs_list = []
    for r in range(rows):
        row_vals = []
        row_idxs = []
        for b in range(num_blocks):
            m_block = mask_blocks[r, b]
            w_block = w_blocks[r, b]
            kept_indices = np.where(m_block > 0.5)[0]
            for idx in kept_indices:
                row_vals.append(w_block[idx])
                row_idxs.append(float(idx))
        vals_list.append(row_vals)
        idxs_list.append(row_idxs)

    values = jnp.asarray(vals_list, dtype=jnp.float32)
    indices = jnp.asarray(idxs_list, dtype=jnp.float32)
    return values, indices


def rigl_update_mask(
    w: jnp.ndarray,
    grad: jnp.ndarray,
    current_mask: jnp.ndarray,
    drop_fraction: float = 0.25,
) -> jnp.ndarray:
    flat_w = w.ravel()
    flat_grad = grad.ravel()
    flat_mask = current_mask.ravel()

    active_indices = jnp.where(flat_mask > 0.5)[0]
    inactive_indices = jnp.where(flat_mask <= 0.5)[0]
    n_active = len(active_indices)
    k_drop = max(1, int(drop_fraction * n_active))

    active_mags = jnp.abs(flat_w[active_indices])
    drop_order = jnp.argsort(active_mags)[:k_drop]
    to_drop = active_indices[drop_order]

    inactive_grad_mags = jnp.abs(flat_grad[inactive_indices])
    regrow_order = jnp.argsort(-inactive_grad_mags)[:k_drop]
    to_regrow = inactive_indices[regrow_order]

    new_mask = flat_mask.at[to_drop].set(0.0)
    new_mask = new_mask.at[to_regrow].set(1.0)
    return new_mask.reshape(current_mask.shape)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. 2:4 Pruning
    w_24, mask_24 = prune_2_to_4(g["w"])
    assert float(jnp.max(jnp.abs(w_24 - g["w_2_to_4"]))) < 1e-5
    assert float(jnp.max(jnp.abs(mask_24 - g["mask_2_to_4"]))) < 1e-5

    # 3. 2:4 Compression
    vals, idxs = compress_2_to_4(g["w"], mask_24)
    assert float(jnp.max(jnp.abs(vals - g["packed_values"]))) < 1e-5
    assert float(jnp.max(jnp.abs(idxs - g["packed_indices"]))) < 1e-5

    # 4. RigL dynamic update
    mask_s1 = rigl_update_mask(g["w"], g["grad"], g["initial_mask"], drop_fraction=0.25)
    assert float(jnp.max(jnp.abs(mask_s1 - g["rigl_mask_step1"]))) < 1e-5

    print("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
