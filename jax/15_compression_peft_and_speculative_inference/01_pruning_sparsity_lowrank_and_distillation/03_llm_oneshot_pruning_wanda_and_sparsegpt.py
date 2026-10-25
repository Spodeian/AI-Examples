"""LLM Oneshot Pruning Wanda and Sparsegpt — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt.
Run ``python jax/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def compute_activation_norms(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(x, axis=0)


def wanda_prune(
    w: jnp.ndarray, act_norms: jnp.ndarray, sparsity: float = 0.5
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_out, d_in = w.shape
    scores = jnp.abs(w) * act_norms[None, :]
    k = max(1, int(sparsity * d_in))

    # Row-wise k-th smallest threshold
    # Sort scores per row
    sorted_scores = jnp.sort(scores, axis=-1)
    tau = sorted_scores[:, k - 1 : k]
    mask = (scores >= tau).astype(jnp.float32)
    w_pruned = w * mask
    return w_pruned, mask


def sparsegpt_prune(
    w: jnp.ndarray, h_inv: jnp.ndarray, sparsity: float = 0.5
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_out, d_in = w.shape
    w_work = np.array(w, copy=True)
    diag = np.diag(h_inv)

    col_scores = np.linalg.norm(w_work, axis=0) / np.sqrt(np.maximum(diag, 1e-8))
    k = max(1, int(sparsity * d_in))
    prune_cols = np.sort(np.argsort(col_scores)[:k])

    mask = np.ones((d_out, d_in), dtype=np.float32)
    for j in prune_cols:
        col_w = np.copy(w_work[:, j])
        denom = float(h_inv[j, j])
        if denom < 1e-8:
            denom = 1e-8
        w_work[:, j] = 0.0
        mask[:, j] = 0.0
        if j + 1 < d_in:
            h_slice = np.array(h_inv[j, (j + 1) :])
            w_work[:, (j + 1) :] -= np.outer(col_w, h_slice) / denom

    return jnp.asarray(w_work), jnp.asarray(mask)


def compute_reconstruction_error(
    x: jnp.ndarray, w_orig: jnp.ndarray, w_pruned: jnp.ndarray
) -> float:
    y_orig = x @ w_orig.T
    y_pruned = x @ w_pruned.T
    num = jnp.sum((y_orig - y_pruned) ** 2)
    den = jnp.sum(y_orig**2) + 1e-8
    return float(num / den)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Activation norms
    act_norms = compute_activation_norms(g["x"])
    assert float(jnp.max(jnp.abs(act_norms - g["act_norms"]))) < 1e-5

    # 3. Wanda pruning
    w_wanda, mask_wanda = wanda_prune(g["w"], act_norms, sparsity=0.5)
    assert float(jnp.max(jnp.abs(w_wanda - g["w_wanda"]))) < 1e-5
    assert float(jnp.max(jnp.abs(mask_wanda - g["mask_wanda"]))) < 1e-5

    # 4. SparseGPT pruning
    w_sgpt, mask_sgpt = sparsegpt_prune(g["w"], g["h_inv"], sparsity=0.5)
    assert float(jnp.max(jnp.abs(w_sgpt - g["w_sparsegpt"]))) < 1e-5
    assert float(jnp.max(jnp.abs(mask_sgpt - g["mask_sparsegpt"]))) < 1e-5

    # 5. Error metrics
    err_w = compute_reconstruction_error(g["x"], g["w"], w_wanda)
    err_s = compute_reconstruction_error(g["x"], g["w"], w_sgpt)
    err_vec = jnp.array([err_w, err_s], dtype=jnp.float32)
    assert float(jnp.max(jnp.abs(err_vec - g["err_metrics"]))) < 1e-5

    print("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
