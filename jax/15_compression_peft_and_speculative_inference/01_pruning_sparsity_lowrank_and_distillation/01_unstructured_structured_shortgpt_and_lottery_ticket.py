"""Unstructured Structured Shortgpt and Lottery Ticket — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket. Run ``python jax/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def unstructured_magnitude_prune(w: jnp.ndarray, sparsity: float = 0.5) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unstructured magnitude pruning in JAX."""
    k = max(1, int(sparsity * w.size))
    sorted_abs = jnp.sort(jnp.abs(w).flatten())
    tau = sorted_abs[k - 1]
    mask = (jnp.abs(w) >= tau).astype(jnp.float32)
    return w * mask, mask


def structured_row_prune(w: jnp.ndarray, keep_ratio: float = 0.75) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Structured row pruning based on L2 norm in JAX."""
    row_norms = jnp.linalg.norm(w, axis=1)
    k_rows = max(1, int(keep_ratio * w.shape[0]))
    indices = jnp.sort(jnp.argsort(row_norms)[-k_rows:])
    w_sliced = w[indices, :]
    return w_sliced, indices.astype(jnp.float32)


def compute_block_influence(x_in: jnp.ndarray, x_out: jnp.ndarray) -> float:
    """ShortGPT Block Influence in JAX."""
    dot = jnp.sum(x_in * x_out, axis=-1)
    norm_in = jnp.linalg.norm(x_in, axis=-1)
    norm_out = jnp.linalg.norm(x_out, axis=-1)
    cos = dot / (norm_in * norm_out + 1e-8)
    return float(1.0 - jnp.mean(cos))


def lottery_ticket_rewind(
    w_initial: jnp.ndarray, w_trained: jnp.ndarray, sparsity: float = 0.5
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Lottery ticket subnetwork extraction in JAX."""
    k = max(1, int(sparsity * w_trained.size))
    sorted_abs = jnp.sort(jnp.abs(w_trained).flatten())
    tau = sorted_abs[k - 1]
    mask = (jnp.abs(w_trained) >= tau).astype(jnp.float32)
    return w_initial * mask, mask


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Unstructured
    w_unstr, mask_unstr = unstructured_magnitude_prune(g["w"], sparsity=0.5)
    assert float(jnp.max(jnp.abs(w_unstr - g["w_unstructured"]))) < 1e-5
    assert float(jnp.max(jnp.abs(mask_unstr - g["mask_unstructured"]))) < 1e-5

    # 2. Structured
    w_str, indices_str = structured_row_prune(g["w"], keep_ratio=0.75)
    assert float(jnp.max(jnp.abs(w_str - g["w_structured"]))) < 1e-5
    assert float(jnp.max(jnp.abs(indices_str - g["indices_structured"]))) < 1e-5

    # 3. ShortGPT
    bi_red = compute_block_influence(g["x_in"], g["x_out_red"])
    bi_act = compute_block_influence(g["x_in"], g["x_out_act"])
    assert abs(bi_red - float(g["bi_scores"][0])) < 1e-5
    assert abs(bi_act - float(g["bi_scores"][1])) < 1e-5

    # 4. Lottery Ticket
    w_subnetwork, mask_lottery = lottery_ticket_rewind(g["w"], g["w_trained"], sparsity=0.5)
    assert float(jnp.max(jnp.abs(w_subnetwork - g["w_lottery_subnetwork"]))) < 1e-5
    assert float(jnp.max(jnp.abs(mask_lottery - g["mask_lottery"]))) < 1e-5

    print("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
