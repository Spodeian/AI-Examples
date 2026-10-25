"""PEFT Representation Finetuning Loreft — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft. Run ``python jax/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def loreft_subspace_delta(
    h: jnp.ndarray,
    r_matrix: jnp.ndarray,
    w_matrix: jnp.ndarray,
    b_vec: jnp.ndarray,
) -> jnp.ndarray:
    """Compute LoReFT representation delta: delta_h = R^T (W h + b - R h)."""
    proj_r = jnp.matmul(h, r_matrix.T)
    proj_w = jnp.matmul(h, w_matrix.T) + b_vec
    delta_s = proj_w - proj_r
    return jnp.matmul(delta_s, r_matrix)


def loreft_intervene(
    h: jnp.ndarray,
    r_matrix: jnp.ndarray,
    w_matrix: jnp.ndarray,
    b_vec: jnp.ndarray,
    position_mask: jnp.ndarray,
) -> jnp.ndarray:
    """LoReFT intervention at selective token positions: h_out = h + mask * delta_h."""
    delta_h = loreft_subspace_delta(h, r_matrix, w_matrix, b_vec)
    mask = jnp.expand_dims(position_mask, -1)
    return h + mask * delta_h


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Delta H
    delta_h = loreft_subspace_delta(g["h"], g["r_matrix"], g["w_matrix"], g["b_vec"])
    err = float(jnp.max(jnp.abs(delta_h - g["delta_h"])))
    assert err < 1e-5, f"delta_h error {err:.2e}"

    # 2. LoReFT Intervene
    loreft_out = loreft_intervene(g["h"], g["r_matrix"], g["w_matrix"], g["b_vec"], g["position_mask"])
    err = float(jnp.max(jnp.abs(loreft_out - g["loreft_out"])))
    assert err < 1e-5, f"loreft_out error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
