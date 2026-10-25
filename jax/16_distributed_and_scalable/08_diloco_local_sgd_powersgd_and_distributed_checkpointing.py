"""Diloco Local SGD Powersgd and Distributed Checkpointing — functional JAX implementation.

Topic: 16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing.
Run ``python jax/16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def diloco_outer_step(
    theta_0: jnp.ndarray,
    theta_workers: jnp.ndarray,
    v_outer: jnp.ndarray,
    lr_outer: float = 0.7,
    beta_outer: float = 0.9,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    deltas = jnp.expand_dims(theta_0, 0) - theta_workers
    delta_avg = jnp.mean(deltas, axis=0)
    v_new = beta_outer * v_outer + delta_avg
    theta_new = theta_0 - lr_outer * v_new
    return theta_new, v_new, delta_avg


def powersgd_compress_decompress(
    grad_matrix: jnp.ndarray,
    probe_q: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    p = grad_matrix @ probe_q
    norm_p = jnp.linalg.norm(p) + 1e-8
    p_norm = p / norm_p
    q_new = grad_matrix.T @ p_norm
    recon_grad = p_norm @ q_new.T
    return recon_grad, p_norm, q_new


def distributed_checkpoint_reshard(
    shards_2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    full_tensor = jnp.concatenate([shards_2[0], shards_2[1]], axis=0)
    shards_4 = jnp.stack(jnp.split(full_tensor, 4, axis=0), axis=0)
    return full_tensor, shards_4


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # DiLoCo
    theta_new, v_new, delta_avg = diloco_outer_step(g["theta_0"], g["theta_workers"], g["v_outer_init"], 0.7, 0.9)
    err = float(jnp.max(jnp.abs(theta_new - g["theta_new"])))
    assert err < 1e-5, f"theta_new error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_new - g["v_outer_new"])))
    assert err < 1e-5, f"v_new error {err:.2e}"
    err = float(jnp.max(jnp.abs(delta_avg - g["delta_avg"])))
    assert err < 1e-5, f"delta_avg error {err:.2e}"

    # PowerSGD
    recon, p_norm, q_new = powersgd_compress_decompress(g["grad_matrix"], g["probe_q"])
    err = float(jnp.max(jnp.abs(recon - g["powersgd_recon"])))
    assert err < 1e-5, f"recon error {err:.2e}"
    err = float(jnp.max(jnp.abs(p_norm - g["powersgd_p"])))
    assert err < 1e-5, f"p_norm error {err:.2e}"
    err = float(jnp.max(jnp.abs(q_new - g["powersgd_q"])))
    assert err < 1e-5, f"q_new error {err:.2e}"

    # DCP reshard
    full_tensor, shards_4 = distributed_checkpoint_reshard(g["shards_2"])
    err = float(jnp.max(jnp.abs(shards_4 - g["shards_4"])))
    assert err < 1e-5, f"shards_4 error {err:.2e}"

    print("16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
