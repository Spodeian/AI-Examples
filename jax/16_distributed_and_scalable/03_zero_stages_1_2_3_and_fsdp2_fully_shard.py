"""Zero Stages 1 2 3 and FSDP2 Fully Shard — functional JAX implementation.

Topic: 16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard. Run ``python jax/16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def fsdp2_allgather_param(param_shards: jnp.ndarray) -> jnp.ndarray:
    """FSDP2 AllGather: reconstructs full 2D weight matrix from parameter shards."""
    world_size = param_shards.shape[0]
    return jnp.concatenate([param_shards[r] for r in range(world_size)], axis=0)


def fsdp2_forward_matmul(
    x: jnp.ndarray,
    param_shards: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """FSDP2 forward: AllGather parameter shards, compute matmul."""
    w_full = fsdp2_allgather_param(param_shards)
    h = x @ w_full
    return h, w_full


def fsdp2_backward_reducescatter(
    x: jnp.ndarray,
    grad_out: jnp.ndarray,
    world_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """FSDP2 backward: compute full gradient and ReduceScatter across ranks."""
    grad_w_full = x.T @ grad_out
    chunks = jnp.split(grad_w_full, world_size, axis=0)
    grad_w_shards = jnp.stack(chunks, axis=0)
    return grad_w_full, grad_w_shards


def zero1_adam_step(
    p: jnp.ndarray,
    g: jnp.ndarray,
    m: jnp.ndarray,
    v: jnp.ndarray,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    step: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ZeRO-1 Adam optimizer step on partitioned parameter slice."""
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g ** 2)

    bias_corr1 = 1.0 - beta1 ** step
    bias_corr2 = 1.0 - beta2 ** step

    m_hat = m_new / bias_corr1
    v_hat = v_new / bias_corr2

    p_new = p - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return p_new, m_new, v_new


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. FSDP2 Forward
    h, w_full = fsdp2_forward_matmul(g["x"], g["param_shards"])
    err = float(jnp.max(jnp.abs(w_full - g["w_full"])))
    assert err < 1e-5, f"w_full error {err:.2e}"
    err = float(jnp.max(jnp.abs(h - g["fsdp_fwd_out"])))
    assert err < 1e-5, f"fsdp_fwd_out error {err:.2e}"

    # 2. FSDP2 Backward
    grad_w_full, grad_w_shards = fsdp2_backward_reducescatter(g["x"], g["grad_out"], 2)
    err = float(jnp.max(jnp.abs(grad_w_full - g["grad_w_full"])))
    assert err < 1e-5, f"grad_w_full error {err:.2e}"
    err = float(jnp.max(jnp.abs(grad_w_shards - g["grad_w_shards"])))
    assert err < 1e-5, f"grad_w_shards error {err:.2e}"

    # 3. ZeRO-1 Adam Step
    p_new, m_new, v_new = zero1_adam_step(g["p_init"], g["g_init"], g["m_init"], g["v_init"], lr=0.01, step=1)
    err = float(jnp.max(jnp.abs(p_new - g["p_new"])))
    assert err < 1e-5, f"p_new error {err:.2e}"
    err = float(jnp.max(jnp.abs(m_new - g["m_new"])))
    assert err < 1e-5, f"m_new error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_new - g["v_new"])))
    assert err < 1e-5, f"v_new error {err:.2e}"

    print("16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
