"""Quantized Optimizer States: 8-bit and 4-bit Adam — functional JAX implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam.
Run ``python jax/14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam.py``
to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def quantize_8bit_first_moment(
    m: jnp.ndarray,
    block_size: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Block-wise symmetric dynamic INT8 quantization for first momentum m_t."""
    orig_shape = m.shape
    m_flat = m.reshape(-1, block_size)
    scale = jnp.clip(jnp.max(jnp.abs(m_flat), axis=-1, keepdims=True) / 127.0, a_min=1e-7)
    m_int = jnp.clip(jnp.round(m_flat / scale), -127.0, 127.0)
    m_deq = m_int * scale
    return m_deq.reshape(orig_shape), m_int.reshape(orig_shape), jnp.squeeze(scale, -1)


def quantize_8bit_second_moment(
    v: jnp.ndarray,
    block_size: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Block-wise non-linear log2 quantization for non-negative second momentum v_t."""
    orig_shape = v.shape
    v_flat = v.reshape(-1, block_size)
    v_max = jnp.clip(jnp.max(v_flat, axis=-1, keepdims=True), a_min=1e-8)
    v_norm = jnp.clip(v_flat / v_max, a_min=1e-8)
    v_log = jnp.log2(v_norm)

    v_int = jnp.clip(jnp.round((v_log + 24.0) * (255.0 / 24.0)), 0.0, 255.0)
    v_deq_norm = 2.0 ** ((v_int * (24.0 / 255.0)) - 24.0)
    v_deq = v_deq_norm * v_max
    return v_deq.reshape(orig_shape), v_int.reshape(orig_shape), jnp.squeeze(v_max, -1)


def quantize_4bit_moments(
    m: jnp.ndarray,
    v: jnp.ndarray,
    block_size: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """4-bit block-wise optimizer state quantization: signed 4-bit m_t and log 4-bit v_t."""
    orig_shape = m.shape
    m_flat = m.reshape(-1, block_size)
    scale_m = jnp.clip(jnp.max(jnp.abs(m_flat), axis=-1, keepdims=True) / 7.0, a_min=1e-7)
    m_int4 = jnp.clip(jnp.round(m_flat / scale_m), -7.0, 7.0)
    m_deq4 = m_int4 * scale_m

    v_flat = v.reshape(-1, block_size)
    v_max = jnp.clip(jnp.max(v_flat, axis=-1, keepdims=True), a_min=1e-8)
    v_norm = jnp.clip(v_flat / v_max, a_min=1e-6)
    v_log = jnp.log2(v_norm)
    v_int4 = jnp.clip(jnp.round((v_log + 15.0) * (15.0 / 15.0)), 0.0, 15.0)
    v_deq_norm = 2.0 ** ((v_int4 * (15.0 / 15.0)) - 15.0)
    v_deq4 = v_deq_norm * v_max

    return (
        m_deq4.reshape(orig_shape),
        m_int4.reshape(orig_shape),
        v_deq4.reshape(orig_shape),
        v_int4.reshape(orig_shape),
    )


def adamw_update_step(
    theta: jnp.ndarray,
    grad: jnp.ndarray,
    m_prev_deq: jnp.ndarray,
    v_prev_deq: jnp.ndarray,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
    step: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full AdamW parameter update with explicit momentum integration."""
    m_t = beta1 * m_prev_deq + (1.0 - beta1) * grad
    v_t = beta2 * v_prev_deq + (1.0 - beta2) * (grad ** 2)

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    m_hat = m_t / bc1
    v_hat = v_t / bc2

    update = (m_hat / (jnp.sqrt(v_hat) + eps)) + weight_decay * theta
    theta_next = theta - lr * update
    return theta_next, m_t, v_t


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor parity
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. 8-bit momentum quantization parity
    m_deq8_init, m_int8_init, scale_m8 = quantize_8bit_first_moment(g["m_prev"], block_size=16)
    err = float(jnp.max(jnp.abs(m_deq8_init - g["m_deq8_init"])))
    assert err < 1e-5, f"m_deq8_init error {err:.2e}"
    err = float(jnp.max(jnp.abs(m_int8_init - g["m_int8_init"])))
    assert err < 1e-5, f"m_int8_init error {err:.2e}"
    err = float(jnp.max(jnp.abs(scale_m8 - g["scale_m8"])))
    assert err < 1e-5, f"scale_m8 error {err:.2e}"

    v_deq8_init, v_int8_init, max_v8 = quantize_8bit_second_moment(g["v_prev"], block_size=16)
    err = float(jnp.max(jnp.abs(v_deq8_init - g["v_deq8_init"])))
    assert err < 1e-5, f"v_deq8_init error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_int8_init - g["v_int8_init"])))
    assert err < 1e-5, f"v_int8_init error {err:.2e}"
    err = float(jnp.max(jnp.abs(max_v8 - g["max_v8"])))
    assert err < 1e-5, f"max_v8 error {err:.2e}"

    # 3. 8-bit AdamW step parity
    theta_next_8bit, m_next8, v_next8 = adamw_update_step(
        g["theta"], g["grad"], g["m_deq8_init"], g["v_deq8_init"], beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=1e-2, step=1
    )
    err = float(jnp.max(jnp.abs(theta_next_8bit - g["theta_next_8bit"])))
    assert err < 1e-5, f"theta_next_8bit error {err:.2e}"

    m_deq8_next, _, _ = quantize_8bit_first_moment(m_next8, block_size=16)
    err = float(jnp.max(jnp.abs(m_deq8_next - g["m_deq8_next"])))
    assert err < 1e-5, f"m_deq8_next error {err:.2e}"

    v_deq8_next, _, _ = quantize_8bit_second_moment(v_next8, block_size=16)
    err = float(jnp.max(jnp.abs(v_deq8_next - g["v_deq8_next"])))
    assert err < 1e-5, f"v_deq8_next error {err:.2e}"

    # 4. 4-bit momentum quantization parity
    m_deq4_init, m_int4_init, v_deq4_init, v_int4_init = quantize_4bit_moments(g["m_prev"], g["v_prev"], block_size=16)
    err = float(jnp.max(jnp.abs(m_deq4_init - g["m_deq4_init"])))
    assert err < 1e-5, f"m_deq4_init error {err:.2e}"
    err = float(jnp.max(jnp.abs(m_int4_init - g["m_int4_init"])))
    assert err < 1e-5, f"m_int4_init error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_deq4_init - g["v_deq4_init"])))
    assert err < 1e-5, f"v_deq4_init error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_int4_init - g["v_int4_init"])))
    assert err < 1e-5, f"v_int4_init error {err:.2e}"

    # 5. 4-bit AdamW step parity
    theta_next_4bit, m_next4, v_next4 = adamw_update_step(
        g["theta"], g["grad"], g["m_deq4_init"], g["v_deq4_init"], beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=1e-2, step=1
    )
    err = float(jnp.max(jnp.abs(theta_next_4bit - g["theta_next_4bit"])))
    assert err < 1e-5, f"theta_next_4bit error {err:.2e}"

    m_deq4_next, _, v_deq4_next, _ = quantize_4bit_moments(m_next4, v_next4, block_size=16)
    err = float(jnp.max(jnp.abs(m_deq4_next - g["m_deq4_next"])))
    assert err < 1e-5, f"m_deq4_next error {err:.2e}"
    err = float(jnp.max(jnp.abs(v_deq4_next - g["v_deq4_next"])))
    assert err < 1e-5, f"v_deq4_next error {err:.2e}"

    print("14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
