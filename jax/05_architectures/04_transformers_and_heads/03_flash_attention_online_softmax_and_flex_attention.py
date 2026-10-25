"""Flash Attention, Online Softmax, and Flex Attention — JAX reference implementation.

Topic: 05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention
Golden weights: weights/05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def online_softmax_flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    block_size: int,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Tiled FlashAttention algorithm with Online Softmax accumulation in JAX."""
    B, T, D = q.shape
    S = k.shape[1]

    out = jnp.zeros((B, T, D), dtype=q.dtype)
    m = jnp.full((B, T, 1), -1e9, dtype=q.dtype)
    l = jnp.zeros((B, T, 1), dtype=q.dtype)

    num_blocks = (S + block_size - 1) // block_size

    for j in range(num_blocks):
        start = j * block_size
        end = min(start + block_size, S)

        k_block = k[:, start:end, :]
        v_block = v[:, start:end, :]

        scores_j = jnp.matmul(q, jnp.swapaxes(k_block, -1, -2)) * scale
        block_max = jnp.max(scores_j, axis=-1, keepdims=True)
        m_new = jnp.maximum(m, block_max)

        alpha = jnp.exp(m - m_new)
        p_j = jnp.exp(scores_j - m_new)

        l_new = alpha * l + jnp.sum(p_j, axis=-1, keepdims=True)
        out = (alpha * l * out + jnp.matmul(p_j, v_block)) / l_new
        m = m_new
        l = l_new

    return out, m, l


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. FlashAttention
    scale = float(golden["scale"][0])
    block_size = int(golden["block_size"][0])
    flash_out, final_m, final_l = online_softmax_flash_attention(
        golden["q"], golden["k"], golden["v"], block_size, scale
    )
    err_f = float(jnp.max(jnp.abs(flash_out - golden["flash_out"])))
    assert err_f < 1e-5, f"flash_out max abs error {err_f:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
