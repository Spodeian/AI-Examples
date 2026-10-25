"""Sparse, Sliding Window, and Sink Attention — JAX reference implementation.

Topic: 05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention
Golden weights: weights/05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def streaming_sink_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    num_sinks: int,
    window_size: int,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """StreamingLLM Attention with Attention Sinks and Sliding Window in JAX."""
    B, T, D = q.shape
    S = k.shape[1]

    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale

    i_idx = jnp.expand_dims(jnp.arange(T), 1)
    j_idx = jnp.expand_dims(jnp.arange(S), 0)

    is_sink = j_idx < num_sinks
    is_window = (i_idx - j_idx < window_size) & (i_idx >= j_idx)
    valid_mask = (is_sink | is_window) & (j_idx <= i_idx)

    mask_value = -1e9
    scores_masked = jnp.where(jnp.expand_dims(valid_mask, 0), scores, mask_value)

    scores_max = jnp.max(scores_masked, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores_masked - scores_max)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.matmul(attn_weights, v)
    return out, attn_weights


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. Sink Attention
    num_sinks = int(golden["num_sinks"][0])
    window_size = int(golden["window_size"][0])
    scale = float(golden["scale"][0])

    out, weights = streaming_sink_attention(
        golden["q"], golden["k"], golden["v"], num_sinks, window_size, scale
    )
    err_o = float(jnp.max(jnp.abs(out - golden["sink_out"])))
    err_w = float(jnp.max(jnp.abs(weights - golden["sink_weights"])))
    assert err_o < 1e-5, f"sink_out max abs error {err_o:.2e}"
    assert err_w < 1e-5, f"sink_weights max abs error {err_w:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
