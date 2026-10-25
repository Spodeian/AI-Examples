"""Transformer Archetypes: BERT, GPT, T5, ViT, Swin, Perceiver, and Jamba — JAX reference implementation.

Topic: 05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba
Golden weights: weights/05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def perceiver_cross_attention(
    latents: jnp.ndarray,
    inputs: jnp.ndarray,
    w_q: jnp.ndarray,
    w_k: jnp.ndarray,
    w_v: jnp.ndarray,
    w_o: jnp.ndarray,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perceiver cross-attention bottleneck in JAX."""
    q = latents @ w_q  # [B, N, D_head]
    k = inputs @ w_k   # [B, M, D_head]
    v = inputs @ w_v   # [B, M, D_head]

    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale  # [B, N, M]
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - scores_max)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    ctx = jnp.matmul(attn_weights, v)
    out = ctx @ w_o
    return out, attn_weights


def causal_decoder_mask(scores: jnp.ndarray) -> jnp.ndarray:
    """Applies causal autoregressive mask in JAX."""
    T = scores.shape[-1]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    return jnp.where(jnp.expand_dims(mask, 0), scores, -1e9)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. Perceiver
    scale = float(golden["scale"][0])
    p_out, p_attn = perceiver_cross_attention(
        golden["latents"], golden["inputs"],
        golden["w_q"], golden["w_k"], golden["w_v"], golden["w_o"], scale
    )
    err_p = float(jnp.max(jnp.abs(p_out - golden["perceiver_out"])))
    err_pa = float(jnp.max(jnp.abs(p_attn - golden["perceiver_attn"])))
    assert err_p < 1e-5, f"perceiver_out max abs error {err_p:.2e}"
    assert err_pa < 1e-5, f"perceiver_attn max abs error {err_pa:.2e}"

    # 3. Causal mask
    c_scores = causal_decoder_mask(golden["raw_scores"])
    err_c = float(jnp.max(jnp.abs(c_scores - golden["causal_scores"])))
    assert err_c < 1e-5, f"causal_scores max abs error {err_c:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
