"""Attention Variants: MHA, MQA, GQA, MLA, and DiffAttn — JAX reference implementation.

Topic: 05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn
Golden weights: weights/05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def grouped_query_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Grouped-Query Attention (GQA) in JAX."""
    B, H, T, D = q.shape
    G = k.shape[1]
    group_size = H // G

    # Repeat k and v across group dimension
    k_exp = jnp.repeat(k, group_size, axis=1)  # [B, H, S, D]
    v_exp = jnp.repeat(v, group_size, axis=1)  # [B, H, S, D]

    scores = jnp.matmul(q, jnp.swapaxes(k_exp, -1, -2)) * scale
    # Softmax
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - scores_max)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.matmul(attn_weights, v_exp)
    return out, attn_weights


def differential_attention(
    q1: jnp.ndarray,
    k1: jnp.ndarray,
    q2: jnp.ndarray,
    k2: jnp.ndarray,
    v: jnp.ndarray,
    lambda_param: float,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differential Attention (DiffAttn) in JAX."""
    scores1 = jnp.matmul(q1, jnp.swapaxes(k1, -1, -2)) * scale
    scores2 = jnp.matmul(q2, jnp.swapaxes(k2, -1, -2)) * scale

    a1 = jnp.exp(scores1 - jnp.max(scores1, axis=-1, keepdims=True))
    a1 = a1 / jnp.sum(a1, axis=-1, keepdims=True)

    a2 = jnp.exp(scores2 - jnp.max(scores2, axis=-1, keepdims=True))
    a2 = a2 / jnp.sum(a2, axis=-1, keepdims=True)

    diff_weights = a1 - lambda_param * a2
    out = jnp.matmul(diff_weights, v)
    return out, diff_weights


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. GQA
    scale = float(golden["scale"][0])
    gqa_out, gqa_weights = grouped_query_attention(golden["q"], golden["k"], golden["v"], scale)
    err_g = float(jnp.max(jnp.abs(gqa_out - golden["gqa_out"])))
    err_gw = float(jnp.max(jnp.abs(gqa_weights - golden["gqa_weights"])))
    assert err_g < 1e-5, f"gqa_out max abs error {err_g:.2e}"
    assert err_gw < 1e-5, f"gqa_weights max abs error {err_gw:.2e}"

    # 3. DiffAttn
    lambda_val = float(golden["lambda_val"][0])
    diff_out, diff_weights = differential_attention(
        golden["q1"], golden["k1"], golden["q2"], golden["k2"], golden["v_diff"], lambda_val, scale
    )
    err_d = float(jnp.max(jnp.abs(diff_out - golden["diff_out"])))
    err_dw = float(jnp.max(jnp.abs(diff_weights - golden["diff_weights"])))
    assert err_d < 1e-5, f"diff_out max abs error {err_d:.2e}"
    assert err_dw < 1e-5, f"diff_weights max abs error {err_dw:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
