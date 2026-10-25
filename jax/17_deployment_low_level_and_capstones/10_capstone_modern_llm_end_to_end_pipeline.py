"""Capstone Modern LLM End to End Pipeline — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline. Run ``python jax/17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def rmsnorm(x: jnp.ndarray, gamma: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Root Mean Square Normalization."""
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def causal_attention_block(
    x: jnp.ndarray,
    wq: jnp.ndarray,
    wk: jnp.ndarray,
    wv: jnp.ndarray,
    wo: jnp.ndarray,
) -> jnp.ndarray:
    """Causal self-attention layer."""
    _, s, d = x.shape
    q = x @ wq
    k = x @ wk
    v = x @ wv

    scores = (q @ jnp.swapaxes(k, -2, -1)) / math.sqrt(d)
    mask = jnp.triu(jnp.full((s, s), -10000.0), k=1)
    attn = jax.nn.softmax(scores + mask, axis=-1)
    return (attn @ v) @ wo


def swiglu_ffn_block(
    x: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
) -> jnp.ndarray:
    """SwiGLU feed-forward network block."""
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down


def modern_llm_pipeline_forward(
    x: jnp.ndarray,
    gamma1: jnp.ndarray,
    wq: jnp.ndarray,
    wk: jnp.ndarray,
    wv: jnp.ndarray,
    wo: jnp.ndarray,
    gamma2: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    gamma_final: jnp.ndarray,
    w_head: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """End-to-end modern LLM forward pipeline."""
    normed_1 = rmsnorm(x, gamma1)
    attn = causal_attention_block(normed_1, wq, wk, wv, wo)
    h1 = x + attn

    normed_2 = rmsnorm(h1, gamma2)
    ffn = swiglu_ffn_block(normed_2, w_gate, w_up, w_down)
    h_out = h1 + ffn

    h_final = rmsnorm(h_out, gamma_final)
    logits = h_final @ w_head
    return logits, h_out, attn, ffn


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    logits, h_out, attn, ffn = modern_llm_pipeline_forward(
        g["x"],
        g["gamma1"],
        g["wq"],
        g["wk"],
        g["wv"],
        g["wo"],
        g["gamma2"],
        g["w_gate"],
        g["w_up"],
        g["w_down"],
        g["gamma_final"],
        g["w_head"],
    )
    assert float(jnp.max(jnp.abs(logits - g["logits"]))) < 1e-5
    assert float(jnp.max(jnp.abs(h_out - g["h_out"]))) < 1e-5
    assert float(jnp.max(jnp.abs(attn - g["attn"]))) < 1e-5
    assert float(jnp.max(jnp.abs(ffn - g["ffn"]))) < 1e-5

    print("17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
