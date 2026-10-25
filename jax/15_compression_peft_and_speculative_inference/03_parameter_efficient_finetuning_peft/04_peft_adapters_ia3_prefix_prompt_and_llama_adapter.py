"""PEFT Adapters Ia3 Prefix Prompt and Llama Adapter — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter. Run ``python jax/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ia3_rescale(activation: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """(IA)^3 multiplicative scaling."""
    return activation * scale


def prefix_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    prefix_k: jnp.ndarray,
    prefix_v: jnp.ndarray,
) -> jnp.ndarray:
    """Prefix-Tuning attention with prepended key and value prefixes."""
    k_cat = jnp.concatenate([prefix_k, k], axis=1)
    v_cat = jnp.concatenate([prefix_v, v], axis=1)
    d = q.shape[-1]
    scale = 1.0 / jnp.sqrt(d)
    scores = jnp.matmul(q, jnp.swapaxes(k_cat, -1, -2)) * scale
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(attn, v_cat)


def llama_adapter_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    prefix_k: jnp.ndarray,
    prefix_v: jnp.ndarray,
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """LLaMA-Adapter zero-gated prefix attention."""
    k_cat = jnp.concatenate([prefix_k, k], axis=1)
    d = q.shape[-1]
    scale = 1.0 / jnp.sqrt(d)
    scores = jnp.matmul(q, jnp.swapaxes(k_cat, -1, -2)) * scale
    attn = jax.nn.softmax(scores, axis=-1)

    prefix_len = prefix_k.shape[1]
    attn_prefix = attn[:, :, :prefix_len]
    attn_seq = attn[:, :, prefix_len:]

    seq_out = jnp.matmul(attn_seq, v)
    prefix_out = jnp.matmul(attn_prefix, prefix_v)
    return seq_out + gate * prefix_out


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. (IA)^3
    ia3 = ia3_rescale(g["v"], g["ia3_scale"])
    err = float(jnp.max(jnp.abs(ia3 - g["ia3_out"])))
    assert err < 1e-5, f"ia3 error {err:.2e}"

    # 2. Prefix Attention
    prefix_out = prefix_attention(g["q"], g["k"], g["v"], g["prefix_k"], g["prefix_v"])
    err = float(jnp.max(jnp.abs(prefix_out - g["prefix_attn_out"])))
    assert err < 1e-5, f"prefix_attn error {err:.2e}"

    # 3. LLaMA-Adapter Zero-Gate
    llama_out = llama_adapter_forward(g["q"], g["k"], g["v"], g["prefix_k"], g["prefix_v"], g["gate"])
    err = float(jnp.max(jnp.abs(llama_out - g["llama_adapter_out"])))
    assert err < 1e-5, f"llama_adapter error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
