"""Text Generation Decoding and Structured Grammar Sampling — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling. Run ``python jax/17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling.safetensors"

MASK_VALUE = -10000.0


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def apply_repetition_penalty(logits: jnp.ndarray, history: jnp.ndarray, penalty: float = 1.25) -> jnp.ndarray:
    """Applies Keskar et al. multiplicative repetition penalty based on past token history."""
    b_size, _ = logits.shape
    out = logits
    for b in range(b_size):
        for s in range(history.shape[1]):
            idx = int(history[b, s])
            val = out[b, idx]
            new_val = jnp.where(val > 0.0, val / penalty, val * penalty)
            out = out.at[b, idx].set(new_val)
    return out


def apply_grammar_mask(logits: jnp.ndarray, grammar_mask: jnp.ndarray, mask_value: float = MASK_VALUE) -> jnp.ndarray:
    """Pre-softmax grammar/FSM bitmask: invalid transition tokens are set to mask_value."""
    return jnp.where(grammar_mask > 0.5, logits, mask_value)


def apply_temperature(logits: jnp.ndarray, temperature: float = 0.8) -> jnp.ndarray:
    """Scales logits by temperature T before softmax."""
    return logits / max(temperature, 1e-5)


def apply_top_k(logits: jnp.ndarray, k: int = 4, mask_value: float = MASK_VALUE) -> jnp.ndarray:
    """Keeps top-k logits per row, masking out all lower logits to mask_value."""
    sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
    k_th = sorted_logits[:, k - 1 : k]
    return jnp.where(logits >= k_th, logits, mask_value)


def apply_min_p(probs: jnp.ndarray, min_p: float = 0.1) -> jnp.ndarray:
    """Min-P sampling filter: retains tokens with p_i >= p_max * min_p, then renormalizes."""
    p_max = jnp.max(probs, axis=-1, keepdims=True)
    threshold = p_max * min_p
    filtered = jnp.where(probs >= threshold, probs, 0.0)
    sums = jnp.clip(jnp.sum(filtered, axis=-1, keepdims=True), a_min=1e-12)
    return filtered / sums


def decoding_pipeline(
    logits: jnp.ndarray,
    history: jnp.ndarray,
    grammar_mask: jnp.ndarray,
    temperature: float = 0.8,
    repetition_penalty: float = 1.25,
    top_k: int = 4,
    min_p: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Complete decoding pipeline: rep penalty -> grammar mask -> temp -> top-k -> softmax -> min-p."""
    x = apply_repetition_penalty(logits, history, repetition_penalty)
    x = apply_grammar_mask(x, grammar_mask)
    x = apply_temperature(x, temperature)
    x = apply_top_k(x, top_k)
    probs = jax.nn.softmax(x, axis=-1)
    probs = apply_min_p(probs, min_p)
    greedy_tokens = jnp.argmax(probs, axis=-1).astype(jnp.float32)
    return probs, greedy_tokens


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    rep = apply_repetition_penalty(g["logits"], g["history"], 1.25)
    err_rep = float(jnp.max(jnp.abs(rep - g["rep_penalized"])))
    assert err_rep < 1e-5, f"rep penalty error {err_rep:.2e}"

    masked = apply_grammar_mask(rep, g["grammar_mask"])
    err_mask = float(jnp.max(jnp.abs(masked - g["masked_logits"])))
    assert err_mask < 1e-5, f"grammar mask error {err_mask:.2e}"

    temp = apply_temperature(masked, 0.8)
    err_temp = float(jnp.max(jnp.abs(temp - g["temp_logits"])))
    assert err_temp < 1e-5, f"temperature error {err_temp:.2e}"

    top_k = apply_top_k(temp, 4)
    err_topk = float(jnp.max(jnp.abs(top_k - g["top_k_logits"])))
    assert err_topk < 1e-5, f"top-k error {err_topk:.2e}"

    probs, greedy = decoding_pipeline(
        g["logits"], g["history"], g["grammar_mask"], temperature=0.8, repetition_penalty=1.25, top_k=4, min_p=0.1
    )
    err_probs = float(jnp.max(jnp.abs(probs - g["probs"])))
    assert err_probs < 1e-5, f"probs error {err_probs:.2e}"

    err_tokens = float(jnp.max(jnp.abs(greedy - g["greedy_tokens"])))
    assert err_tokens < 1e-5, f"greedy tokens error {err_tokens:.2e}"

    print("17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
