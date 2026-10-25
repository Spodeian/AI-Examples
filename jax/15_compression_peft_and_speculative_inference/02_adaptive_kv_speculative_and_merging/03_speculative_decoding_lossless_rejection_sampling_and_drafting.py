"""Speculative Decoding Lossless Rejection Sampling and Drafting — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting. Run ``python jax/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def speculative_rejection_step(
    target_probs: jnp.ndarray,
    draft_probs: jnp.ndarray,
    draft_tokens: jnp.ndarray,
    uniform_rands: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch, gamma, vocab = target_probs.shape
    accepted_mask = jnp.zeros((batch, gamma), dtype=jnp.float32)
    num_accepted = jnp.zeros((batch,), dtype=jnp.float32)
    residual_probs = jnp.zeros((batch, vocab), dtype=jnp.float32)

    for b in range(batch):
        rejected_pos = -1
        acc_count = 0.0
        for i in range(gamma):
            t_id = int(draft_tokens[b, i])
            p_val = float(target_probs[b, i, t_id])
            q_val = float(draft_probs[b, i, t_id])
            ratio = min(1.0, p_val / max(q_val, 1e-8))

            if float(uniform_rands[b, i]) <= ratio:
                accepted_mask = accepted_mask.at[b, i].set(1.0)
                acc_count += 1.0
            else:
                rejected_pos = i
                break

        num_accepted = num_accepted.at[b].set(acc_count)

        if rejected_pos >= 0:
            diff = jnp.maximum(target_probs[b, rejected_pos] - draft_probs[b, rejected_pos], 0.0)
            norm = float(jnp.sum(diff))
            if norm > 1e-8:
                residual_probs = residual_probs.at[b].set(diff / norm)
            else:
                residual_probs = residual_probs.at[b].set(target_probs[b, rejected_pos])
        else:
            residual_probs = residual_probs.at[b].set(target_probs[b, -1])

    return accepted_mask, num_accepted, residual_probs


def prompt_lookup_ngram_draft(
    input_ids: jnp.ndarray,
    ngram_size: int = 3,
    gamma: int = 4,
) -> jnp.ndarray:
    seq_len = input_ids.shape[0]
    if seq_len < ngram_size + 1:
        return jnp.zeros(gamma, dtype=jnp.float32)

    suffix = input_ids[-ngram_size:]
    best_pos = -1

    for i in range(seq_len - ngram_size - 1, -1, -1):
        if bool(jnp.all(input_ids[i : i + ngram_size] == suffix)):
            best_pos = i + ngram_size
            break

    draft = []
    if best_pos >= 0:
        for k in range(gamma):
            idx = best_pos + k
            if idx < seq_len:
                draft.append(float(input_ids[idx]))
            else:
                draft.append(0.0)
    else:
        draft = [0.0] * gamma

    return jnp.array(draft, dtype=jnp.float32)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Speculative rejection sampling
    acc_mask, n_acc, res_p = speculative_rejection_step(
        g["target_probs"], g["draft_probs"], g["draft_tokens"], g["uniform_rands"]
    )
    err = float(jnp.max(jnp.abs(acc_mask - g["accepted_mask"])))
    assert err < 1e-5, f"accepted_mask error {err:.2e}"
    err = float(jnp.max(jnp.abs(n_acc - g["num_accepted"])))
    assert err < 1e-5, f"num_accepted error {err:.2e}"
    err = float(jnp.max(jnp.abs(res_p - g["residual_probs"])))
    assert err < 1e-5, f"residual_probs error {err:.2e}"

    # 3. Prompt lookup draft
    draft = prompt_lookup_ngram_draft(g["prompt_seq"], ngram_size=3, gamma=4)
    err = float(jnp.max(jnp.abs(draft - g["draft_ngram"])))
    assert err < 1e-5, f"draft_ngram error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
