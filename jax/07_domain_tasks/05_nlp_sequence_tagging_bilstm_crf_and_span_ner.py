"""Nlp Sequence Tagging Bilstm Crf and Span Ner — functional JAX implementation.

Topic: 07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner. Run ``python jax/07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def crf_log_partition(
    emissions: jnp.ndarray,
    transitions: jnp.ndarray,
) -> jnp.ndarray:
    b, t_len, k = emissions.shape
    alpha = emissions[:, 0, :]
    trans = jnp.expand_dims(transitions, 0)

    for t in range(1, t_len):
        emit = jnp.expand_dims(emissions[:, t, :], 1)
        scores = jnp.expand_dims(alpha, 2) + trans + emit
        alpha = jax.nn.logsumexp(scores, axis=1)

    return jax.nn.logsumexp(alpha, axis=-1)


def crf_sequence_score(
    emissions: jnp.ndarray,
    transitions: jnp.ndarray,
    tags: jnp.ndarray,
) -> jnp.ndarray:
    tags_int = tags.astype(jnp.int32)
    b, t_len, _k = emissions.shape
    batch_idx = jnp.arange(b)
    score = emissions[batch_idx, 0, tags_int[:, 0]]

    for t in range(1, t_len):
        prev_tag = tags_int[:, t - 1]
        curr_tag = tags_int[:, t]
        emit = emissions[batch_idx, t, curr_tag]
        trans = transitions[prev_tag, curr_tag]
        score = score + emit + trans

    return score


def crf_nll_loss(
    emissions: jnp.ndarray,
    transitions: jnp.ndarray,
    tags: jnp.ndarray,
) -> jnp.ndarray:
    return crf_log_partition(emissions, transitions) - crf_sequence_score(emissions, transitions, tags)


def span_endpoint_features(
    seq_embeds: jnp.ndarray,
    start_idx: jnp.ndarray,
    end_idx: jnp.ndarray,
) -> jnp.ndarray:
    s_int = start_idx.astype(jnp.int32)
    e_int = end_idx.astype(jnp.int32)
    b, s_len = s_int.shape
    batch_idx = jnp.broadcast_to(jnp.arange(b)[:, None], (b, s_len))
    h_start = seq_embeds[batch_idx, s_int]
    h_end = seq_embeds[batch_idx, e_int]
    h_diff = h_end - h_start
    return jnp.concatenate([h_start, h_end, h_diff], axis=-1)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. CRF log Z
    log_z_calc = crf_log_partition(g["emissions"], g["transitions"])
    err_z = float(jnp.max(jnp.abs(log_z_calc - g["log_z"])))
    assert err_z < 1e-5, f"crf log z error {err_z:.2e}"

    # 2. CRF Gold Sequence Score
    score_calc = crf_sequence_score(g["emissions"], g["transitions"], g["tags"])
    err_score = float(jnp.max(jnp.abs(score_calc - g["gold_score"])))
    assert err_score < 1e-5, f"crf gold score error {err_score:.2e}"

    # 3. CRF NLL Loss
    loss_calc = crf_nll_loss(g["emissions"], g["transitions"], g["tags"])
    err_loss = float(jnp.max(jnp.abs(loss_calc - g["crf_loss"])))
    assert err_loss < 1e-5, f"crf nll loss error {err_loss:.2e}"

    # 4. Span Features
    span_calc = span_endpoint_features(g["seq_embeds"], g["start_idx"], g["end_idx"])
    err_span = float(jnp.max(jnp.abs(span_calc - g["span_feats"])))
    assert err_span < 1e-5, f"span feats error {err_span:.2e}"

    print("07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
