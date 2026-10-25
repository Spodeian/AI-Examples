"""Transformer Heads: Induction, OV/QK Circuits, MTP, and Registers — JAX reference implementation.

Topic: 05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers
Golden weights: weights/05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ov_qk_circuits(
    x_q: jnp.ndarray,
    x_k: jnp.ndarray,
    x_v: jnp.ndarray,
    w_q: jnp.ndarray,
    w_k: jnp.ndarray,
    w_v: jnp.ndarray,
    w_o: jnp.ndarray,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluates decoupled QK and OV circuits in JAX."""
    q = x_q @ w_q
    k = x_k @ w_k
    v = x_v @ w_v
    ov = v @ w_o

    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - scores_max)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.matmul(attn_weights, ov)
    return out, attn_weights


def register_prepend_and_mtp(
    x_seq: jnp.ndarray,
    registers: jnp.ndarray,
    w_mtp: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Prepends register tokens and computes Multi-Token Prediction heads in JAX."""
    x_aug = jnp.concatenate([registers, x_seq], axis=1)
    x_last = x_seq[:, -1, :]
    mtp_logits = jnp.einsum("bd,kvd->bkv", x_last, w_mtp)
    return x_aug, mtp_logits


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. Circuits
    scale = float(golden["scale"][0])
    out, attn = ov_qk_circuits(
        golden["x_q"], golden["x_k"], golden["x_v"],
        golden["w_q"], golden["w_k"], golden["w_v"], golden["w_o"], scale
    )
    err_o = float(jnp.max(jnp.abs(out - golden["circuit_out"])))
    err_at = float(jnp.max(jnp.abs(attn - golden["circuit_attn"])))
    assert err_o < 1e-5, f"circuit_out max abs error {err_o:.2e}"
    assert err_at < 1e-5, f"circuit_attn max abs error {err_at:.2e}"

    # 3. Registers & MTP
    x_aug, mtp_logits = register_prepend_and_mtp(golden["x_q"], golden["registers"], golden["w_mtp"])
    err_aug = float(jnp.max(jnp.abs(x_aug - golden["x_aug"])))
    err_mtp = float(jnp.max(jnp.abs(mtp_logits - golden["mtp_logits"])))
    assert err_aug < 1e-5, f"x_aug max abs error {err_aug:.2e}"
    assert err_mtp < 1e-5, f"mtp_logits max abs error {err_mtp:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
