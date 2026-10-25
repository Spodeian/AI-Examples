"""AI Watermarking AI Detection and Model Obfuscation — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation. Run ``python jax/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def watermark_logit_bias(
    logits: jnp.ndarray,
    green_mask: jnp.ndarray,
    delta: float,
) -> jnp.ndarray:
    """Kirchenbauer et al. (2023) green-list logit biasing: l' = l + delta * m_green."""
    return logits + float(delta) * jnp.expand_dims(green_mask, 0)


def watermark_z_score(
    green_counts: jnp.ndarray,
    seq_lens: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """One-tailed z-score hypothesis test for green token frequency."""
    expected = float(gamma) * seq_lens
    std = jnp.sqrt(seq_lens * float(gamma) * (1.0 - float(gamma)) + 1e-8)
    return (green_counts - expected) / std


def fast_detect_gpt_score(
    log_probs: jnp.ndarray,
    ref_log_probs: jnp.ndarray,
) -> jnp.ndarray:
    """Fast-DetectGPT (Bao et al. 2023) 1-pass conditional likelihood curvature score."""
    return jnp.mean(log_probs - ref_log_probs, axis=-1)


def passport_layer_forward(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    passport_key: jnp.ndarray,
    passport_weight: jnp.ndarray,
) -> jnp.ndarray:
    """Passport Layer (Fan et al. 2019) secret key gating."""
    scale = jax.nn.sigmoid(passport_key @ passport_weight)
    base = x @ weight + bias
    return base * jnp.expand_dims(scale, 0)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Watermark logit bias
    w_logits = watermark_logit_bias(g["logits"], g["green_mask"], delta=2.5)
    err = float(jnp.max(jnp.abs(w_logits - g["watermarked_logits_ref"])))
    assert err < 1e-5, f"watermark_logits error {err:.2e}"

    # 3. Z-scores
    z_scores = watermark_z_score(g["green_counts"], g["seq_lens"], gamma=0.5)
    err = float(jnp.max(jnp.abs(z_scores - g["z_scores_ref"])))
    assert err < 1e-5, f"z_scores error {err:.2e}"

    # 4. Fast-DetectGPT
    detect_scores = fast_detect_gpt_score(g["log_probs"], g["ref_log_probs"])
    err = float(jnp.max(jnp.abs(detect_scores - g["detect_scores_ref"])))
    assert err < 1e-5, f"detect_scores error {err:.2e}"

    # 5. Passport layer
    pass_out = passport_layer_forward(g["pass_x"], g["pass_w"], g["pass_b"], g["passport_key"], g["passport_weight"])
    err = float(jnp.max(jnp.abs(pass_out - g["passport_out_ref"])))
    assert err < 1e-5, f"passport_out error {err:.2e}"

    print("11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
