"""Backdoor Trojan and Poisoning Defense Neural Cleanse and Spectral — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral. Run ``python jax/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def spectral_signature_scores(representations: jnp.ndarray) -> jnp.ndarray:
    """Spectral signature outlier scores via representation covariance."""
    mu = jnp.mean(representations, axis=0, keepdims=True)
    centered = representations - mu
    cov = (centered.T @ centered) / float(representations.shape[0])
    
    d = cov.shape[0]
    v = jnp.ones(d) / (float(d) ** 0.5)
    for _ in range(30):
        v = cov @ v
        v = v / (jnp.linalg.norm(v) + 1e-8)
    
    proj = centered @ v
    return proj ** 2


def neural_cleanse_anomaly_index(l1_norms: jnp.ndarray) -> jnp.ndarray:
    """Neural Cleanse MAD outlier detection index."""
    med = jnp.median(l1_norms)
    abs_dev = jnp.abs(l1_norms - med)
    mad = jnp.median(abs_dev) + 1e-8
    return (med - l1_norms) / (1.4826 * mad)


def strip_entropy(superposed_logits: jnp.ndarray) -> jnp.ndarray:
    """STRIP runtime input trojan detection via Shannon entropy."""
    probs = jax.nn.softmax(superposed_logits, axis=-1)
    entropy_per_sample = -jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)
    return jnp.mean(entropy_per_sample)


def trojan_trigger_apply(
    x: jnp.ndarray,
    mask: jnp.ndarray,
    pattern: jnp.ndarray,
) -> jnp.ndarray:
    """BadNets blending trojan trigger application."""
    return x * (1.0 - mask) + pattern * mask


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. Spectral signatures
    spec_scores = spectral_signature_scores(g["representations"])
    assert float(jnp.max(jnp.abs(spec_scores - g["spectral_scores_ref"]))) < 1e-5
    
    # 3. Neural Cleanse MAD
    mad_anomaly = neural_cleanse_anomaly_index(g["l1_norms"])
    assert float(jnp.max(jnp.abs(mad_anomaly - g["mad_anomaly_ref"]))) < 1e-5
    
    # 4. STRIP
    h_benign = strip_entropy(g["logits_benign"])
    assert float(jnp.abs(h_benign - g["entropy_benign_ref"])) < 1e-5
    
    h_trojan = strip_entropy(g["logits_trojan"])
    assert float(jnp.abs(h_trojan - g["entropy_trojan_ref"])) < 1e-5
    
    # 5. Trigger apply
    x_tr = trojan_trigger_apply(g["x_input"], g["mask"], g["pattern"])
    assert float(jnp.max(jnp.abs(x_tr - g["x_trojan_ref"]))) < 1e-5
    
    print("11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
