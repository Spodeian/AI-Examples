"""Machine Unlearning SISA Influence Scrub NPO and RMU — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu. Run ``python jax/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def npo_loss(
    logits_theta: jnp.ndarray,
    logits_ref: jnp.ndarray,
    labels: jnp.ndarray,
    beta: float = 0.1,
) -> jnp.ndarray:
    """Negative Preference Optimization (NPO) loss."""
    log_p_theta = jax.nn.log_softmax(logits_theta, axis=-1)
    log_p_ref = jax.nn.log_softmax(logits_ref, axis=-1)
    
    b_idx = jnp.arange(logits_theta.shape[0])
    log_pi_theta = log_p_theta[b_idx, labels]
    log_pi_ref = log_p_ref[b_idx, labels]
    
    log_ratio = log_pi_theta - log_pi_ref
    loss = (2.0 / beta) * jax.nn.softplus(beta * log_ratio)
    return jnp.mean(loss)


def rmu_loss(
    h_theta_forget: jnp.ndarray,
    h_target_random: jnp.ndarray,
    h_theta_retain: jnp.ndarray,
    h_ref_retain: jnp.ndarray,
    alpha: float = 1.0,
) -> jnp.ndarray:
    """Representation Misdirection for Unlearning (RMU)."""
    l_forget = jnp.mean((h_theta_forget - h_target_random) ** 2)
    l_retain = jnp.mean((h_theta_retain - h_ref_retain) ** 2)
    return l_forget + float(alpha) * l_retain


def scrub_loss(
    logits_f_theta: jnp.ndarray,
    labels_f: jnp.ndarray,
    logits_r_theta: jnp.ndarray,
    logits_r_ref: jnp.ndarray,
    gamma: float = 1.0,
    tau: float = 5.0,
) -> jnp.ndarray:
    """SCRUB student-teacher unlearning loss."""
    # Cross entropy on forget set
    log_p_f = jax.nn.log_softmax(logits_f_theta, axis=-1)
    b_idx = jnp.arange(logits_f_theta.shape[0])
    ce_forget = -jnp.mean(log_p_f[b_idx, labels_f])
    l_unlearn = -jnp.clip(ce_forget, a_max=tau)
    
    # KL on retain set
    p_ref = jax.nn.softmax(logits_r_ref, axis=-1)
    log_p_ref = jax.nn.log_softmax(logits_r_ref, axis=-1)
    log_p_theta = jax.nn.log_softmax(logits_r_theta, axis=-1)
    kl_retain = jnp.mean(jnp.sum(p_ref * (log_p_ref - log_p_theta), axis=-1))
    
    return l_unlearn + float(gamma) * kl_retain


def sisa_ensemble_predict(shard_logits: jnp.ndarray) -> jnp.ndarray:
    """SISA multi-shard probability aggregation."""
    probs = jax.nn.softmax(shard_logits, axis=-1)
    return jnp.mean(probs, axis=0)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. NPO
    npo_val = npo_loss(g["logits_theta"], g["logits_ref"], g["labels"], beta=0.2)
    assert float(jnp.abs(npo_val - g["npo_ref"])) < 1e-5
    
    # 3. RMU
    rmu_val = rmu_loss(g["h_theta_forget"], g["h_target_random"], g["h_theta_retain"], g["h_ref_retain"], alpha=1.5)
    assert float(jnp.abs(rmu_val - g["rmu_ref"])) < 1e-5
    
    # 4. SCRUB
    scrub_val = scrub_loss(g["logits_f_theta"], g["labels_f"], g["logits_r_theta"], g["logits_r_ref"], gamma=1.2, tau=4.0)
    assert float(jnp.abs(scrub_val - g["scrub_ref"])) < 1e-5
    
    # 5. SISA
    sisa_val = sisa_ensemble_predict(g["shard_logits"])
    assert float(jnp.max(jnp.abs(sisa_val - g["sisa_ref"]))) < 1e-5
    
    print("11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
