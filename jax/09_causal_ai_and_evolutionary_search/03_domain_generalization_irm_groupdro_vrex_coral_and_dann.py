"""Domain Generalization IRM GroupDRO VREx CORAL and DANN — functional JAX implementation.

Topic: 09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann. Run ``python jax/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def irm_v1_penalty(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Invariant Risk Minimization (IRMv1) gradient penalty."""
    probs = jax.nn.sigmoid(logits)
    error = probs - targets
    grad = jnp.mean(logits * error)
    return (grad * grad)[None]


def vrex_variance_penalty(env_losses: jnp.ndarray) -> jnp.ndarray:
    """Variance Risk Extrapolation (V-REx) penalty across environments."""
    mean_loss = jnp.mean(env_losses)
    diff = env_losses - mean_loss
    return jnp.mean(diff * diff)[None]


def group_dro_update(
    env_losses: jnp.ndarray,
    q_weights: jnp.ndarray,
    eta: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Group Distributionally Robust Optimization (GroupDRO)."""
    dro_loss = jnp.sum(q_weights * env_losses)[None]
    unnorm_q = q_weights * jnp.exp(eta * env_losses)
    q_up = unnorm_q / jnp.sum(unnorm_q)
    return dro_loss, q_up


def deep_coral_loss(phi_s: jnp.ndarray, phi_t: jnp.ndarray) -> jnp.ndarray:
    """Deep CORAL covariance alignment loss."""
    ns, d = phi_s.shape
    nt, _ = phi_t.shape

    cent_s = phi_s - jnp.mean(phi_s, axis=0, keepdims=True)
    cent_t = phi_t - jnp.mean(phi_t, axis=0, keepdims=True)

    cov_s = (cent_s.T @ cent_s) / float(ns - 1)
    cov_t = (cent_t.T @ cent_t) / float(nt - 1)

    diff = cov_s - cov_t
    return (jnp.sum(diff * diff) / (4.0 * float(d * d)))[None]


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # IRM
    irm_pen = irm_v1_penalty(g["irm_logits"], g["irm_targets"])
    err_irm = float(jnp.max(jnp.abs(irm_pen - g["irm_penalty_ref"])))
    assert err_irm < 1e-5, f"IRM penalty error {err_irm:.2e}"

    # V-REx
    vrex = vrex_variance_penalty(g["env_losses"])
    err_vrex = float(jnp.max(jnp.abs(vrex - g["vrex_penalty_ref"])))
    assert err_vrex < 1e-5, f"V-REx penalty error {err_vrex:.2e}"

    # GroupDRO
    dro_loss, q_up = group_dro_update(g["env_losses"], g["q_weights"], eta=0.1)
    err_dro = float(jnp.max(jnp.abs(dro_loss - g["dro_loss_ref"])))
    assert err_dro < 1e-5, f"GroupDRO loss error {err_dro:.2e}"
    err_qup = float(jnp.max(jnp.abs(q_up - g["q_updated_ref"])))
    assert err_qup < 1e-5, f"GroupDRO q update error {err_qup:.2e}"

    # Deep CORAL
    coral = deep_coral_loss(g["phi_source"], g["phi_target"])
    err_coral = float(jnp.max(jnp.abs(coral - g["coral_loss_ref"])))
    assert err_coral < 1e-5, f"Deep CORAL error {err_coral:.2e}"

    print("09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
