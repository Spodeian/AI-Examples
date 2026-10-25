"""Adversarial Robustness FGSM PGD TRADES GCG and Smoothing — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing. Run ``python jax/11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.nn
import jax.numpy as jnp
import jax.scipy.special as jsp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def fgsm_perturbation(
    x: jnp.ndarray,
    grad: jnp.ndarray,
    eps: float = 0.1,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> jnp.ndarray:
    x_adv = x + float(eps) * jnp.sign(grad)
    return jnp.clip(x_adv, float(x_min), float(x_max))


def pgd_step(
    x_adv: jnp.ndarray,
    x_orig: jnp.ndarray,
    grad: jnp.ndarray,
    alpha: float = 0.04,
    eps: float = 0.1,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> jnp.ndarray:
    step = x_adv + float(alpha) * jnp.sign(grad)
    step_proj = jnp.clip(step, x_orig - float(eps), x_orig + float(eps))
    return jnp.clip(step_proj, float(x_min), float(x_max))


def trades_loss(
    clean_logits: jnp.ndarray,
    adv_logits: jnp.ndarray,
    labels: jnp.ndarray,
    beta: float = 1.0,
) -> jnp.ndarray:
    log_p_clean = clean_logits - jax.nn.logsumexp(clean_logits, axis=-1, keepdims=True)
    labels_int = labels.astype(jnp.int32)
    ce_loss = -jnp.mean(jnp.take_along_axis(log_p_clean, labels_int[:, None], axis=-1))

    p_clean = jax.nn.softmax(clean_logits, axis=-1)
    log_p_adv = adv_logits - jax.nn.logsumexp(adv_logits, axis=-1, keepdims=True)
    kl = jnp.mean(jnp.sum(p_clean * (log_p_clean - log_p_adv), axis=-1))

    return ce_loss + float(beta) * kl


def randomized_smoothing_radius(
    p_top: jnp.ndarray,
    p_runnerup: jnp.ndarray,
    sigma: float = 0.25,
) -> jnp.ndarray:
    sqrt2 = 1.4142135623730951
    probit_top = sqrt2 * jsp.erfinv(2.0 * p_top - 1.0)
    probit_run = sqrt2 * jsp.erfinv(2.0 * p_runnerup - 1.0)
    return (float(sigma) / 2.0) * (probit_top - probit_run)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. FGSM
    fgsm_adv = fgsm_perturbation(g["adv_x"], g["adv_grad"], 0.1, 0.0, 1.0)
    assert float(jnp.max(jnp.abs(fgsm_adv - g["fgsm_adv_ref"]))) < 1e-5, "FGSM mismatch"

    # 3. PGD
    pgd_adv = pgd_step(fgsm_adv, g["adv_x"], g["adv_grad"], 0.04, 0.1, 0.0, 1.0)
    assert float(jnp.max(jnp.abs(pgd_adv - g["pgd_step_ref"]))) < 1e-5, "PGD mismatch"

    # 4. TRADES loss
    t_loss = trades_loss(g["clean_logits"], g["adv_logits"], g["labels"], 1.0)
    assert float(jnp.max(jnp.abs(t_loss - g["trades_loss_ref"]))) < 1e-5, "TRADES loss mismatch"

    # 5. Randomized smoothing radius
    radius = randomized_smoothing_radius(g["p_top"], g["p_runnerup"], 0.25)
    assert float(jnp.max(jnp.abs(radius - g["smoothing_radius_ref"]))) < 1e-5, "Smoothing radius mismatch"

    print("11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
