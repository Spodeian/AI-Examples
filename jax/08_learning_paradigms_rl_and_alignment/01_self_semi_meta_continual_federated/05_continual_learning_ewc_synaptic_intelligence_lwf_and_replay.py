"""Continual Learning EWC Synaptic Intelligence Lwf and Replay — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ewc_penalty(
    theta: jnp.ndarray,
    theta_star: jnp.ndarray,
    fisher_diag: jnp.ndarray,
    lambda_ewc: float = 10.0,
) -> jnp.ndarray:
    penalty = 0.5 * lambda_ewc * jnp.sum(fisher_diag * (theta - theta_star) ** 2)
    return jnp.expand_dims(penalty, axis=0)


def synaptic_intelligence_omega(
    delta_theta: jnp.ndarray,
    path_integral: jnp.ndarray,
    xi: float = 0.1,
) -> jnp.ndarray:
    return jnp.maximum(0.0, path_integral) / (delta_theta ** 2 + xi)


def lwf_distillation_loss(
    new_logits: jnp.ndarray,
    old_logits: jnp.ndarray,
    temperature: float = 2.0,
) -> jnp.ndarray:
    p_old = jax.nn.softmax(old_logits / temperature, axis=-1)
    log_p_new = jax.nn.log_softmax(new_logits / temperature, axis=-1)
    log_p_old = jax.nn.log_softmax(old_logits / temperature, axis=-1)
    kl = jnp.mean(jnp.sum(p_old * (log_p_old - log_p_new), axis=-1))
    return jnp.expand_dims(temperature * temperature * kl, axis=0)


def der_dark_replay_loss(
    current_logits: jnp.ndarray,
    past_logits: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> jnp.ndarray:
    mse_term = jnp.mean((current_logits - past_logits) ** 2)
    log_p = jax.nn.log_softmax(current_logits, axis=-1)
    b = current_logits.shape[0]
    ce_term = jnp.mean(-log_p[jnp.arange(b), labels])
    return jnp.expand_dims(alpha * mse_term + beta * ce_term, axis=0)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. EWC
    penalty = ewc_penalty(g["theta"], g["theta_star"], g["fisher_diag"], lambda_ewc=10.0)
    err_ewc = float(jnp.max(jnp.abs(penalty - g["ewc_penalty_ref"])))
    assert err_ewc < 1e-5, f"ewc error {err_ewc:.2e}"

    # 3. SI
    omega = synaptic_intelligence_omega(g["delta_theta"], g["path_integral"], xi=0.1)
    err_si = float(jnp.max(jnp.abs(omega - g["si_omega_ref"])))
    assert err_si < 1e-5, f"si error {err_si:.2e}"

    # 4. LwF
    lwf = lwf_distillation_loss(g["new_logits"], g["old_logits"], temperature=2.0)
    err_lwf = float(jnp.max(jnp.abs(lwf - g["lwf_loss_ref"])))
    assert err_lwf < 1e-5, f"lwf error {err_lwf:.2e}"

    # 5. DER++
    der = der_dark_replay_loss(g["current_logits"], g["past_logits"], g["labels"], alpha=0.5, beta=0.5)
    err_der = float(jnp.max(jnp.abs(der - g["der_loss_ref"])))
    assert err_der < 1e-5, f"der error {err_der:.2e}"

    print("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
