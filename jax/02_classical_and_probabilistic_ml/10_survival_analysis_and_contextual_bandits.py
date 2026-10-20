"""Survival Analysis and Contextual Bandits in JAX.

Features:
- Functional Kaplan-Meier survival estimator
- Cox Proportional Hazards (DeepSurv) negative partial log-likelihood via scan-based logcumsumexp
- Vectorized Linear Upper Confidence Bound (LinUCB) contextual bandit policy
- Sherman-Morrison O(d^2) matrix inverse rank-1 updates
"""

from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
import safetensors.torch
import numpy as np


def kaplan_meier_survival(
    T: jnp.ndarray,
    E: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes Kaplan-Meier survival curve S(t) = prod_{t_i <= t} (1 - d_i / n_i)."""
    event_mask = E == 1
    event_times = T[event_mask]
    unique_times = jnp.unique(event_times)
    
    surv_probs = []
    current_s = 1.0

    for t_i in unique_times:
        n_i = jnp.sum(T >= t_i)
        d_i = jnp.sum((T == t_i) & (E == 1))
        current_s = current_s * (1.0 - d_i / n_i)
        surv_probs.append(current_s)

    return unique_times, jnp.array(surv_probs, dtype=jnp.float32)


@jax.jit
def cox_ph_loss(
    risk_scores: jnp.ndarray,
    T: jnp.ndarray,
    E: jnp.ndarray,
) -> jnp.ndarray:
    """Computes negative Cox partial log-likelihood using scan-based logcumsumexp."""
    order = jnp.argsort(-T)
    sorted_risk = risk_scores[order]
    sorted_E = E[order]

    def logcumsumexp_step(carry, x_t):
        next_val = jnp.logaddexp(carry, x_t)
        return next_val, next_val

    _, log_risk_set = jax.lax.scan(logcumsumexp_step, -jnp.inf, sorted_risk)

    event_mask = sorted_E == 1
    loss = -jnp.sum(jnp.where(event_mask, sorted_risk - log_risk_set, 0.0))
    return loss


@jax.jit
def linucb_score_arms(
    x_context: jnp.ndarray,
    A_arms: jnp.ndarray,
    b_arms: jnp.ndarray,
    alpha: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes LinUCB exploration scores across bandit arms."""
    A_inv = jnp.linalg.inv(A_arms)
    theta_hat = jnp.einsum("aij,aj->ai", A_inv, b_arms)
    mean_reward = jnp.einsum("ai,i->a", theta_hat, x_context)
    var = jnp.einsum("i,aij,j->a", x_context, A_inv, x_context)
    scores = mean_reward + alpha * jnp.sqrt(var)
    selected = jnp.argmax(scores)
    return scores, selected


@jax.jit
def sherman_morrison_rank1_update(
    A_inv: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Updates matrix inverse in O(d^2) time after rank-1 addition A + x x^T."""
    A_inv_x = A_inv @ x
    denom = 1.0 + jnp.dot(x, A_inv_x)
    return A_inv - jnp.outer(A_inv_x, A_inv_x) / denom


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    X_surv = jnp.array(chk["X_surv"].numpy())
    T_surv = jnp.array(chk["T_surv"].numpy())
    E_surv = jnp.array(chk["E_surv"].numpy())
    w_surv = jnp.array(chk["w_surv"].numpy())
    b_surv = jnp.array(chk["b_surv"].numpy())
    golden_risk = chk["risk_scores"].numpy()
    golden_cox_loss = float(chk["cox_loss"][0].numpy())
    golden_km_times = chk["km_times"].numpy()
    golden_km_survival = chk["km_survival"].numpy()

    # 1. Cox Partial Log-Likelihood Parity
    risk = X_surv @ w_surv + b_surv
    diff_risk = float(np.max(np.abs(np.array(risk) - golden_risk)))
    assert diff_risk < 1e-5, f"JAX risk score mismatch: {diff_risk:.2e}"

    loss = float(cox_ph_loss(risk, T_surv, E_surv))
    diff_loss = abs(loss - golden_cox_loss)
    assert diff_loss < 1e-4, f"JAX Cox loss mismatch: {diff_loss:.2e}"

    # 2. Kaplan-Meier Parity
    km_t, km_s = kaplan_meier_survival(T_surv, E_surv)
    assert np.allclose(np.array(km_t), golden_km_times), "JAX KM event times mismatch"
    diff_km = float(np.max(np.abs(np.array(km_s) - golden_km_survival)))
    assert diff_km < 1e-5, f"JAX Kaplan-Meier survival mismatch: {diff_km:.2e}"

    # 3. LinUCB Parity
    A_arms = jnp.array(chk["A_arms"].numpy())
    b_arms = jnp.array(chk["b_arms"].numpy())
    x_ctx = jnp.array(chk["x_context"].numpy())
    golden_scores = chk["linucb_scores"].numpy()
    golden_arm = int(chk["selected_arm"][0].numpy())

    scores, selected = linucb_score_arms(x_ctx, A_arms, b_arms, alpha=1.0)
    diff_scores = float(np.max(np.abs(np.array(scores) - golden_scores)))
    assert diff_scores < 1e-5, f"JAX LinUCB scores mismatch: {diff_scores:.2e}"
    assert int(selected) == golden_arm, f"JAX selected arm mismatch: {int(selected)} vs {golden_arm}"

    # 4. Sherman-Morrison Update Parity
    A_inv_0 = jnp.linalg.inv(A_arms[0])
    A_inv_updated = sherman_morrison_rank1_update(A_inv_0, x_ctx)
    A_exact_inv = jnp.linalg.inv(A_arms[0] + jnp.outer(x_ctx, x_ctx))
    diff_sm = float(np.max(np.abs(np.array(A_inv_updated) - np.array(A_exact_inv))))
    assert diff_sm < 1e-5, f"JAX Sherman-Morrison update error: {diff_sm:.2e}"

    print("JAX Topic 10 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
