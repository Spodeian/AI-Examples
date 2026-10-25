"""Safe RL Constrained MDP PPO Lagrangian CVaR and CBF — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf. Run ``python jax/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ppo_lagrangian_loss(
    logp: jnp.ndarray,
    old_logp: jnp.ndarray,
    adv_reward: jnp.ndarray,
    adv_cost: jnp.ndarray,
    lambda_param: float,
    clip_eps: float = 0.2,
) -> jnp.ndarray:
    """PPO-Lagrangian primal policy loss."""
    ratio = jnp.exp(logp - old_logp)
    surr1 = ratio * adv_reward
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_reward
    l_reward = jnp.mean(jnp.minimum(surr1, surr2))
    l_cost = jnp.mean(ratio * adv_cost)
    return -l_reward + float(lambda_param) * l_cost


def lagrangian_multiplier_step(
    lambda_val: float,
    cost_mean: float,
    cost_limit: float,
    lr_lambda: float = 0.05,
) -> float:
    """Dual ascent step on Lagrangian safety multiplier."""
    return max(0.0, float(lambda_val) + float(lr_lambda) * (float(cost_mean) - float(cost_limit)))


def cvar_risk_measure(
    costs: jnp.ndarray,
    alpha: float = 0.25,
) -> jnp.ndarray:
    """CVaR_alpha tail-risk metric."""
    costs_sorted = jnp.sort(costs.ravel())
    n = costs_sorted.shape[0]
    k = max(1, int(round(float(alpha) * n)))
    worst_k = costs_sorted[n - k :]
    return jnp.mean(worst_k)


def cbf_qp_safety_filter(
    u_nominal: jnp.ndarray,
    grad_h: jnp.ndarray,
    h_val: jnp.ndarray,
    alpha_cbf: float = 1.0,
) -> jnp.ndarray:
    """Control Barrier Function (CBF) quadratic program safety filter."""
    b = float(alpha_cbf) * h_val.ravel()
    a_dot_u = jnp.sum(grad_h * u_nominal, axis=-1)
    c_viol = a_dot_u + b
    
    deficit = jnp.clip(-c_viol, a_min=0.0)
    a_norm_sq = jnp.sum(grad_h * grad_h, axis=-1) + 1e-8
    correction_mag = deficit / a_norm_sq
    
    return u_nominal + correction_mag[:, None] * grad_h


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. PPO-Lagrangian
    ppo_val = ppo_lagrangian_loss(g["logp"], g["old_logp"], g["adv_reward"], g["adv_cost"], 1.2, clip_eps=0.2)
    assert float(jnp.abs(ppo_val - g["ppo_lag_ref"])) < 1e-5
    
    # 3. Lagrangian multiplier step
    lam_updated = lagrangian_multiplier_step(1.2, 0.35, 0.20, lr_lambda=0.1)
    assert float(jnp.abs(lam_updated - g["lam_updated_ref"])) < 1e-5
    
    # 4. CVaR
    cvar_val = cvar_risk_measure(g["costs"], alpha=0.25)
    assert float(jnp.abs(cvar_val - g["cvar_ref"])) < 1e-5
    
    # 5. CBF QP filter
    u_safe = cbf_qp_safety_filter(g["u_nominal"], g["grad_h"], g["h_val"], alpha_cbf=1.5)
    assert float(jnp.max(jnp.abs(u_safe - g["u_safe_ref"]))) < 1e-5
    
    print("11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
