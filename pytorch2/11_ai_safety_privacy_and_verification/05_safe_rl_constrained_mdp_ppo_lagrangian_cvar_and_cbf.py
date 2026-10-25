"""Safe RL Constrained MDP PPO Lagrangian CVaR and CBF — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf
Golden weights: weights/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ppo_lagrangian_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv_reward: torch.Tensor,
    adv_cost: torch.Tensor,
    lambda_param: float,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO-Lagrangian primal policy loss for Constrained MDPs (Altman 1999; Ray et al. 2019).
    
    L_pi = - E[ min(r * A_R, clip(r) * A_R) ] + lambda * E[ r * A_C ]
    """
    ratio = torch.exp(logp - old_logp)
    surr1 = ratio * adv_reward
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_reward
    l_reward = torch.min(surr1, surr2).mean()
    l_cost = (ratio * adv_cost).mean()
    return -l_reward + float(lambda_param) * l_cost


def lagrangian_multiplier_step(
    lambda_val: float,
    cost_mean: float,
    cost_limit: float,
    lr_lambda: float = 0.05,
) -> float:
    """Dual ascent step on Lagrangian safety multiplier lambda:
    lambda' = max(0, lambda + lr * (J_C - d_max)).
    """
    return max(0.0, float(lambda_val) + float(lr_lambda) * (float(cost_mean) - float(cost_limit)))


def cvar_risk_measure(
    costs: torch.Tensor,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Conditional Value-at-Risk (CVaR_alpha) tail-risk metric (Chow et al. 2015).
    
    Averages top alpha worst-case costs: E[ C | C >= VaR_alpha(C) ].
    """
    costs_sorted, _ = torch.sort(costs.view(-1))
    n = costs_sorted.shape[0]
    k = max(1, int(round(float(alpha) * n)))
    worst_k = costs_sorted[n - k :]
    return torch.mean(worst_k)


def cbf_qp_safety_filter(
    u_nominal: torch.Tensor,
    grad_h: torch.Tensor,
    h_val: torch.Tensor,
    alpha_cbf: float = 1.0,
) -> torch.Tensor:
    """Control Barrier Function (CBF) quadratic program safety filter (Ames et al. 2019).
    
    Ensures forward invariance: dot(grad_h, u) + alpha * h(x) >= 0.
    Closed-form projection: u* = u_nom + max(0, -c_viol) / ||grad_h||^2 * grad_h.
    """
    # Constraint violation on nominal control: a^T u + b
    b = float(alpha_cbf) * h_val.view(-1)
    a_dot_u = torch.sum(grad_h * u_nominal, dim=-1)
    c_viol = a_dot_u + b
    
    # Violation deficit: max(0, -c_viol)
    deficit = torch.clamp(-c_viol, min=0.0)
    a_norm_sq = torch.sum(grad_h * grad_h, dim=-1) + 1e-8
    correction_mag = deficit / a_norm_sq  # [B]
    
    return u_nominal + correction_mag.unsqueeze(-1) * grad_h


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. PPO-Lagrangian inputs: B=8
    logp = torch.randn(8) * 0.2
    old_logp = torch.randn(8) * 0.2
    adv_reward = torch.randn(8) + 1.0
    adv_cost = torch.randn(8) + 0.5
    lambda_param = 1.2
    ppo_lag_ref = ppo_lagrangian_loss(logp, old_logp, adv_reward, adv_cost, lambda_param, clip_eps=0.2)
    
    # 2. Lagrangian dual step
    cost_mean = 0.35
    cost_limit = 0.20
    lam_updated_ref = torch.tensor(lagrangian_multiplier_step(lambda_param, cost_mean, cost_limit, lr_lambda=0.1))
    
    # 3. CVaR inputs: N=16
    costs = torch.randn(16) * 2.0 + 3.0
    cvar_ref = cvar_risk_measure(costs, alpha=0.25)
    
    # 4. CBF QP filter inputs: B=4, D=3
    u_nominal = torch.randn(4, 3)
    grad_h = torch.randn(4, 3)
    h_val = torch.tensor([0.5, -0.2, 0.0, -1.0])
    u_safe_ref = cbf_qp_safety_filter(u_nominal, grad_h, h_val, alpha_cbf=1.5)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "logp": logp,
        "old_logp": old_logp,
        "adv_reward": adv_reward,
        "adv_cost": adv_cost,
        "ppo_lag_ref": ppo_lag_ref,
        "costs": costs,
        "cvar_ref": cvar_ref,
        "lam_updated_ref": lam_updated_ref,
        "u_nominal": u_nominal,
        "grad_h": grad_h,
        "h_val": h_val,
        "u_safe_ref": u_safe_ref,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
