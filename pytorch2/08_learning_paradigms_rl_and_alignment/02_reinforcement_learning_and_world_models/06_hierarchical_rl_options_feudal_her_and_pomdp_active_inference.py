"""Hierarchical RL Options Feudal Her and POMDP Active Inference — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def her_goal_relabeling(
    states: torch.Tensor,
    goals: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Hindsight Experience Replay (HER) sparse binary goal achievement reward:
    r = 1.0 if ||states - goals||_2 <= threshold else 0.0
    states: [B, D]
    goals: [B, D]
    Returns: [B] binary rewards (0.0 or 1.0).
    """
    dists = torch.norm(states - goals, p=2, dim=-1)
    return (dists <= threshold).float()


def feudal_directional_reward(
    delta_s: torch.Tensor,
    subgoal_g: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """FeUdal Networks cosine directional reward:
    r_w = (delta_s . g) / (||delta_s|| * ||g|| + eps)
    delta_s: [B, D] state displacement
    subgoal_g: [B, D] manager sub-goal direction
    Returns: [B] cosine similarity reward in [-1.0, 1.0].
    """
    dot_product = torch.sum(delta_s * subgoal_g, dim=-1)
    norm_delta = torch.norm(delta_s, p=2, dim=-1)
    norm_g = torch.norm(subgoal_g, p=2, dim=-1)
    denom = torch.clamp(norm_delta * norm_g, min=eps)
    return dot_product / denom


def option_critic_termination_loss(
    beta_omega: torch.Tensor,
    q_omega: torch.Tensor,
    v_s: torch.Tensor,
    deliberation_cost: float = 0.01,
) -> torch.Tensor:
    """Option-Critic termination gradient loss with deliberation cost:
    L(beta) = mean( beta_omega * (Q_omega(s, omega) - V_s(s) + deliberation_cost) )
    beta_omega: [B] termination probability in (0, 1)
    q_omega: [B] value of continuing option omega
    v_s: [B] baseline state value
    deliberation_cost: scalar penalty xi >= 0 preventing option collapse
    Returns: scalar loss.
    """
    advantage_cost = q_omega - v_s + deliberation_cost
    return torch.mean(beta_omega * advantage_cost)


def active_inference_free_energy(
    q_mean: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mean: torch.Tensor,
    p_log_std: torch.Tensor,
    obs: torch.Tensor,
    recon: torch.Tensor,
    obs_noise_var: float = 1.0,
) -> torch.Tensor:
    """Variational Free Energy F in Active Inference / POMDP:
    F = D_KL(q(s) || p(s)) - E_q[log p(o | s)]
    q_mean, q_log_std, p_mean, p_log_std: [B, D_s]
    obs, recon: [B, D_o]
    obs_noise_var: Gaussian observation variance
    Returns: [B] free energy per sample.
    """
    # 1. Latent KL divergence
    var_q = torch.exp(2.0 * q_log_std)
    var_p = torch.exp(2.0 * p_log_std)
    diff = q_mean - p_mean
    kl = 0.5 * torch.sum((var_q + diff ** 2) / var_p - 1.0 + 2.0 * (p_log_std - q_log_std), dim=-1)

    # 2. Observation negative log-likelihood (Gaussian)
    diff_obs = obs - recon
    nll = 0.5 * torch.sum(
        (diff_obs ** 2) / obs_noise_var + math.log(2.0 * math.pi * obs_noise_var),
        dim=-1,
    )
    return kl + nll


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. HER Relabeling
    states = torch.randn(8, 3)
    goals = states.clone()
    goals[0] += 0.02  # match (< 0.1)
    goals[1] += 0.05  # match (< 0.1)
    goals[2] += 0.50  # no match (> 0.1)
    goals[3] += 0.80  # no match (> 0.1)
    her_rewards_ref = her_goal_relabeling(states, goals, threshold=0.1)

    # 2. FeUdal Cosine Direction
    delta_s = torch.randn(8, 4)
    subgoal_g = torch.randn(8, 4)
    feudal_rewards_ref = feudal_directional_reward(delta_s, subgoal_g)

    # 3. Option-Critic Termination
    beta_omega = torch.rand(8) * 0.8 + 0.1
    q_omega = torch.randn(8) * 2.0
    v_s = torch.randn(8) * 2.0
    option_loss_ref = option_critic_termination_loss(beta_omega, q_omega, v_s, deliberation_cost=0.01)

    # 4. Active Inference Free Energy
    q_mean = torch.randn(8, 5) * 0.5
    q_log_std = torch.randn(8, 5) * 0.2 - 0.5
    p_mean = torch.randn(8, 5) * 0.5
    p_log_std = torch.randn(8, 5) * 0.2 - 0.5
    obs = torch.randn(8, 6)
    recon = torch.randn(8, 6)
    free_energy_ref = active_inference_free_energy(q_mean, q_log_std, p_mean, p_log_std, obs, recon, obs_noise_var=1.0)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "states": states,
        "goals": goals,
        "her_rewards_ref": her_rewards_ref,
        "delta_s": delta_s,
        "subgoal_g": subgoal_g,
        "feudal_rewards_ref": feudal_rewards_ref,
        "beta_omega": beta_omega,
        "q_omega": q_omega,
        "v_s": v_s,
        "option_loss_ref": option_loss_ref,
        "q_mean": q_mean,
        "q_log_std": q_log_std,
        "p_mean": p_mean,
        "p_log_std": p_log_std,
        "obs": obs,
        "recon": recon,
        "free_energy_ref": free_energy_ref,
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
