"""Continuous Control Ddpg Td3 and SAC — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def td3_target_smoothing(
    target_actions: torch.Tensor,
    noise: torch.Tensor,
    noise_clip: float = 0.5,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> torch.Tensor:
    """TD3 Target Action Smoothing:
    a_smoothed = clip(a_target + clip(noise, -c, c), a_min, a_max)
    target_actions: [B, D]
    noise: [B, D]
    Returns: [B, D] smoothed target actions.
    """
    clipped_noise = torch.clamp(noise, -noise_clip, noise_clip)
    return torch.clamp(target_actions + clipped_noise, action_min, action_max)


def td3_clipped_critic_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    q1_target: torch.Tensor,
    q2_target: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """TD3 Clipped Double-Q Bellman Target:
    y = r + gamma * (1 - done) * min(Q1_target, Q2_target)
    rewards: [B]
    dones: [B]
    q1_target: [B]
    q2_target: [B]
    Returns: [B] target values.
    """
    min_q = torch.min(q1_target, q2_target)
    return rewards + gamma * (1.0 - dones) * min_q


def sac_squashed_gaussian_log_prob(
    u: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SAC Squashed Gaussian policy action and log-probability:
    a = tanh(u)
    log pi(a|s) = log N(u; mean, std^2) - sum_i log(1 - tanh(u_i)^2 + eps)
    u: [B, D] unsquashed Gaussian latent
    mean: [B, D]
    log_std: [B, D]
    Returns:
        actions: [B, D] in (-1, 1)
        log_probs: [B]
    """
    std = torch.exp(log_std)
    # Gaussian log-prob: -0.5 * sum_i ( ((u - mean)/std)^2 + 2*log_std + log(2*pi) )
    var = std ** 2
    log_scale = 2.0 * log_std
    gaussian_log_p = -0.5 * torch.sum(
        ((u - mean) ** 2) / var + log_scale + math.log(2.0 * math.pi),
        dim=-1,
    )  # [B]

    actions = torch.tanh(u)  # [B, D]
    # Jacobian correction term: sum_i log(1 - a_i^2 + eps)
    jacobian = torch.sum(torch.log(torch.clamp(1.0 - actions ** 2, min=eps)), dim=-1)  # [B]
    log_probs = gaussian_log_p - jacobian
    return actions, log_probs


def sac_temperature_loss(
    log_alpha: torch.Tensor,
    log_probs: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    """SAC Temperature Auto-Tuning Loss:
    L(alpha) = - mean( exp(log_alpha) * (log_pi(a|s) + target_entropy) )
    log_alpha: scalar tensor
    log_probs: [B]
    target_entropy: float scalar (typically -dim(A))
    Returns: scalar loss.
    """
    alpha = torch.exp(log_alpha)
    return -torch.mean(alpha * (log_probs + target_entropy))


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. TD3 Target Smoothing
    target_actions = torch.randn(8, 3) * 0.8
    noise = torch.randn(8, 3) * 0.3
    smoothed_actions_ref = td3_target_smoothing(target_actions, noise, noise_clip=0.5, action_min=-1.0, action_max=1.0)

    # 2. TD3 Clipped Critic Target
    rewards = torch.randn(8) * 0.5 + 0.1
    dones = (torch.rand(8) > 0.8).float()
    q1_target = torch.randn(8) * 3.0
    q2_target = torch.randn(8) * 3.0
    td3_targets_ref = td3_clipped_critic_target(rewards, dones, q1_target, q2_target, gamma=0.99)

    # 3. SAC Squashed Gaussian Policy
    u = torch.randn(8, 3)
    mean = torch.randn(8, 3) * 0.5
    log_std = torch.randn(8, 3) * 0.2 - 0.5
    sac_actions_ref, sac_log_probs_ref = sac_squashed_gaussian_log_prob(u, mean, log_std)

    # 4. SAC Alpha Temperature Loss
    log_alpha = torch.tensor(0.5)
    target_entropy = -3.0  # -dim(A)
    alpha_loss_ref = sac_temperature_loss(log_alpha, sac_log_probs_ref, target_entropy)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "target_actions": target_actions,
        "noise": noise,
        "smoothed_actions_ref": smoothed_actions_ref,
        "rewards": rewards,
        "dones": dones,
        "q1_target": q1_target,
        "q2_target": q2_target,
        "td3_targets_ref": td3_targets_ref,
        "u": u,
        "mean": mean,
        "log_std": log_std,
        "sac_actions_ref": sac_actions_ref,
        "sac_log_probs_ref": sac_log_probs_ref,
        "log_alpha": log_alpha,
        "alpha_loss_ref": alpha_loss_ref,
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
