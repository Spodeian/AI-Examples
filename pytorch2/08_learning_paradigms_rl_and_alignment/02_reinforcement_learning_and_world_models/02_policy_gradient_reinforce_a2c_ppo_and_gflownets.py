"""Policy Gradient Reinforce A2c PPO and GFlowNets — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ppo_clip_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO-Clip policy surrogate loss:
    r_t = exp(log_prob - old_log_prob)
    L_clip = - mean(min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t))
    log_probs: [B]
    old_log_probs: [B]
    advantages: [B]
    clip_eps: clipping threshold
    Returns: scalar loss.
    """
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    clipped_surrogate = torch.min(surr1, surr2)
    return -torch.mean(clipped_surrogate)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Generalized Advantage Estimation (GAE):
    delta_t = r_t + gamma * (1 - d_t) * V_{t+1} - V_t
    A_t = delta_t + gamma * lambda * (1 - d_t) * A_{t+1}
    rewards: [T]
    values: [T]
    dones: [T]
    next_value: scalar tensor
    Returns: advantages [T].
    """
    t_len = rewards.shape[0]
    advantages = torch.zeros(t_len, dtype=torch.float32, device=rewards.device)
    last_gae = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(t_len)):
        v_next = next_value if t == t_len - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * not_done * v_next - values[t]
        last_gae = delta + gamma * lambda_ * not_done * last_gae
        advantages[t] = last_gae

    return advantages


def gflownet_trajectory_balance_loss(
    log_z: torch.Tensor,
    log_pf_trajectory: torch.Tensor,
    log_pb_trajectory: torch.Tensor,
    reward: torch.Tensor,
) -> torch.Tensor:
    """GFlowNet Trajectory Balance (TB) objective:
    L_TB = mean( (log Z + sum_t log P_F(s_{t+1}|s_t) - log R(x) - sum_t log P_B(s_t|s_{t+1}))^2 )
    log_z: scalar tensor (log partition function)
    log_pf_trajectory: [B, T] forward transition log probabilities
    log_pb_trajectory: [B, T] backward transition log probabilities
    reward: [B] terminal state rewards R(x) > 0
    Returns: scalar loss.
    """
    sum_log_pf = torch.sum(log_pf_trajectory, dim=-1)  # [B]
    sum_log_pb = torch.sum(log_pb_trajectory, dim=-1)  # [B]
    log_r = torch.log(torch.clamp(reward, min=1e-8))    # [B]

    numerator = log_z + sum_log_pf
    denominator = log_r + sum_log_pb
    tb_error = numerator - denominator
    return torch.mean(tb_error ** 2)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. PPO Clip
    log_probs = torch.randn(8) * 0.2 - 0.5
    old_log_probs = torch.randn(8) * 0.2 - 0.5
    advantages = torch.randn(8) * 1.2
    ppo_loss_ref = ppo_clip_loss(log_probs, old_log_probs, advantages, clip_eps=0.2)

    # 2. GAE
    t_steps = 10
    rewards = torch.randn(t_steps) * 0.5 + 0.1
    values = torch.randn(t_steps) * 2.0
    dones = torch.zeros(t_steps)
    dones[4] = 1.0  # episode boundary mid-trajectory
    next_value = torch.tensor(1.5)
    gae_advantages_ref = compute_gae(rewards, values, dones, next_value, gamma=0.99, lambda_=0.95)

    # 3. GFlowNet Trajectory Balance
    b_size = 4
    t_len = 5
    log_z = torch.tensor(2.5)
    log_pf = torch.randn(b_size, t_len) * 0.5 - 1.0
    log_pb = torch.randn(b_size, t_len) * 0.5 - 1.0
    reward = torch.rand(b_size) * 5.0 + 0.5
    tb_loss_ref = gflownet_trajectory_balance_loss(log_z, log_pf, log_pb, reward)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "log_probs": log_probs,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "ppo_loss_ref": ppo_loss_ref,
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "next_value": next_value,
        "gae_advantages_ref": gae_advantages_ref,
        "log_z": log_z,
        "log_pf": log_pf,
        "log_pb": log_pb,
        "reward": reward,
        "tb_loss_ref": tb_loss_ref,
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
