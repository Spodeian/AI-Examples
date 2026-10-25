"""Value Based RL Dqn Rainbow and Qr Dqn — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dueling_q_values(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """Dueling Network Q-value aggregation:
    Q(s, a) = V(s) + (A(s, a) - (1 / |A|) * sum_{a'} A(s, a')).
    value: [B, 1] state value stream.
    advantage: [B, A] action advantage stream.
    Returns: [B, A] combined Q-values.
    """
    mean_adv = torch.mean(advantage, dim=-1, keepdim=True)
    return value + (advantage - mean_adv)


def double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    q_next_online: torch.Tensor,
    q_next_target: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Double DQN Bellman target calculation:
    a* = argmax_{a'} Q_online(s', a')
    y = r + gamma * (1 - done) * Q_target(s', a*)
    rewards: [B]
    dones: [B] (0.0 or 1.0)
    q_next_online: [B, A]
    q_next_target: [B, A]
    Returns: [B] target values.
    """
    best_actions = torch.argmax(q_next_online, dim=-1, keepdim=True)  # [B, 1]
    q_target_selected = torch.gather(q_next_target, dim=-1, index=best_actions).squeeze(-1)  # [B]
    return rewards + gamma * (1.0 - dones) * q_target_selected


def per_loss(
    q_eval: torch.Tensor,
    targets: torch.Tensor,
    is_weights: torch.Tensor,
) -> torch.Tensor:
    """Prioritized Experience Replay (PER) weighted Huber / MSE loss:
    loss = (1 / B) * sum_i w_i * (q_eval_i - y_i)^2
    q_eval: [B]
    targets: [B]
    is_weights: [B] normalized importance sampling weights
    Returns: scalar loss.
    """
    td_errors = q_eval - targets
    weighted_sq_errors = is_weights * (td_errors ** 2)
    return torch.mean(weighted_sq_errors)


def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Regression DQN (QR-DQN) pairwise quantile Huber loss:
    current_quantiles: [B, N] predicted quantiles theta_i(s, a)
    target_quantiles: [B, N] target quantiles theta_j(s', a')
    tau: [N] quantile midpoints tau_i = (2i - 1) / (2N)
    kappa: Huber threshold
    Returns: scalar loss.
    """
    # Pairwise TD error: u_{i,j} = target_j - current_i -> shape [B, N, N]
    # target_quantiles: [B, 1, N]
    # current_quantiles: [B, N, 1]
    diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)  # [B, N, N]
    abs_diff = torch.abs(diff)

    # Huber loss
    huber = torch.where(
        abs_diff <= kappa,
        0.5 * (diff ** 2),
        kappa * (abs_diff - 0.5 * kappa),
    )

    # Quantile weight: |tau_i - I(diff < 0)|
    # tau: [1, N, 1]
    tau_expanded = tau.view(1, -1, 1)
    indicator = (diff < 0.0).float()
    weight = torch.abs(tau_expanded - indicator)

    elementwise_loss = weight * huber / kappa
    return torch.mean(elementwise_loss)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Dueling Q
    value = torch.randn(8, 1)
    advantage = torch.randn(8, 4)
    dueling_q_ref = dueling_q_values(value, advantage)

    # 2. Double DQN
    rewards = torch.randn(8) * 0.5 + 0.1
    dones = (torch.rand(8) > 0.8).float()
    q_next_online = torch.randn(8, 4)
    q_next_target = torch.randn(8, 4)
    ddqn_targets_ref = double_dqn_targets(rewards, dones, q_next_online, q_next_target, gamma=0.99)

    # 3. PER Loss
    q_eval = torch.randn(8)
    targets = ddqn_targets_ref.clone()
    is_weights = torch.rand(8) * 0.8 + 0.2
    is_weights = is_weights / torch.max(is_weights)
    per_loss_ref = per_loss(q_eval, targets, is_weights)

    # 4. QR-DQN
    num_quantiles = 8
    current_quantiles = torch.randn(8, num_quantiles)
    target_quantiles = torch.randn(8, num_quantiles)
    tau = (2.0 * torch.arange(num_quantiles, dtype=torch.float32) + 1.0) / (2.0 * num_quantiles)
    qr_loss_ref = quantile_huber_loss(current_quantiles, target_quantiles, tau, kappa=1.0)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "value": value,
        "advantage": advantage,
        "dueling_q_ref": dueling_q_ref,
        "rewards": rewards,
        "dones": dones,
        "q_next_online": q_next_online,
        "q_next_target": q_next_target,
        "ddqn_targets_ref": ddqn_targets_ref,
        "q_eval": q_eval,
        "targets": targets,
        "is_weights": is_weights,
        "per_loss_ref": per_loss_ref,
        "current_quantiles": current_quantiles,
        "target_quantiles": target_quantiles,
        "tau": tau,
        "qr_loss_ref": qr_loss_ref,
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
