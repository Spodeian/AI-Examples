"""GRPO Dapo and Verifiable Reward Reasoning — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning
Golden weights: weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def grpo_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Group Relative Policy Optimization (GRPO / DeepSeekMath & DeepSeek-R1) advantage estimator:
    Given a group of G sampled rollouts per prompt:
    mu_b = 1/G sum_{i=1}^G R_{b, i}
    std_b = sqrt( 1/G sum_{i=1}^G (R_{b, i} - mu_b)^2 + eps )
    A_{b, i} = (R_{b, i} - mu_b) / std_b
    rewards: [B, G] scalar reward for G sampled completions per prompt
    Returns: [B, G] group-normalized advantages.
    """
    mean = torch.mean(rewards, dim=-1, keepdim=True)
    std = torch.std(rewards, dim=-1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def k3_kl_divergence(
    log_probs_policy: torch.Tensor,
    log_probs_ref: torch.Tensor,
) -> torch.Tensor:
    """Schulman's k3 unbiased low-variance non-negative KL divergence estimator:
    Delta = ln pi_theta - ln pi_ref
    D_KL(pi_theta || pi_ref) = exp(-Delta) - 1 + Delta
    log_probs_policy: [B, G, T]
    log_probs_ref:    [B, G, T]
    Returns:          [B, G, T] elementwise KL divergence >= 0.
    """
    delta = log_probs_policy - log_probs_ref
    return torch.exp(-delta) - 1.0 + delta


def grpo_clipped_loss(
    log_probs_policy: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    kl_div: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """GRPO clipped surrogate objective with reference KL regularization:
    r_{i, t} = exp(ln pi_theta(t) - ln pi_old(t))
    L = - mean( min(r * A, clip(r, 1 - eps, 1 + eps) * A) - beta * D_KL )
    log_probs_policy: [B, G, T]
    log_probs_old:    [B, G, T]
    advantages:       [B, G] group normalized advantages
    kl_div:           [B, G, T] k3 KL divergence
    Returns:          scalar GRPO loss.
    """
    ratio = torch.exp(log_probs_policy - log_probs_old)
    adv = advantages.unsqueeze(-1)  # [B, G, 1]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    ppo_obj = torch.min(surr1, surr2)
    token_obj = ppo_obj - beta * kl_div
    return -torch.mean(token_obj)


def composite_verifiable_reward(
    accuracy_flags: torch.Tensor,
    format_flags: torch.Tensor,
    acc_weight: float = 1.0,
    format_weight: float = 0.5,
) -> torch.Tensor:
    """Verifiable Rule-Based Composite Reward for Reasoning (DeepSeek-R1):
    R = acc_weight * I(correct answer) + format_weight * I(valid <think> tags)
    accuracy_flags: [B, G] binary correctness
    format_flags:   [B, G] binary formatting compliance
    Returns:        [B, G] composite rule-based rewards.
    """
    return acc_weight * accuracy_flags + format_weight * format_flags


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Composite Verifiable Rewards
    # B=4 prompts, G=4 rollouts per prompt
    accuracy_flags = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    format_flags = torch.tensor([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ])
    rewards_ref = composite_verifiable_reward(accuracy_flags, format_flags, acc_weight=1.0, format_weight=0.5)

    # 2. GRPO Group Advantages
    advantages_ref = grpo_group_advantages(rewards_ref)

    # 3. K3 KL Divergence & GRPO Clipped Loss
    # B=4, G=4, T=6 tokens
    log_probs_policy = -torch.rand(4, 4, 6) * 3.0
    log_probs_old = log_probs_policy + torch.randn(4, 4, 6) * 0.05
    log_probs_ref = log_probs_policy + torch.randn(4, 4, 6) * 0.10

    kl_div_ref = k3_kl_divergence(log_probs_policy, log_probs_ref)
    grpo_loss_ref = grpo_clipped_loss(log_probs_policy, log_probs_old, advantages_ref, kl_div_ref, clip_eps=0.2, beta=0.04)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "accuracy_flags": accuracy_flags,
        "format_flags": format_flags,
        "rewards_ref": rewards_ref,
        "advantages_ref": advantages_ref,
        "log_probs_policy": log_probs_policy,
        "log_probs_old": log_probs_old,
        "log_probs_ref": log_probs_ref,
        "kl_div_ref": kl_div_ref,
        "grpo_loss_ref": grpo_loss_ref,
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
