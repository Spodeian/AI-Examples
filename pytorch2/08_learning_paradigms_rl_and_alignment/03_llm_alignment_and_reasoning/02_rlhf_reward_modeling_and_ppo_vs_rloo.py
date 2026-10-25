"""RLHF Reward Modeling and PPO vs Rloo — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo
Golden weights: weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def bradley_terry_reward_loss(
    rewards_chosen: torch.Tensor,
    rewards_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry preference model negative log-likelihood:
    L_RM(theta) = - E_{(x, y_w, y_l)} [ ln sigma(r_theta(x, y_w) - r_theta(x, y_l)) ]
    rewards_chosen:   [B] reward scores of winning completions
    rewards_rejected: [B] reward scores of losing completions
    Returns:          scalar preference loss.
    """
    diff = rewards_chosen - rewards_rejected
    return torch.mean(F.softplus(-diff))


def rlhf_token_level_rewards(
    log_probs_policy: torch.Tensor,
    log_probs_ref: torch.Tensor,
    terminal_scores: torch.Tensor,
    kl_coeff: float = 0.05,
) -> torch.Tensor:
    """RLHF Token-Level Reward shaping with reference policy KL penalty:
    r_t = - kl_coeff * (ln pi_theta(y_t) - ln pi_ref(y_t)) for t < T - 1
    r_{T-1} = terminal_score - kl_coeff * (ln pi_theta(y_{T-1}) - ln pi_ref(y_{T-1}))
    log_probs_policy: [B, T] log probs of generated tokens under active policy
    log_probs_ref:    [B, T] log probs of generated tokens under frozen reference
    terminal_scores:  [B] scalar reward model prediction on full sequence
    kl_coeff:         beta KL regularization multiplier
    Returns:          [B, T] shaped token reward trajectory for GAE.
    """
    token_kl = log_probs_policy - log_probs_ref  # [B, T]
    rewards = -kl_coeff * token_kl
    # Add terminal sequence reward model evaluation to the final token
    rewards[:, -1] += terminal_scores
    return rewards


def rloo_leave_one_out_advantages(
    scores: torch.Tensor,
) -> torch.Tensor:
    """REINFORCE Leave-One-Out (RLOO) baseline and advantage estimation:
    Given K sampled completions per prompt:
    b_i = (sum_{j=1}^K score_j - score_i) / (K - 1)
    A_i = score_i - b_i
    scores:  [B, K] sequence rewards (reward minus sequence KL)
    Returns: [B, K] zero-sum advantages satisfying sum_k A_{b, k} == 0.
    """
    b, k = scores.shape
    total_scores = torch.sum(scores, dim=-1, keepdim=True)  # [B, 1]
    baselines = (total_scores - scores) / float(k - 1)      # [B, K]
    return scores - baselines


def rloo_policy_gradient_loss(
    advantages: torch.Tensor,
    log_probs_sum: torch.Tensor,
) -> torch.Tensor:
    """RLOO REINFORCE policy gradient objective:
    L_RLOO = - 1/(B * K) sum_{b, k} A_{b, k} * sum_{t} ln pi(y_{b, k, t})
    advantages:    [B, K] Leave-One-Out advantages
    log_probs_sum: [B, K] sum of token log probabilities for each completion
    Returns:       scalar policy gradient loss.
    """
    return -torch.mean(advantages * log_probs_sum)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Bradley-Terry Reward Loss
    # B=8 pairs
    rewards_chosen = torch.randn(8) * 1.5 + 0.5
    rewards_rejected = torch.randn(8) * 1.5 - 0.5
    bt_loss_ref = bradley_terry_reward_loss(rewards_chosen, rewards_rejected)

    # 2. RLHF Token-Level Rewards
    # B=8, T=6 tokens
    log_probs_policy = -torch.rand(8, 6) * 3.0
    log_probs_ref = -torch.rand(8, 6) * 3.0
    terminal_scores = torch.randn(8) * 2.0
    rlhf_rewards_ref = rlhf_token_level_rewards(log_probs_policy, log_probs_ref, terminal_scores, kl_coeff=0.05)

    # 3. RLOO Advantages & Loss
    # B=8 prompts, K=4 completions per prompt
    scores = torch.randn(8, 4) * 2.0
    rloo_adv_ref = rloo_leave_one_out_advantages(scores)
    log_probs_sum = -torch.rand(8, 4) * 10.0
    rloo_loss_ref = rloo_policy_gradient_loss(rloo_adv_ref, log_probs_sum)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "rewards_chosen": rewards_chosen,
        "rewards_rejected": rewards_rejected,
        "bt_loss_ref": bt_loss_ref,
        "log_probs_policy": log_probs_policy,
        "log_probs_ref": log_probs_ref,
        "terminal_scores": terminal_scores,
        "rlhf_rewards_ref": rlhf_rewards_ref,
        "scores": scores,
        "rloo_adv_ref": rloo_adv_ref,
        "log_probs_sum": log_probs_sum,
        "rloo_loss_ref": rloo_loss_ref,
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
