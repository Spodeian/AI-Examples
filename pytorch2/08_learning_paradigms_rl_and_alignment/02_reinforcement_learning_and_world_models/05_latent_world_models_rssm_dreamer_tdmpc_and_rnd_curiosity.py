"""Latent World Models Rssm Dreamer Tdmpc and Rnd Curiosity — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def gaussian_kl_divergence(
    mu_q: torch.Tensor,
    log_std_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_std_p: torch.Tensor,
) -> torch.Tensor:
    """Analytical KL divergence between diagonal Gaussians:
    D_KL(q || p) = 0.5 * sum_i ( (var_q + (mu_q - mu_p)^2) / var_p - 1 + 2*(log_std_p - log_std_q) )
    mu_q, log_std_q, mu_p, log_std_p: [B, D]
    Returns: [B] per-sample KL divergence.
    """
    var_q = torch.exp(2.0 * log_std_q)
    var_p = torch.exp(2.0 * log_std_p)
    diff = mu_q - mu_p
    kl_element = (var_q + diff ** 2) / var_p - 1.0 + 2.0 * (log_std_p - log_std_q)
    return 0.5 * torch.sum(kl_element, dim=-1)


def rssm_kl_balancing(
    mu_q: torch.Tensor,
    log_std_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_std_p: torch.Tensor,
    beta_prior: float = 0.8,
    beta_post: float = 0.2,
) -> torch.Tensor:
    """Dreamer / RSSM KL balancing objective:
    L_KL = beta_prior * mean(D_KL(sg[q] || p)) + beta_post * mean(D_KL(q || sg[p]))
    Returns: scalar loss.
    """
    kl_prior = gaussian_kl_divergence(mu_q.detach(), log_std_q.detach(), mu_p, log_std_p)
    kl_post = gaussian_kl_divergence(mu_q, log_std_q, mu_p.detach(), log_std_p.detach())
    return beta_prior * torch.mean(kl_prior) + beta_post * torch.mean(kl_post)


def rnd_intrinsic_reward(
    target_features: torch.Tensor,
    predictor_features: torch.Tensor,
) -> torch.Tensor:
    """Random Network Distillation (RND) intrinsic curiosity reward:
    r_i = mean_d ( (predictor_d - target_d)^2 )
    target_features: [B, D]
    predictor_features: [B, D]
    Returns: [B] intrinsic reward per sample.
    """
    return torch.mean((predictor_features - target_features) ** 2, dim=-1)


def tdmpc_latent_rollout_return(
    latent_rewards: torch.Tensor,
    terminal_q: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """TD-MPC2 latent imagination rollout return:
    G = sum_{t=0}^{H-1} gamma^t * r_t + gamma^H * Q(z_H)
    latent_rewards: [B, H]
    terminal_q: [B]
    Returns: [B] discounted rollout returns.
    """
    horizon = latent_rewards.shape[1]
    discounts = gamma ** torch.arange(horizon, dtype=torch.float32, device=latent_rewards.device)  # [H]
    discounted_rewards = torch.sum(latent_rewards * discounts.unsqueeze(0), dim=-1)  # [B]
    terminal_discount = gamma ** horizon
    return discounted_rewards + terminal_discount * terminal_q


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. RSSM KL Divergence & Balancing
    mu_q = torch.randn(8, 6) * 0.5
    log_std_q = torch.randn(8, 6) * 0.2 - 0.5
    mu_p = torch.randn(8, 6) * 0.5
    log_std_p = torch.randn(8, 6) * 0.2 - 0.5
    kl_div_ref = gaussian_kl_divergence(mu_q, log_std_q, mu_p, log_std_p)
    kl_balanced_ref = rssm_kl_balancing(mu_q, log_std_q, mu_p, log_std_p, beta_prior=0.8, beta_post=0.2)

    # 2. RND Curiosity
    target_features = torch.randn(8, 12)
    predictor_features = torch.randn(8, 12)
    rnd_rewards_ref = rnd_intrinsic_reward(target_features, predictor_features)

    # 3. TD-MPC2 Rollout Return
    latent_rewards = torch.randn(8, 5) * 0.5 + 0.1
    terminal_q = torch.randn(8) * 2.0
    tdmpc_return_ref = tdmpc_latent_rollout_return(latent_rewards, terminal_q, gamma=0.99)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "mu_q": mu_q,
        "log_std_q": log_std_q,
        "mu_p": mu_p,
        "log_std_p": log_std_p,
        "kl_div_ref": kl_div_ref,
        "kl_balanced_ref": kl_balanced_ref,
        "target_features": target_features,
        "predictor_features": predictor_features,
        "rnd_rewards_ref": rnd_rewards_ref,
        "latent_rewards": latent_rewards,
        "terminal_q": terminal_q,
        "tdmpc_return_ref": tdmpc_return_ref,
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
