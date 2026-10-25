"""Successor Representations Forward Backward Zero Shot RL and Diayn — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def successor_features_td_error(
    psi_current: torch.Tensor,
    features: torch.Tensor,
    psi_next: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Temporal difference error for Successor Features:
    delta_psi = phi(s) + gamma * (1 - done) * psi(s', a') - psi(s, a)
    psi_current: [B, D]
    features:    [B, D] phi(s) cumulant features
    psi_next:    [B, D] psi(s', a')
    dones:       [B]
    Returns:     [B, D] successor feature TD error.
    """
    non_terminal = (1.0 - dones).unsqueeze(-1)
    target = features + gamma * non_terminal * psi_next
    return target - psi_current


def gpi_transfer_value(
    psi_policies: torch.Tensor,
    reward_weights: torch.Tensor,
) -> torch.Tensor:
    """Generalized Policy Improvement (GPI) across K candidate policies:
    Q_GPI(s, a) = max_{i=1..K} psi_i(s, a) . w
    psi_policies:   [B, K, D] successor features for K policies
    reward_weights: [D] task reward parameterization
    Returns:        [B] upper-envelope transfer Q-value.
    """
    # [B, K, D] @ [D] -> [B, K]
    q_candidates = torch.matmul(psi_policies, reward_weights)
    return torch.max(q_candidates, dim=-1)[0]


def fb_zero_shot_value(
    forward_map: torch.Tensor,
    task_embedding: torch.Tensor,
) -> torch.Tensor:
    """Forward-Backward (FB) Zero-Shot RL bilinear value evaluation:
    Q_FB(s, a, z) = <F(s, a, z), z>
    forward_map:    [B, D] F(s, a, z) forward representation
    task_embedding: [B, D] z task / goal embedding
    Returns:        [B] zero-shot predicted return.
    """
    return torch.sum(forward_map * task_embedding, dim=-1)


def diayn_skill_reward_and_loss(
    logits: torch.Tensor,
    skill_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diversity Is All You Need (DIAYN) mutual information intrinsic reward:
    r_DIAYN(s, z) = ln q_phi(z | s) - ln p(z) = ln q_phi(z | s) + ln K
    L_disc(phi)   = - 1/B sum ln q_phi(z | s)
    logits:       [B, K] discriminator logits for K discrete skills
    skill_labels: [B] ground truth integer skill indices in 0..K-1
    Returns:      ([B] intrinsic rewards, scalar discriminator cross-entropy loss).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    b = logits.shape[0]
    num_skills = logits.shape[-1]
    log_prior = -math.log(num_skills)

    # Gather log prob of selected skills
    skill_log_probs = log_probs.gather(dim=-1, index=skill_labels.unsqueeze(-1)).squeeze(-1)
    rewards = skill_log_probs - log_prior
    loss = -torch.mean(skill_log_probs)
    return rewards, loss


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Successor Features TD Error
    # B=8, D=6 feature dimension
    psi_current = torch.randn(8, 6)
    features = torch.randn(8, 6) * 0.5
    psi_next = torch.randn(8, 6)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    sf_td_error_ref = successor_features_td_error(psi_current, features, psi_next, dones, gamma=0.99)

    # 2. Generalized Policy Improvement (GPI) Transfer Value
    # B=8, K=4 policies, D=6
    psi_policies = torch.randn(8, 4, 6)
    reward_weights = torch.randn(6) * 1.5
    gpi_values_ref = gpi_transfer_value(psi_policies, reward_weights)

    # 3. Forward-Backward Zero-Shot Value
    # B=8, D=8 latent dimension
    forward_map = torch.randn(8, 8)
    task_embedding = torch.randn(8, 8)
    fb_values_ref = fb_zero_shot_value(forward_map, task_embedding)

    # 4. DIAYN Skill Intrinsic Reward and Loss
    # B=8, K=5 skills
    logits = torch.randn(8, 5) * 2.0
    skill_labels = torch.tensor([0, 2, 4, 1, 3, 0, 2, 4], dtype=torch.long)
    diayn_rewards_ref, diayn_loss_ref = diayn_skill_reward_and_loss(logits, skill_labels)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "psi_current": psi_current,
        "features": features,
        "psi_next": psi_next,
        "dones": dones,
        "sf_td_error_ref": sf_td_error_ref,
        "psi_policies": psi_policies,
        "reward_weights": reward_weights,
        "gpi_values_ref": gpi_values_ref,
        "forward_map": forward_map,
        "task_embedding": task_embedding,
        "fb_values_ref": fb_values_ref,
        "logits": logits,
        "skill_labels": skill_labels,
        "diayn_rewards_ref": diayn_rewards_ref,
        "diayn_loss_ref": diayn_loss_ref,
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
