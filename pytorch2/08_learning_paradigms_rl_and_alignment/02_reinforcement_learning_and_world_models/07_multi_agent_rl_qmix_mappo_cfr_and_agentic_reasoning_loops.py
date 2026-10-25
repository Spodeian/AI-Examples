"""Multi Agent RL QMIX Mappo Cfr and Agentic Reasoning Loops — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops
Golden weights: weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def qmix_monotonic_mixing(
    agent_qs: torch.Tensor,
    w1_raw: torch.Tensor,
    b1: torch.Tensor,
    w2_raw: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """QMIX monotonic hypernetwork mixing operator:
    agent_qs: [B, 1, N] individual agent utility values Q_i(s, a_i)
    w1_raw:   [B, N, H] hypernetwork 1 weights (enforced |w1| >= 0)
    b1:       [B, 1, H] bias 1
    w2_raw:   [B, H, 1] hypernetwork 2 weights (enforced |w2| >= 0)
    b2:       [B, 1, 1] bias 2
    Returns:  [B] mixed joint Q_tot value.
    """
    w1 = torch.abs(w1_raw)
    w2 = torch.abs(w2_raw)
    # Layer 1: [B, 1, N] @ [B, N, H] + [B, 1, H] -> [B, 1, H]
    hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
    # Layer 2: [B, 1, H] @ [B, H, 1] + [B, 1, 1] -> [B, 1, 1]
    q_tot = torch.bmm(hidden, w2) + b2
    return q_tot.squeeze(-1).squeeze(-1)


def mappo_centralized_gae(
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """Multi-Agent PPO (MAPPO) Centralized Generalized Advantage Estimation (GAE).
    values:     [B, T] centralized critic evaluations V(s_t)
    rewards:    [B, T] team shared rewards
    dones:      [B, T] episode termination masks
    next_value: [B] bootstrap value V(s_{T+1})
    Returns:    [B, T] centralized GAE advantages.
    """
    b, t = values.shape
    advantages = torch.zeros_like(values)
    gae = torch.zeros(b, dtype=values.dtype, device=values.device)

    for step in reversed(range(t)):
        if step == t - 1:
            next_v = next_value
        else:
            next_v = values[:, step + 1]

        non_terminal = 1.0 - dones[:, step]
        delta = rewards[:, step] + gamma * next_v * non_terminal - values[:, step]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[:, step] = gae

    return advantages


def cfr_regret_matching(
    cumulative_regrets: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Counterfactual Regret Minimization (CFR) regret matching.
    cumulative_regrets: [B, A] cumulative regret per action
    Returns: [B, A] normalized strategy profile probabilities in simplex Delta(A).
    """
    positive_regrets = torch.clamp(cumulative_regrets, min=0.0)
    sum_positive = torch.sum(positive_regrets, dim=-1, keepdim=True)
    num_actions = cumulative_regrets.shape[-1]
    uniform_prob = torch.full_like(cumulative_regrets, 1.0 / num_actions)

    strategy = torch.where(sum_positive > eps, positive_regrets / (sum_positive + eps), uniform_prob)
    return strategy


def agentic_consensus_metric(
    embeddings: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-agent deliberation / reasoning loop pairwise consensus metric.
    embeddings: [B, N, D] thought representations of N debating agents
    Returns: [B] average pairwise cosine similarity consensus score in [-1.0, 1.0].
    """
    b, n, d = embeddings.shape
    norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)  # [B, N, 1]
    normed = embeddings / torch.clamp(norms, min=eps)  # [B, N, D]
    sim_matrix = torch.bmm(normed, normed.transpose(1, 2))  # [B, N, N]

    # Exclude diagonal (self-similarity = 1.0)
    diag_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device).unsqueeze(0)
    off_diag_sims = sim_matrix.masked_select(~diag_mask).view(b, n * (n - 1))
    return torch.mean(off_diag_sims, dim=-1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. QMIX Monotonic Mixing Network
    # B=8, N=4 agents, H=8 mixing hidden dim
    agent_qs = torch.randn(8, 1, 4) * 2.0
    w1_raw = torch.randn(8, 4, 8)
    b1 = torch.randn(8, 1, 8) * 0.1
    w2_raw = torch.randn(8, 8, 1)
    b2 = torch.randn(8, 1, 1) * 0.1
    qmix_q_tot_ref = qmix_monotonic_mixing(agent_qs, w1_raw, b1, w2_raw, b2)

    # 2. MAPPO Centralized GAE
    # B=8, T=5 trajectory length
    values = torch.randn(8, 5) * 2.0
    rewards = torch.randn(8, 5) * 0.5
    dones = torch.zeros(8, 5)
    dones[1, 3] = 1.0
    dones[3, 2] = 1.0
    next_value = torch.randn(8) * 2.0
    mappo_advantages_ref = mappo_centralized_gae(values, rewards, dones, next_value, gamma=0.99, gae_lambda=0.95)

    # 3. CFR Regret Matching
    # B=8, A=4 actions
    cumulative_regrets = torch.randn(8, 4) * 5.0
    cumulative_regrets[0] = torch.tensor([-1.0, -2.0, -0.5, -3.0])  # all negative -> uniform fallback
    cfr_strategy_ref = cfr_regret_matching(cumulative_regrets)

    # 4. Agentic Reasoning Loop Consensus Metric
    # B=8, N=3 debating agents, D=6 thought embedding dimension
    embeddings = torch.randn(8, 3, 6)
    consensus_ref = agentic_consensus_metric(embeddings)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "agent_qs": agent_qs,
        "w1_raw": w1_raw,
        "b1": b1,
        "w2_raw": w2_raw,
        "b2": b2,
        "qmix_q_tot_ref": qmix_q_tot_ref,
        "values": values,
        "rewards": rewards,
        "dones": dones,
        "next_value": next_value,
        "mappo_advantages_ref": mappo_advantages_ref,
        "cumulative_regrets": cumulative_regrets,
        "cfr_strategy_ref": cfr_strategy_ref,
        "embeddings": embeddings,
        "consensus_ref": consensus_ref,
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
