"""Levels of Decision Making: Bayes Risk, Bandits, MDP, POMDP, and Games.

Implements the 6-level decision hierarchy from static Bayes risk with Chow's reject option,
through contextual bandit off-policy evaluation, closed-loop MDP Bellman backups,
POMDP belief-state filtering, to game-theoretic differentiable auctions.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_file


class BayesDecisionRule(nn.Module):
    """Level 0: Static Bayes decision rule with Chow's rejection option."""

    def __init__(self, cost_matrix: Float[torch.Tensor, "num_actions num_classes"]) -> None:
        super().__init__()
        self.register_buffer("cost_matrix", cost_matrix)

    def forward(
        self, probs: Float[torch.Tensor, "batch num_classes"], c_reject: float = 0.5
    ) -> Tuple[
        Float[torch.Tensor, "batch num_actions"],
        Float[torch.Tensor, "batch"],
        Float[torch.Tensor, "batch"],
    ]:
        # Expected risk for each action a: R(a|x) = sum_y C(a, y) * P(y|x)
        expected_risk = torch.matmul(probs, self.cost_matrix.t())
        best_actions = torch.argmin(expected_risk, dim=-1).to(torch.float32)
        min_risk = torch.min(expected_risk, dim=-1).values
        reject_mask = (min_risk > c_reject).to(torch.float32)
        return expected_risk, best_actions, reject_mask


class BanditsOPE(nn.Module):
    """Level 1: Contextual bandits off-policy evaluation (IPS, Doubly Robust)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        target_probs: Float[torch.Tensor, "batch"],
        logged_propensities: Float[torch.Tensor, "batch"],
        rewards: Float[torch.Tensor, "batch"],
        baseline_q: Float[torch.Tensor, "batch"],
    ) -> Tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        ips_weights = target_probs / logged_propensities
        doubly_robust = baseline_q + ips_weights * (rewards - baseline_q)
        return ips_weights, doubly_robust


class MDPBellmanOperator(nn.Module):
    """Level 2: Closed-loop MDP Bellman optimality backup operator."""

    def __init__(self, gamma: float = 0.9) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        v_current: Float[torch.Tensor, "num_states"],
        transitions: Float[torch.Tensor, "num_states num_actions num_states"],
        rewards: Float[torch.Tensor, "num_states num_actions"],
    ) -> Tuple[Float[torch.Tensor, "num_states num_actions"], Float[torch.Tensor, "num_states"]]:
        future_v = torch.einsum("sak,k->sa", transitions, v_current)
        q_values = rewards + self.gamma * future_v
        v_next = torch.max(q_values, dim=-1).values
        return q_values, v_next


class POMDPBeliefFilter(nn.Module):
    """Level 3: Bayesian belief-state filtering for POMDPs."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        prior_belief: Float[torch.Tensor, "num_states"],
        transition: Float[torch.Tensor, "num_states num_states"],
        observation_likelihood: Float[torch.Tensor, "num_states"],
    ) -> Float[torch.Tensor, "num_states"]:
        b_pred = torch.matmul(prior_belief, transition)
        b_unnorm = observation_likelihood * b_pred
        return b_unnorm / torch.sum(b_unnorm)


class DifferentiableAuction(nn.Module):
    """Level 5: Differentiable 2-bidder auction mechanism."""

    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self.tau = tau

    def forward(
        self, bids: Float[torch.Tensor, "batch num_bidders"]
    ) -> Tuple[Float[torch.Tensor, "batch num_bidders"], Float[torch.Tensor, "batch num_bidders"]]:
        allocations = torch.softmax(bids / self.tau, dim=-1)
        min_bid = torch.min(bids, dim=-1, keepdim=True).values
        payments = allocations * min_bid
        return allocations, payments


class DecisionHierarchyModule(nn.Module):
    """Top-level unified module encapsulating the 6-level decision hierarchy."""

    def __init__(self, cost_matrix: Float[torch.Tensor, "3 3"]) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.bayes = BayesDecisionRule(cost_matrix)
        self.ope = BanditsOPE()
        self.mdp = MDPBellmanOperator(gamma=0.9)
        self.pomdp = POMDPBeliefFilter()
        self.auction = DifferentiableAuction(tau=0.5)

    def forward(self, x: Float[torch.Tensor, "batch in_dim"]) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[DecisionHierarchyModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = DecisionHierarchyModule(cost_matrix=weights["l0_cost_matrix"])
    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])
    return module, weights


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.safetensors"
    )
    model, tensors = load_checkpoint(str(ckpt))

    # Verify linear anchor
    out_anchor = model(tensors["x_anchor"])
    assert torch.allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5), "Linear anchor mismatch"

    # Verify L0 Bayes decision
    l0_risk, l0_acts, l0_rej = model.bayes(tensors["l0_probs"])
    assert torch.allclose(l0_risk, tensors["l0_expected_risk"], atol=1e-5), "L0 risk mismatch"
    assert torch.allclose(l0_acts, tensors["l0_best_actions"], atol=1e-5), "L0 actions mismatch"
    assert torch.allclose(l0_rej, tensors["l0_reject_mask"], atol=1e-5), "L0 reject mask mismatch"

    # Verify L1 Bandits OPE
    l1_w, l1_dr = model.ope(
        tensors["l1_target_policy_probs"],
        tensors["l1_logged_propensities"],
        tensors["l1_rewards"],
        tensors["l1_baseline_q"],
    )
    assert torch.allclose(l1_w, tensors["l1_ips_weights"], atol=1e-5), "L1 weights mismatch"
    assert torch.allclose(l1_dr, tensors["l1_doubly_robust_estimates"], atol=1e-5), "L1 DR mismatch"

    # Verify L2 MDP Bellman
    l2_q, l2_v = model.mdp(tensors["l2_v_init"], tensors["l2_transitions"], tensors["l2_rewards"])
    assert torch.allclose(l2_q, tensors["l2_q_values"], atol=1e-5), "L2 Q mismatch"
    assert torch.allclose(l2_v, tensors["l2_v_backup"], atol=1e-5), "L2 V mismatch"

    # Verify L3 POMDP Belief
    l3_post = model.pomdp(
        tensors["l3_prior_belief"], tensors["l3_transition"], tensors["l3_observation_likelihood"]
    )
    assert torch.allclose(l3_post, tensors["l3_posterior_belief"], atol=1e-5), "L3 posterior mismatch"

    # Verify L5 Auction
    l5_alloc, l5_pay = model.auction(tensors["l5_bids"])
    assert torch.allclose(l5_alloc, tensors["l5_allocations"], atol=1e-5), "L5 alloc mismatch"
    assert torch.allclose(l5_pay, tensors["l5_payments"], atol=1e-5), "L5 pay mismatch"

    print("PyTorch 2 Decision Hierarchy successfully verified with zero errors!")
