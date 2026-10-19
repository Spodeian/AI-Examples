"""Generate golden reference safetensors checkpoint for Category 04 Topic 11:
04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn
in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation."""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import save_file


def run_sinkhorn(log_alpha: torch.Tensor, n_iters: int = 10, tau: float = 1.0) -> torch.Tensor:
    """Sinkhorn normalization over a square cost/score matrix."""
    P = torch.exp(log_alpha / tau)
    for _ in range(n_iters):
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-12)
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-12)
    return P


def step_cem(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    elite_frac: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-Entropy Method step: select elite candidates, update mean and std."""
    n = candidates.shape[0]
    n_elite = max(1, int(n * elite_frac))
    elite_indices = torch.topk(scores, n_elite, largest=False).indices
    elite_samples = candidates[elite_indices]
    new_mean = elite_samples.mean(dim=0)
    new_std = elite_samples.std(dim=0, unbiased=False) + 1e-6
    return new_mean, new_std


def sa_acceptance_prob(delta_e: torch.Tensor, temp: float) -> torch.Tensor:
    """Metropolis-Hastings acceptance probability: min(1, exp(-delta_e / T))."""
    return torch.clamp(torch.exp(-delta_e / temp), max=1.0)


def parallel_tempering_swap_prob(e1: float, e2: float, temp1: float, temp2: float) -> float:
    """Replica exchange swap probability: min(1, exp((beta1 - beta2) * (e1 - e2)))."""
    beta1 = 1.0 / temp1
    beta2 = 1.0 / temp2
    delta = (beta1 - beta2) * (e1 - e2)
    return float(torch.clamp(torch.exp(torch.tensor(delta, dtype=torch.float32)), max=1.0).item())


def main():
    torch.manual_seed(42)

    # 1. Gumbel-Sinkhorn Scores
    log_alpha = torch.tensor([
        [1.5, 0.2, -0.8, 0.1],
        [-0.5, 2.1, 0.4, -0.2],
        [0.3, -0.1, 1.8, 0.9],
        [-1.2, 0.5, 0.2, 1.7]
    ], dtype=torch.float32)

    golden_sinkhorn_out = run_sinkhorn(log_alpha, n_iters=10, tau=0.5)

    # 2. Cross-Entropy Method (CEM) Candidates
    candidates = torch.randn(20, 4, dtype=torch.float32)
    # Simple sphere function score: sum(x^2)
    scores = torch.sum(candidates ** 2, dim=-1)
    golden_cem_mean, golden_cem_std = step_cem(candidates, scores, elite_frac=0.25)

    # 3. Simulated Annealing acceptance checks
    delta_e = torch.tensor([0.5, -0.2, 2.0, 0.0, 1.2, -1.0], dtype=torch.float32)
    sa_temp = 1.5
    golden_sa_probs = sa_acceptance_prob(delta_e, sa_temp)

    # 4. Parallel Tempering Swap Probability
    e1, e2 = 3.5, 1.2
    t1, t2 = 1.0, 3.0
    golden_pt_swap_prob = torch.tensor([parallel_tempering_swap_prob(e1, e2, t1, t2)], dtype=torch.float32)

    # 5. Linear Anchor for cross-track parity
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    tensors = {
        "log_alpha": log_alpha,
        "golden_sinkhorn_out": golden_sinkhorn_out,
        "candidates": candidates,
        "scores": scores,
        "golden_cem_mean": golden_cem_mean,
        "golden_cem_std": golden_cem_std,
        "delta_e": delta_e,
        "sa_temp": torch.tensor([sa_temp], dtype=torch.float32),
        "golden_sa_probs": golden_sa_probs,
        "pt_e1": torch.tensor([e1], dtype=torch.float32),
        "pt_e2": torch.tensor([e2], dtype=torch.float32),
        "pt_t1": torch.tensor([t1], dtype=torch.float32),
        "pt_t2": torch.tensor([t2], dtype=torch.float32),
        "golden_pt_swap_prob": golden_pt_swap_prob,
        "anchor_x": anchor_x,
        "anchor_w": anchor_w,
        "anchor_b": anchor_b,
        "golden_anchor_out": golden_anchor_out,
        # Standard Parity keys
        "weight": anchor_w.clone(),
        "bias": anchor_b.clone(),
        "test_input": anchor_x.clone(),
        "test_output": golden_anchor_out.clone(),
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {len(tensors)} tensors to {out_path}")


if __name__ == "__main__":
    main()
