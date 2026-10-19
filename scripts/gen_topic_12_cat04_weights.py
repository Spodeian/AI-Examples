"""Generate golden reference safetensors checkpoint for Category 04 Topic 12:
05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed
in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation."""

import math
from pathlib import Path
import torch
from safetensors.torch import save_file


def compute_sha_top_k(losses: torch.Tensor, eta: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top k = n // eta candidates with lowest loss."""
    k = max(1, losses.shape[0] // eta)
    sorted_losses, sorted_indices = torch.sort(losses)
    return sorted_indices[:k], sorted_losses[:k]


def compute_hyperband_schedule(R: int = 81, eta: int = 3) -> torch.Tensor:
    """Compute (n_s, r_s) pairs across brackets s in [s_max, ..., 0]."""
    s_max = int(math.floor(math.log(R, eta)))
    B = (s_max + 1) * R
    schedule = []
    for s in range(s_max, -1, -1):
        n_s = int(math.ceil((B / R) * ((eta ** s) / (s + 1))))
        r_s = int(R * (eta ** (-s)))
        schedule.append([float(n_s), float(r_s)])
    return torch.tensor(schedule, dtype=torch.float32)


def compute_tpe_log_ratio(
    x_query: torch.Tensor,
    good_samples: torch.Tensor,
    bad_samples: torch.Tensor,
    bandwidth: float = 0.5,
) -> torch.Tensor:
    """Evaluate 1D TPE log-likelihood ratio: log l(x) - log g(x)."""
    def log_kde(x, samples):
        # diff: [m, n_samples]
        diff = (x.unsqueeze(-1) - samples.unsqueeze(0)) / bandwidth
        log_kernel = -0.5 * diff ** 2 - 0.5 * math.log(2.0 * math.pi) - math.log(bandwidth)
        return torch.logsumexp(log_kernel, dim=-1) - math.log(samples.shape[0])

    log_l = log_kde(x_query, good_samples)
    log_g = log_kde(x_query, bad_samples)
    return log_l - log_g


def check_asha_promotion(rung_losses: torch.Tensor, candidate_loss: float, eta: int = 3) -> float:
    """Return 1.0 if candidate_loss is eligible for promotion (in top 1/eta), else 0.0."""
    k = max(1, rung_losses.shape[0] // eta)
    sorted_losses, _ = torch.sort(rung_losses)
    cutoff = sorted_losses[k - 1].item()
    return 1.0 if candidate_loss <= cutoff else 0.0


def main():
    torch.manual_seed(42)

    # 1. Successive Halving Algorithm (SHA)
    sha_losses = torch.tensor([0.45, 0.12, 0.88, 0.23, 0.05, 0.67, 0.31, 0.92, 0.18], dtype=torch.float32)
    golden_sha_idx, golden_sha_losses = compute_sha_top_k(sha_losses, eta=3)

    # 2. Hyperband Allocation Schedule
    golden_hb_schedule = compute_hyperband_schedule(R=81, eta=3)

    # 3. TPE Log Ratio
    x_query = torch.linspace(-2.0, 2.0, 5, dtype=torch.float32)
    good_samples = torch.tensor([-1.2, -0.8, -1.0], dtype=torch.float32)
    bad_samples = torch.tensor([0.5, 1.2, 1.8, 0.2, 1.0], dtype=torch.float32)
    golden_tpe_log_ratio = compute_tpe_log_ratio(x_query, good_samples, bad_samples, bandwidth=0.5)

    # 4. ASHA Promotion
    rung_losses = torch.tensor([0.15, 0.42, 0.08, 0.33, 0.61, 0.25], dtype=torch.float32)
    cand_loss_eligible = 0.12  # in top 2 (top 2 are 0.08, 0.15) -> eligible!
    cand_loss_ineligible = 0.35  # not in top 2 -> ineligible
    golden_asha_promoted = torch.tensor([
        check_asha_promotion(rung_losses, cand_loss_eligible, eta=3),
        check_asha_promotion(rung_losses, cand_loss_ineligible, eta=3),
    ], dtype=torch.float32)

    # 5. Linear Anchor for cross-track parity
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    tensors = {
        "sha_losses": sha_losses,
        "golden_sha_idx": golden_sha_idx.to(torch.float32),
        "golden_sha_losses": golden_sha_losses,
        "golden_hb_schedule": golden_hb_schedule,
        "x_query": x_query,
        "good_samples": good_samples,
        "bad_samples": bad_samples,
        "golden_tpe_log_ratio": golden_tpe_log_ratio,
        "rung_losses": rung_losses,
        "candidate_losses": torch.tensor([cand_loss_eligible, cand_loss_ineligible], dtype=torch.float32),
        "golden_asha_promoted": golden_asha_promoted,
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
    out_file = out_dir / "05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed.safetensors"
    save_file(tensors, str(out_file))
    print(f"Saved {len(tensors)} tensors to {out_file}")


if __name__ == "__main__":
    main()
