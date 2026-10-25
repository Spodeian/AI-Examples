"""PyTorch 2.x implementation for Category 04 Topic 12:
Multi-Fidelity Hyperparameter Optimisation: Hyperband, ASHA, and BOED.

Provides production-grade implementations of:
1. Successive Halving Algorithm (SHA) multi-fidelity configuration pruning.
2. Hyperband bracket resource allocation schedule calculation.
3. Tree-structured Parzen Estimator (TPE) non-parametric density ratio surrogate.
4. Asynchronous Successive Halving (ASHA) non-blocking promotion logic.
"""

import math
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


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


class LinearAnchor(nn.Module):
    """Linear anchor layer for cross-track parity verification."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class MultiFidelityUnifiedModule(nn.Module):
    """Unified module encapsulating multi-fidelity HPO algorithms."""

    def __init__(self, weights_path: str | Path):
        super().__init__()
        weights = load_file(str(weights_path))

        self.sha_losses = weights["sha_losses"]
        self.golden_sha_idx = weights["golden_sha_idx"]
        self.golden_sha_losses = weights["golden_sha_losses"]
        self.golden_hb_schedule = weights["golden_hb_schedule"]

        self.x_query = weights["x_query"]
        self.good_samples = weights["good_samples"]
        self.bad_samples = weights["bad_samples"]
        self.golden_tpe_log_ratio = weights["golden_tpe_log_ratio"]

        self.rung_losses = weights["rung_losses"]
        self.candidate_losses = weights["candidate_losses"]
        self.golden_asha_promoted = weights["golden_asha_promoted"]

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verify parity against golden reference tensors."""
        errors: dict[str, float] = {}

        # 1. SHA Invariant
        idx, losses = compute_sha_top_k(self.sha_losses, eta=3)
        err_sha_idx = float(torch.max(torch.abs(idx.to(torch.float32) - self.golden_sha_idx)).item())
        err_sha_l = float(torch.max(torch.abs(losses - self.golden_sha_losses)).item())
        err_sha = max(err_sha_idx, err_sha_l)
        assert err_sha < atol, f"SHA invariant failed: err={err_sha:.2e} >= {atol:.2e}"
        errors["sha_err"] = err_sha

        # 2. Hyperband Schedule Invariant
        schedule = compute_hyperband_schedule(R=81, eta=3)
        err_hb = float(torch.max(torch.abs(schedule - self.golden_hb_schedule)).item())
        assert err_hb < atol, f"Hyperband invariant failed: err={err_hb:.2e} >= {atol:.2e}"
        errors["hyperband_err"] = err_hb

        # 3. TPE Log Ratio Invariant
        tpe_ratio = compute_tpe_log_ratio(self.x_query, self.good_samples, self.bad_samples, bandwidth=0.5)
        err_tpe = float(torch.max(torch.abs(tpe_ratio - self.golden_tpe_log_ratio)).item())
        assert err_tpe < atol, f"TPE invariant failed: err={err_tpe:.2e} >= {atol:.2e}"
        errors["tpe_err"] = err_tpe

        # 4. ASHA Invariant
        promoted_0 = check_asha_promotion(self.rung_losses, float(self.candidate_losses[0]), eta=3)
        promoted_1 = check_asha_promotion(self.rung_losses, float(self.candidate_losses[1]), eta=3)
        asha_promoted = torch.tensor([promoted_0, promoted_1], dtype=torch.float32)
        err_asha = float(torch.max(torch.abs(asha_promoted - self.golden_asha_promoted)).item())
        assert err_asha < atol, f"ASHA invariant failed: err={err_asha:.2e} >= {atol:.2e}"
        errors["asha_err"] = err_asha

        # 5. Linear Anchor Invariant
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
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


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed.safetensors")
    module = MultiFidelityUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
