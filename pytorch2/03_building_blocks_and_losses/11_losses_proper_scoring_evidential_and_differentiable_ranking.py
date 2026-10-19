"""Proper Scoring Rules, Evidential Deep Learning, and Differentiable Ranking (C03 Topic 11).

Implements uncertainty quantification, calibrated scoring rules, and ranking objectives:
1. Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions (Matheson & Winkler 1976).
2. Brier Score: Proper scoring rule for categorical probability forecasts (Brier 1950).
3. Beta-NLL: Variance-discounted heteroscedastic regression loss (Seitzer et al. 2022).
4. Evidential Deep Learning (EDL MSE): Closed-form Dirichlet expected square error (Sensoy et al. 2018).
5. ListMLE Ranking Loss: Permutation probability objective over item lists (Xia et al. 2008).
6. Soft-Spearman Differentiable Ranks via pairwise sigmoid relaxation.
7. Linear Anchor: Baseline affine transform.
"""

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file

try:
    from jaxtyping import Float, Int
except ImportError:
    class _TypeShim:
        def __getitem__(self, item: Any) -> Any:
            return Any
    Float = _TypeShim()
    Int = _TypeShim()


def compute_crps_gaussian(
    y: Float[torch.Tensor, "batch 1"],
    mu: Float[torch.Tensor, "batch 1"],
    sigma: Float[torch.Tensor, "batch 1"],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Continuous Ranked Probability Score for Gaussian predictions: CRPS(y, N(mu, sigma^2))."""
    sig = sigma.clamp(min=eps)
    z = (y - mu) / sig
    phi = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sig * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps.mean()


def compute_brier_score(
    p: Float[torch.Tensor, "batch num_classes"],
    y_onehot: Float[torch.Tensor, "batch num_classes"],
) -> torch.Tensor:
    """Brier Score: mean squared error between forecast probabilities and one-hot labels."""
    return torch.mean(torch.sum((p - y_onehot).pow(2), dim=-1))


def compute_beta_nll(
    y: Float[torch.Tensor, "batch 1"],
    mu: Float[torch.Tensor, "batch 1"],
    var: Float[torch.Tensor, "batch 1"],
    beta: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Beta-NLL: (var^beta) * [0.5 * log(var) + 0.5 * (y - mu)^2 / var]."""
    v = var.clamp(min=eps)
    loss = 0.5 * (torch.log(v) + (y - mu).pow(2) / v)
    weight = (v.detach()).pow(beta)
    return torch.mean(weight * loss)


def compute_edl_mse(
    alpha: Float[torch.Tensor, "batch num_classes"],
    y_onehot: Float[torch.Tensor, "batch num_classes"],
) -> torch.Tensor:
    """Evidential Deep Learning Expected MSE under Dirichlet distribution: sum (y - p_hat)^2 + Var(p)."""
    S = torch.sum(alpha, dim=-1, keepdim=True)
    p_hat = alpha / S
    var_p = (alpha * (S - alpha)) / (S.pow(2) * (S + 1.0))
    loss = torch.sum((y_onehot - p_hat).pow(2) + var_p, dim=-1)
    return loss.mean()


def compute_listmle(
    scores: Float[torch.Tensor, "batch num_items"],
    perm: Int[torch.Tensor, "batch num_items"],
) -> torch.Tensor:
    """ListMLE Ranking Loss: negative log-likelihood of ground-truth item permutation."""
    B, K = scores.shape
    total_loss = 0.0
    for b in range(B):
        s = scores[b]
        p = perm[b]
        s_ordered = s[p]
        for i in range(K):
            sub_scores = s_ordered[i:]
            log_denom = torch.logsumexp(sub_scores, dim=0)
            total_loss = total_loss - (s_ordered[i] - log_denom)
    return total_loss / float(B * K)


def compute_soft_ranks(
    scores: Float[torch.Tensor, "batch num_items"],
    tau: float = 0.5,
) -> torch.Tensor:
    """Differentiable soft ranks via pairwise sigmoid relaxation: 0.5 + sum_j sigmoid((s_j - s_i)/tau)."""
    diff = (scores.unsqueeze(2) - scores.unsqueeze(1)) / tau
    return 0.5 + torch.sum(torch.sigmoid(diff), dim=-1)


class LinearAnchor(nn.Module):
    """Linear projection layer serving as parity anchor."""

    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class ScoringRankingUnifiedModule(nn.Module):
    """Unified module combining scoring rules, evidential loss, and ranking objectives."""

    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.anchor = LinearAnchor(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.anchor(x)


def load_checkpoint_tensors(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    return load_file(str(checkpoint_path))


if __name__ == "__main__":
    ckpt = Path("weights/03_building_blocks_and_losses/11_losses_proper_scoring_evidential_and_differentiable_ranking.safetensors")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt}")

    data = load_checkpoint_tensors(ckpt)
    print(f"Loaded {len(data)} tensors from {ckpt.name}")

    # 1. Linear Anchor parity
    mod = ScoringRankingUnifiedModule()
    mod.anchor.weight.data.copy_(data["weight"])
    mod.anchor.bias.data.copy_(data["bias"])
    pred = mod(data["test_input"])
    err_anchor = float(torch.max(torch.abs(pred - data["test_output"])).item())
    assert err_anchor < 1e-5, f"Anchor parity error {err_anchor:.2e} >= 1e-5"

    # 2. CRPS parity
    crps_val = compute_crps_gaussian(data["crps_y"], data["crps_mu"], data["crps_sigma"])
    err_crps = float(torch.abs(crps_val - data["expected_crps"]).item())
    assert err_crps < 1e-5, f"CRPS parity error {err_crps:.2e} >= 1e-5"

    # 3. Brier Score parity
    brier_val = compute_brier_score(data["brier_p"], data["brier_y"])
    err_brier = float(torch.abs(brier_val - data["expected_brier"]).item())
    assert err_brier < 1e-5, f"Brier parity error {err_brier:.2e} >= 1e-5"

    # 4. Beta-NLL parity
    beta_nll_val = compute_beta_nll(data["beta_nll_y"], data["beta_nll_mu"], data["beta_nll_var"], beta=0.5)
    err_beta_nll = float(torch.abs(beta_nll_val - data["expected_beta_nll"]).item())
    assert err_beta_nll < 1e-5, f"Beta-NLL parity error {err_beta_nll:.2e} >= 1e-5"

    # 5. EDL MSE parity
    edl_val = compute_edl_mse(data["edl_alpha"], data["edl_y"])
    err_edl = float(torch.abs(edl_val - data["expected_edl"]).item())
    assert err_edl < 1e-5, f"EDL parity error {err_edl:.2e} >= 1e-5"

    # 6. ListMLE parity
    listmle_val = compute_listmle(data["listmle_scores"], data["listmle_perm"])
    err_listmle = float(torch.abs(listmle_val - data["expected_listmle"]).item())
    assert err_listmle < 1e-5, f"ListMLE parity error {err_listmle:.2e} >= 1e-5"

    # 7. Soft Ranks parity
    ranks_val = compute_soft_ranks(data["soft_scores"], tau=0.5)
    err_ranks = float(torch.max(torch.abs(ranks_val - data["expected_soft_ranks"])).item())
    assert err_ranks < 1e-5, f"Soft ranks parity error {err_ranks:.2e} >= 1e-5"

    print(
        f"All 7 invariants verified! Anchor: {err_anchor:.2e}, CRPS: {err_crps:.2e}, Brier: {err_brier:.2e}, "
        f"BetaNLL: {err_beta_nll:.2e}, EDL: {err_edl:.2e}, ListMLE: {err_listmle:.2e}, SoftRanks: {err_ranks:.2e}"
    )
