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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import torch
    from safetensors.torch import save_file

    import core


    def compute_crps_gaussian(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        sig = sigma.clamp(min=eps)
        z = (y - mu) / sig
        phi = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        crps = sig * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
        return crps.mean()


    def compute_brier(p: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum((p - y_onehot).pow(2), dim=-1))


    def compute_beta_nll(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, beta: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
        v = var.clamp(min=eps)
        loss = 0.5 * (torch.log(v) + (y - mu).pow(2) / v)
        weight = (v.detach()).pow(beta)
        return torch.mean(weight * loss)


    def compute_edl_mse(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        S = torch.sum(alpha, dim=-1, keepdim=True)
        p_hat = alpha / S
        var_p = (alpha * (S - alpha)) / (S.pow(2) * (S + 1.0))
        loss = torch.sum((y_onehot - p_hat).pow(2) + var_p, dim=-1)
        return loss.mean()


    def compute_listmle(scores: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
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


    def compute_soft_ranks(scores: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        diff = (scores.unsqueeze(2) - scores.unsqueeze(1)) / tau
        return 0.5 + torch.sum(torch.sigmoid(diff), dim=-1)


    def generate_weights() -> None:
        core.setup_env(seed=42)
        gen = torch.Generator().manual_seed(42)

        batch_size = 8
        in_dim = 4
        out_dim = 4

        # 1. Linear Anchor
        weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
        bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)
        test_input = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
        test_output = test_input @ weight + bias

        # 2. CRPS
        crps_y = torch.randn(batch_size, 1, generator=gen, dtype=torch.float32)
        crps_mu = torch.randn(batch_size, 1, generator=gen, dtype=torch.float32)
        crps_sigma = torch.exp(torch.randn(batch_size, 1, generator=gen, dtype=torch.float32) * 0.5)
        expected_crps = compute_crps_gaussian(crps_y, crps_mu, crps_sigma)

        # 3. Brier
        raw_logits = torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)
        brier_p = torch.softmax(raw_logits, dim=-1)
        classes = torch.randint(0, out_dim, (batch_size,), generator=gen)
        brier_y = torch.zeros(batch_size, out_dim, dtype=torch.float32)
        brier_y[torch.arange(batch_size), classes] = 1.0
        expected_brier = compute_brier(brier_p, brier_y)

        # 4. Beta-NLL
        beta_nll_y = torch.randn(batch_size, 1, generator=gen, dtype=torch.float32)
        beta_nll_mu = torch.randn(batch_size, 1, generator=gen, dtype=torch.float32)
        beta_nll_var = torch.exp(torch.randn(batch_size, 1, generator=gen, dtype=torch.float32))
        expected_beta_nll = compute_beta_nll(beta_nll_y, beta_nll_mu, beta_nll_var, beta=0.5)

        # 5. Evidential Deep Learning MSE
        evidence = torch.relu(torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)) + 0.1
        edl_alpha = evidence + 1.0
        edl_y = brier_y.clone()
        expected_edl = compute_edl_mse(edl_alpha, edl_y)

        # 6. ListMLE Ranking
        listmle_scores = torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)
        # Permutation: sorted descending order indices of true relevances
        perm_list = []
        for _ in range(batch_size):
            p = torch.randperm(out_dim, generator=gen)
            perm_list.append(p)
        listmle_perm = torch.stack(perm_list, dim=0) # [8, 4] int64
        expected_listmle = compute_listmle(listmle_scores, listmle_perm)

        # 7. Soft Ranks
        soft_scores = torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)
        expected_soft_ranks = compute_soft_ranks(soft_scores, tau=0.5)

        tensors = {
            "weight": weight,
            "bias": bias,
            "test_input": test_input,
            "test_output": test_output,
            "crps_y": crps_y,
            "crps_mu": crps_mu,
            "crps_sigma": crps_sigma,
            "expected_crps": expected_crps.reshape(1),
            "brier_p": brier_p,
            "brier_y": brier_y,
            "expected_brier": expected_brier.reshape(1),
            "beta_nll_y": beta_nll_y,
            "beta_nll_mu": beta_nll_mu,
            "beta_nll_var": beta_nll_var,
            "expected_beta_nll": expected_beta_nll.reshape(1),
            "edl_alpha": edl_alpha,
            "edl_y": edl_y,
            "expected_edl": expected_edl.reshape(1),
            "listmle_scores": listmle_scores,
            "listmle_perm": listmle_perm,
            "expected_listmle": expected_listmle.reshape(1),
            "soft_scores": soft_scores,
            "expected_soft_ranks": expected_soft_ranks,
        }

        out_dir = Path("weights/03_building_blocks_and_losses")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "11_losses_proper_scoring_evidential_and_differentiable_ranking.safetensors"
        save_file(tensors, out_path)
        print(f"Saved {len(tensors)} tensors to {out_path}")


    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

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
