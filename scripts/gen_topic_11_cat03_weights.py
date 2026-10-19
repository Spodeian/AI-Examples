"""Generate deterministic reference safetensors checkpoint for Category 03 Topic 11:
11_losses_proper_scoring_evidential_and_differentiable_ranking
"""

from __future__ import annotations

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


if __name__ == "__main__":
    generate_weights()
