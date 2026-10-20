"""Tractable Probabilistic Circuits (SPN), Copulas, Extreme Value Theory (EVT), and SBI in PyTorch 2.x.

Features:
- Tractable Sum-Product Networks (SPN) with smoothness and decomposability for exact marginals
- Archimedean Clayton Copula modeling tail dependence
- Extreme Value Theory (EVT) Generalized Pareto Distribution (GPD) for Peaks-Over-Threshold
- Simulation-Based Inference (SBI) neural density ratio estimation
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch


def evaluate_spn(
    leaves_p: torch.Tensor,      # [4, 2] Bernoulli parameter p(X_i = 1)
    leaf_evidence: torch.Tensor, # [4] Evidence: 1, 0, or -1 (marginalized)
    prod_weights: torch.Tensor,  # [2] Sum layer weights
) -> torch.Tensor:
    """Evaluates smooth and decomposable SPN over 4 variables in linear time O(|E|)."""
    leaf_vals = torch.zeros(4, 2, dtype=torch.float32, device=leaves_p.device)
    for i in range(4):
        ev = leaf_evidence[i].item()
        if ev == 1.0:
            leaf_vals[i] = leaves_p[i]
        elif ev == 0.0:
            leaf_vals[i] = 1.0 - leaves_p[i]
        else: # -1.0: marginalized out
            leaf_vals[i] = torch.tensor([1.0, 1.0], dtype=torch.float32, device=leaves_p.device)

    p0 = leaf_vals[0, 0] * leaf_vals[1, 0] * leaf_vals[2, 0] * leaf_vals[3, 0]
    p1 = leaf_vals[0, 1] * leaf_vals[1, 1] * leaf_vals[2, 1] * leaf_vals[3, 1]
    return prod_weights[0] * p0 + prod_weights[1] * p1


def clayton_copula_cdf(
    u: torch.Tensor,
    v: torch.Tensor,
    theta: float = 2.0,
) -> torch.Tensor:
    """Evaluates bivariate Clayton copula CDF: C(u, v) = max(u^(-theta) + v^(-theta) - 1, 0)^(-1/theta)."""
    inner = torch.clamp(u ** (-theta) + v ** (-theta) - 1.0, min=1e-7)
    return inner ** (-1.0 / theta)


def gpd_tail_cdf(
    y: torch.Tensor,
    xi: float = 0.25,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Evaluates Generalized Pareto Distribution (GPD) CDF for excesses y >= 0."""
    bracket = torch.clamp(1.0 + (xi * y) / sigma, min=1e-7)
    return 1.0 - bracket ** (-1.0 / xi)


def sbi_neural_ratio(
    x: torch.Tensor,
    theta: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates neural likelihood-to-marginal density ratio r(x, theta) = exp(logit)."""
    features = torch.cat([x, theta], dim=-1)
    logit = features @ w + b
    ratio = torch.exp(logit)
    return ratio, logit


class Topic13Model(nn.Module):
    """Linear mapping wrapper for parity verification against golden checkpoint."""
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


def verify_topic13_pytorch():
    ckpt_path = Path("weights/02_classical_and_probabilistic_ml/13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.safetensors")
    if not ckpt_path.exists():
        print("Checkpoint not found, skipping local PyTorch verification.")
        return

    ckpt = safetensors.torch.load_file(ckpt_path)

    # 1. SPN Verification
    leaves_p = ckpt["leaves_p"]
    prod_weights = ckpt["prod_weights"]
    ev_full = ckpt["ev_full"]
    ev_marg = ckpt["ev_marg"]
    spn_full = evaluate_spn(leaves_p, ev_full, prod_weights)
    spn_marg = evaluate_spn(leaves_p, ev_marg, prod_weights)
    assert torch.allclose(spn_full, ckpt["spn_prob_full"][0], atol=1e-5), "SPN full error"
    assert torch.allclose(spn_marg, ckpt["spn_prob_marg"][0], atol=1e-5), "SPN marg error"

    # 2. Copula Verification
    u = ckpt["u_grid"]
    v = ckpt["v_grid"]
    c_vals = clayton_copula_cdf(u, v, theta=2.0)
    assert torch.allclose(c_vals, ckpt["clayton_cdf"], atol=1e-5), "Clayton copula error"

    # 3. GPD EVT Verification
    y = ckpt["y_excesses"]
    gpd_vals = gpd_tail_cdf(y, xi=0.25, sigma=1.5)
    assert torch.allclose(gpd_vals, ckpt["gpd_cdf"], atol=1e-5), "GPD CDF error"

    # 4. SBI Ratio Verification
    x_sbi = ckpt["x_sbi"]
    theta_sbi = ckpt["theta_sbi"]
    w_sbi = ckpt["w_sbi"]
    b_sbi = ckpt["b_sbi"]
    ratio, logit = sbi_neural_ratio(x_sbi, theta_sbi, w_sbi, b_sbi)
    assert torch.allclose(ratio, ckpt["sbi_ratio"], atol=1e-5), "SBI ratio error"
    assert torch.allclose(logit, ckpt["sbi_logit"], atol=1e-5), "SBI logit error"

    print("All Topic 13 PyTorch 2.x verifications passed!")


if __name__ == "__main__":
    verify_topic13_pytorch()
