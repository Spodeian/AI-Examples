"""Generates reference SafeTensors checkpoint for Category 02 Topic 13:
13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.safetensors
"""

from __future__ import annotations
from pathlib import Path
import torch
import safetensors.torch


def evaluate_spn(
    leaves_p: torch.Tensor,     # [4, 2] Bernoulli p(X_i = 1) for 2 leaf components
    leaf_evidence: torch.Tensor,# [4] observed values: 0, 1, or -1 (unobserved / marginalized)
    prod_weights: torch.Tensor, # [2] weights for sum layer
) -> torch.Tensor:
    """Evaluates smooth & decomposable Sum-Product Network (SPN) over 4 variables.
    
    Structure:
    - Leaves: 4 variables, each with 2 Bernoulli components.
      If evidence == 1: p = leaves_p[i, k]
      If evidence == 0: p = 1.0 - leaves_p[i, k]
      If evidence == -1: p = 1.0 (marginalized out / unobserved)
    - Product layer:
      P_0 = L[0, 0] * L[1, 0] * L[2, 0] * L[3, 0] (decomposable: disjoint scopes)
      P_1 = L[0, 1] * L[1, 1] * L[2, 1] * L[3, 1] (decomposable: disjoint scopes)
    - Sum layer:
      S = w[0] * P_0 + w[1] * P_1 (smooth: identical scopes {0,1,2,3})
    """
    leaf_vals = torch.zeros(4, 2, dtype=torch.float32)
    for i in range(4):
        ev = leaf_evidence[i].item()
        if ev == 1.0:
            leaf_vals[i] = leaves_p[i]
        elif ev == 0.0:
            leaf_vals[i] = 1.0 - leaves_p[i]
        else: # -1: marginalized out
            leaf_vals[i] = torch.tensor([1.0, 1.0], dtype=torch.float32)

    p0 = leaf_vals[0, 0] * leaf_vals[1, 0] * leaf_vals[2, 0] * leaf_vals[3, 0]
    p1 = leaf_vals[0, 1] * leaf_vals[1, 1] * leaf_vals[2, 1] * leaf_vals[3, 1]
    spn_out = prod_weights[0] * p0 + prod_weights[1] * p1
    return spn_out


def clayton_copula_cdf(u: torch.Tensor, v: torch.Tensor, theta: float) -> torch.Tensor:
    """Computes bivariate Clayton copula CDF: C(u, v) = max(u^(-theta) + v^(-theta) - 1, 0)^(-1/theta)."""
    inner = torch.clamp(u ** (-theta) + v ** (-theta) - 1.0, min=1e-7)
    return inner ** (-1.0 / theta)


def gpd_tail_cdf(y: torch.Tensor, xi: float, sigma: float) -> torch.Tensor:
    """Computes Generalized Pareto Distribution (GPD) CDF for excesses y >= 0:
    F(y) = 1 - (1 + xi * y / sigma)^(-1/xi)
    """
    bracket = torch.clamp(1.0 + (xi * y) / sigma, min=1e-7)
    return 1.0 - bracket ** (-1.0 / xi)


def sbi_neural_ratio(x: torch.Tensor, theta: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes binary classifier logit and likelihood-to-marginal density ratio r(x, theta)."""
    # x: [B, D_x], theta: [B, D_theta]
    features = torch.cat([x, theta], dim=-1)
    logit = features @ w + b
    # r(x, theta) = exp(logit)
    ratio = torch.exp(logit)
    return ratio, logit


def generate_checkpoint():
    torch.manual_seed(42)

    # 1. SPN reference tensors
    leaves_p = torch.tensor([
        [0.8, 0.2],  # X_0
        [0.7, 0.4],  # X_1
        [0.3, 0.9],  # X_2
        [0.1, 0.6],  # X_3
    ], dtype=torch.float32)
    prod_weights = torch.tensor([0.6, 0.4], dtype=torch.float32)

    # Full evidence: X = [1, 0, 1, 0]
    ev_full = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    spn_prob_full = evaluate_spn(leaves_p, ev_full, prod_weights)

    # Marginal evidence: X_0 = 1, X_2 = 1, X_1 and X_3 marginalized out (-1)
    ev_marg = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
    spn_prob_marg = evaluate_spn(leaves_p, ev_marg, prod_weights)

    # 2. Clayton Copula reference
    u_grid = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
    v_grid = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.95], dtype=torch.float32)
    theta_copula = 2.0
    clayton_cdf = clayton_copula_cdf(u_grid, v_grid, theta_copula)

    # 3. GPD Peaks-Over-Threshold (EVT)
    y_excesses = torch.tensor([0.2, 0.5, 1.0, 2.0, 5.0], dtype=torch.float32)
    xi_gpd = 0.25
    sigma_gpd = 1.5
    gpd_cdf = gpd_tail_cdf(y_excesses, xi_gpd, sigma_gpd)

    # 4. SBI Neural Ratio
    w_sbi = torch.tensor([0.5, -0.8, 0.3, 0.6], dtype=torch.float32).unsqueeze(1)
    b_sbi = torch.tensor([-0.1], dtype=torch.float32)
    x_sbi = torch.tensor([[1.2, -0.4], [0.5, 1.1], [-0.8, 0.2]], dtype=torch.float32)
    theta_sbi = torch.tensor([[0.3, 0.7], [-0.5, 0.2], [1.0, -0.1]], dtype=torch.float32)
    sbi_ratio, sbi_logit = sbi_neural_ratio(x_sbi, theta_sbi, w_sbi, b_sbi)

    # 5. Golden linear mapping for test_cross_track_parity.py
    D_core = 4
    test_input = torch.randn(8, D_core)
    weight = torch.randn(D_core, D_core)
    bias = torch.randn(D_core)
    test_output = test_input @ weight + bias

    checkpoint_data = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "leaves_p": leaves_p.clone().contiguous(),
        "prod_weights": prod_weights.clone().contiguous(),
        "ev_full": ev_full.clone().contiguous(),
        "spn_prob_full": spn_prob_full.unsqueeze(0).clone().contiguous(),
        "ev_marg": ev_marg.clone().contiguous(),
        "spn_prob_marg": spn_prob_marg.unsqueeze(0).clone().contiguous(),
        "u_grid": u_grid.clone().contiguous(),
        "v_grid": v_grid.clone().contiguous(),
        "clayton_cdf": clayton_cdf.clone().contiguous(),
        "y_excesses": y_excesses.clone().contiguous(),
        "gpd_cdf": gpd_cdf.clone().contiguous(),
        "w_sbi": w_sbi.clone().contiguous(),
        "b_sbi": b_sbi.clone().contiguous(),
        "x_sbi": x_sbi.clone().contiguous(),
        "theta_sbi": theta_sbi.clone().contiguous(),
        "sbi_ratio": sbi_ratio.clone().contiguous(),
        "sbi_logit": sbi_logit.clone().contiguous(),
    }

    out_dir = Path("weights/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.safetensors"
    safetensors.torch.save_file(checkpoint_data, out_path)
    print("Saved Topic 13 reference checkpoint to", out_path)


if __name__ == "__main__":
    generate_checkpoint()
