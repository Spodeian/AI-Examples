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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
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


    generate_checkpoint()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_topic13_pytorch()
