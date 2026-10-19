"""PyTorch 2.x implementation for Category 04 Topic 15:
Grokking, Phase Transitions, Edge of Stability, and Implicit Bias.

Provides production-grade implementations of:
1. Grokfast EMA gradient filtering for accelerating phase transition / delayed generalization.
2. Edge of Stability (EOS) cubic oscillator and self-stabilizing dynamics.
3. Implicit Bias logistic margin step on linearly separable data.
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


def step_grokfast(
    g_raw: torch.Tensor,
    mu_prev: torch.Tensor,
    alpha: float = 0.8,
    lambda_gf: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grokfast EMA gradient filtering step (Lee et al., 2024).

    Amplifies slow-moving, generalizing Fourier direction in modular arithmetic.
    """
    mu_next = alpha * mu_prev + (1.0 - alpha) * g_raw
    g_filtered = g_raw + lambda_gf * mu_next
    return g_filtered, mu_next


def step_edge_of_stability(
    x: torch.Tensor,
    H: torch.Tensor,
    gamma: torch.Tensor,
    eta: float = 0.1,
) -> torch.Tensor:
    """Edge of Stability (EOS) non-linear cubic oscillator step (Cohen et al., 2021).

    Simulates the self-stabilizing oscillation when lambda_max(H) > 2/eta.
    """
    grad = H * x + gamma * (x ** 3)
    x_next = x - eta * grad
    return x_next


def step_implicit_bias(
    w: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    eta: float = 0.2,
) -> torch.Tensor:
    """Implicit bias gradient step on separable linear classification (Soudry et al., 2018).

    Provably converges in direction to the L2 max-margin SVM separator.
    """
    margins = y * (X @ w)
    sigmoids = 1.0 / (1.0 + torch.exp(margins))
    grad = -torch.mean((y.unsqueeze(1) * X) * sigmoids.unsqueeze(1), dim=0)
    w_next = w - eta * grad
    return w_next


class LinearAnchor(nn.Module):
    """Linear anchor module for cross-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TrainingDynamicsUnifiedModule(nn.Module):
    """Unified module verifying training dynamics invariants against safetensors golden references."""

    def __init__(self, safetensors_path: str | Path) -> None:
        super().__init__()
        weights = load_file(str(safetensors_path))

        # Grokfast
        self.g_raw = weights["g_raw"]
        self.mu_prev = weights["mu_prev"]
        self.golden_mu = weights["golden_mu"]
        self.golden_g_grokfast = weights["golden_g_grokfast"]

        # EOS
        self.x_eos = weights["x_eos"]
        self.h_eos = weights["h_eos"]
        self.gamma_eos = weights["gamma_eos"]
        self.golden_x_eos = weights["golden_x_eos"]

        # Implicit Bias
        self.w_sep = weights["w_sep"]
        self.x_sep = weights["x_sep"]
        self.y_sep = weights["y_sep"]
        self.golden_w_sep = weights["golden_w_sep"]

        # Linear Anchor
        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. Grokfast invariant
        g_gf, mu_next = step_grokfast(self.g_raw, self.mu_prev, alpha=0.8, lambda_gf=2.0)
        err_mu = float(torch.max(torch.abs(mu_next - self.golden_mu)).item())
        err_gf = float(torch.max(torch.abs(g_gf - self.golden_g_grokfast)).item())
        err_grok = max(err_mu, err_gf)
        assert err_grok < atol, f"Grokfast invariant failed: err={err_grok:.2e} >= {atol:.2e}"
        errors["grokfast_err"] = err_grok

        # 2. EOS invariant
        x_next = step_edge_of_stability(self.x_eos, self.h_eos, self.gamma_eos, eta=0.1)
        err_eos = float(torch.max(torch.abs(x_next - self.golden_x_eos)).item())
        assert err_eos < atol, f"EOS invariant failed: err={err_eos:.2e} >= {atol:.2e}"
        errors["eos_err"] = err_eos

        # 3. Implicit bias invariant
        w_next = step_implicit_bias(self.w_sep, self.x_sep, self.y_sep, eta=0.2)
        err_bias = float(torch.max(torch.abs(w_next - self.golden_w_sep)).item())
        assert err_bias < atol, f"Implicit bias invariant failed: err={err_bias:.2e} >= {atol:.2e}"
        errors["implicit_bias_err"] = err_bias

        # 4. Linear anchor invariant
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.safetensors")
    module = TrainingDynamicsUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
