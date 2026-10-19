"""PyTorch 2.x implementation for Category 04 Topic 22:
Meta-Parameter Sensitivity, fANOVA, Sobol' Indices, and Effective Learning Rate.

Provides production-grade implementations of:
1. Functional ANOVA (fANOVA) variance decomposition and Sobol' first-order / total-effect indices (Hutter et al., 2014; Sobol', 1993).
2. Scale-invariant Effective Learning Rate and equilibrium norm dynamics (Arora et al., 2019; Wan et al., 2021).
3. Chaotic unrolled meta-gradient explosion at the Edge of Stability (Wu et al., 2018; Metz et al., 2019).
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


def compute_fanova_sobol_indices(
    y_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes fANOVA variance decomposition and Sobol' indices across 2D grid (Sobol', 1993; Hutter et al., 2014).

    y_grid: [M, N] evaluation grid over 2 meta-parameters.
    Returns (total_var, v1, v2, s1, s2, st1, st2).
    """
    total_var = torch.var(y_grid, unbiased=False)

    e_y_given_x1 = torch.mean(y_grid, dim=1)
    e_y_given_x2 = torch.mean(y_grid, dim=0)

    v1 = torch.var(e_y_given_x1, unbiased=False)
    v2 = torch.var(e_y_given_x2, unbiased=False)
    v12 = total_var - (v1 + v2)

    s1 = v1 / total_var
    s2 = v2 / total_var
    st1 = (v1 + v12) / total_var
    st2 = (v2 + v12) / total_var

    return total_var, v1, v2, s1, s2, st1, st2


def compute_effective_lr_and_eq_norm(
    w: torch.Tensor,
    eta_nominal: torch.Tensor,
    weight_decay: torch.Tensor,
    grad_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale-invariant effective learning rate and equilibrium norm (Arora et al., 2019; Wan et al., 2021).

    Returns (effective_lr, eq_norm).
    """
    w_norm_sq = torch.sum(w ** 2)
    effective_lr = eta_nominal / w_norm_sq
    eq_norm = ((eta_nominal * (grad_norm ** 2)) / (2.0 * weight_decay)) ** 0.25
    return effective_lr, eq_norm


def compute_chaotic_meta_gradient_explosion(
    eig_max: torch.Tensor,
    eta_meta: torch.Tensor,
    steps_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unrolled meta-gradient spectral multiplier and exponential explosion (Metz et al., 2019).

    Returns (multiplier, accumulated).
    """
    multiplier = torch.abs(1.0 - eta_meta * eig_max)
    accumulated = multiplier ** steps_t
    return multiplier, accumulated


class LinearAnchor(nn.Module):
    """Anchor module for multi-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MetaSensitivityUnifiedModule:
    """Unified container for Category 04 Topic 22 verification."""

    def __init__(self, weights_path: Path):
        weights = load_file(str(weights_path))

        # Sobol
        self.y_grid = weights["y_grid"]
        self.golden_total_var = weights["golden_total_var"]
        self.golden_v1 = weights["golden_v1"]
        self.golden_v2 = weights["golden_v2"]
        self.golden_s1 = weights["golden_s1"]
        self.golden_s2 = weights["golden_s2"]
        self.golden_st1 = weights["golden_st1"]
        self.golden_st2 = weights["golden_st2"]

        # Effective LR
        self.w = weights["w"]
        self.eta_nominal = weights["eta_nominal"]
        self.weight_decay = weights["weight_decay"]
        self.grad_norm = weights["grad_norm"]
        self.golden_effective_lr = weights["golden_effective_lr"]
        self.golden_eq_norm = weights["golden_eq_norm"]

        # Chaotic Meta-Gradient
        self.eig_max = weights["eig_max"]
        self.eta_meta = weights["eta_meta"]
        self.steps_t = weights["steps_t"]
        self.golden_spectral_multiplier = weights["golden_spectral_multiplier"]
        self.golden_accumulated_explosion = weights["golden_accumulated_explosion"]

        # Linear Anchor
        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. Sobol invariant
        tot_var, v1, v2, s1, s2, st1, st2 = compute_fanova_sobol_indices(self.y_grid)
        err_tot = float(torch.abs(tot_var - self.golden_total_var[0]).item())
        err_v1 = float(torch.abs(v1 - self.golden_v1[0]).item())
        err_v2 = float(torch.abs(v2 - self.golden_v2[0]).item())
        err_s1 = float(torch.abs(s1 - self.golden_s1[0]).item())
        err_s2 = float(torch.abs(s2 - self.golden_s2[0]).item())
        err_st1 = float(torch.abs(st1 - self.golden_st1[0]).item())
        err_st2 = float(torch.abs(st2 - self.golden_st2[0]).item())
        err_sobol = max(err_tot, err_v1, err_v2, err_s1, err_s2, err_st1, err_st2)
        assert err_sobol < atol, f"Sobol invariant failed: {err_sobol:.2e}"
        errors["sobol_err"] = err_sobol

        # 2. Effective LR invariant
        eff_lr, eq_norm = compute_effective_lr_and_eq_norm(self.w, self.eta_nominal, self.weight_decay, self.grad_norm)
        err_elr = float(torch.abs(eff_lr - self.golden_effective_lr[0]).item())
        err_eqn = float(torch.abs(eq_norm - self.golden_eq_norm[0]).item())
        err_eff = max(err_elr, err_eqn)
        assert err_eff < atol, f"Effective LR invariant failed: {err_eff:.2e}"
        errors["eff_lr_err"] = err_eff

        # 3. Chaotic Meta-Gradient invariant
        mult, accum = compute_chaotic_meta_gradient_explosion(self.eig_max, self.eta_meta, self.steps_t)
        err_mult = float(torch.abs(mult - self.golden_spectral_multiplier[0]).item())
        err_accum = float(torch.abs(accum - self.golden_accumulated_explosion[0]).item())
        err_chaos = max(err_mult, err_accum)
        assert err_chaos < atol, f"Chaotic Meta-Gradient invariant failed: {err_chaos:.2e}"
        errors["chaos_err"] = err_chaos

        # 4. Linear Anchor invariant
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr.safetensors")
    module = MetaSensitivityUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Topic 22 Invariant Parity Checks Passed: {errs}")
