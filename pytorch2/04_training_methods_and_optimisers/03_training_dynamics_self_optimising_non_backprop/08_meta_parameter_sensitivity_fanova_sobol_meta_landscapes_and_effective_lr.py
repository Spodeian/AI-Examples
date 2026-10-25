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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file

    def generate_topic_22_artifacts():
        torch.manual_seed(42)
        np.random.seed(42)

        # 1. Sobol' Sensitivity & fANOVA
        # Evaluation grid over 2 meta-parameters (x1, x2) in [0, 1]
        # Grid of 4 x 4 = 16 points
        x1_coords = torch.linspace(0.1, 0.9, 4, dtype=torch.float32)
        x2_coords = torch.linspace(0.1, 0.9, 4, dtype=torch.float32)
        grid_x1, grid_x2 = torch.meshgrid(x1_coords, x2_coords, indexing="ij") # [4, 4]

        # Model: Y = 2.0 * x1 + 3.0 * x2^2 + 1.5 * x1 * x2
        y_grid = 2.0 * grid_x1 + 3.0 * (grid_x2 ** 2) + 1.5 * grid_x1 * grid_x2 # [4, 4]

        total_var = torch.var(y_grid, unbiased=False) # Total variance V

        # Marginal expectations E[Y | x1] (mean over dim 1) and E[Y | x2] (mean over dim 0)
        e_y_given_x1 = torch.mean(y_grid, dim=1) # [4]
        e_y_given_x2 = torch.mean(y_grid, dim=0) # [4]

        v1 = torch.var(e_y_given_x1, unbiased=False)
        v2 = torch.var(e_y_given_x2, unbiased=False)
        v12 = total_var - (v1 + v2)

        s1 = v1 / total_var # First-order index S1
        s2 = v2 / total_var # First-order index S2
        st1 = (v1 + v12) / total_var # Total-effect index ST1
        st2 = (v2 + v12) / total_var # Total-effect index ST2

        # 2. Scale-Invariant Effective Learning Rate & Equilibrium Norm
        # w: [8]
        w = torch.tensor([0.5, -0.8, 1.2, -0.4, 0.9, -0.3, 0.7, -0.6], dtype=torch.float32)
        eta_nominal = torch.tensor([0.02], dtype=torch.float32)
        weight_decay = torch.tensor([0.001], dtype=torch.float32)
        grad_norm = torch.tensor([0.4], dtype=torch.float32)

        w_norm_sq = torch.sum(w ** 2)
        effective_lr = eta_nominal / w_norm_sq # eta_eff = eta / ||w||^2

        # Equilibrium norm: ||w*|| = (eta * G^2 / (2 * lambda))^(1/4)
        eq_norm = ((eta_nominal * (grad_norm ** 2)) / (2.0 * weight_decay)) ** 0.25

        # 3. Chaotic Meta-Gradient Explosion
        # Operator: (I - eta * H), eigenvalues: [1 - eta * eig1, 1 - eta * eig2]
        # For eig = 2.5, eta = 1.0 => multiplier = -1.5 (unstable edge of stability!)
        eig_max = torch.tensor([2.5], dtype=torch.float32)
        eta_meta = torch.tensor([1.0], dtype=torch.float32)
        steps_t = torch.tensor([6.0], dtype=torch.float32)

        spectral_multiplier = torch.abs(1.0 - eta_meta * eig_max) # 1.5
        accumulated_explosion = spectral_multiplier ** steps_t # 1.5^6 = 11.390625

        # 4. Linear Anchor
        anchor_x = torch.randn(2, 8, dtype=torch.float32)
        anchor_w = torch.randn(8, 4, dtype=torch.float32) / math.sqrt(8)
        anchor_b = torch.randn(4, dtype=torch.float32) * 0.1
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        # Construct tensors dictionary
        tensors = {
            # Sobol & fANOVA
            "y_grid": y_grid.clone().contiguous(),
            "golden_total_var": total_var.unsqueeze(0).clone().contiguous(),
            "golden_v1": v1.unsqueeze(0).clone().contiguous(),
            "golden_v2": v2.unsqueeze(0).clone().contiguous(),
            "golden_s1": s1.unsqueeze(0).clone().contiguous(),
            "golden_s2": s2.unsqueeze(0).clone().contiguous(),
            "golden_st1": st1.unsqueeze(0).clone().contiguous(),
            "golden_st2": st2.unsqueeze(0).clone().contiguous(),

            # Effective LR
            "w": w.clone().contiguous(),
            "eta_nominal": eta_nominal.clone().contiguous(),
            "weight_decay": weight_decay.clone().contiguous(),
            "grad_norm": grad_norm.clone().contiguous(),
            "golden_effective_lr": effective_lr.clone().contiguous(),
            "golden_eq_norm": eq_norm.clone().contiguous(),

            # Chaotic Meta-Gradient
            "eig_max": eig_max.clone().contiguous(),
            "eta_meta": eta_meta.clone().contiguous(),
            "steps_t": steps_t.clone().contiguous(),
            "golden_spectral_multiplier": spectral_multiplier.clone().contiguous(),
            "golden_accumulated_explosion": accumulated_explosion.clone().contiguous(),

            # Linear Anchor
            "anchor_x": anchor_x.clone().contiguous(),
            "anchor_w": anchor_w.clone().contiguous(),
            "anchor_b": anchor_b.clone().contiguous(),
            "golden_anchor_out": golden_anchor_out.clone().contiguous(),

            # Standard test inputs/outputs for parity test runners
            "test_input": anchor_x.clone().contiguous(),
            "test_output": golden_anchor_out.clone().contiguous(),
            "weight": anchor_w.clone().contiguous(),
            "bias": anchor_b.clone().contiguous(),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr.safetensors"

        save_file(tensors, out_path)
        print(f"Saved {len(tensors)} tensors to {out_path}")

        # Verify values
        print(f"Sobol Total Var: {total_var.item():.4f}, S1: {s1.item():.4f}, S2: {s2.item():.4f}, ST1: {st1.item():.4f}, ST2: {st2.item():.4f}")
        print(f"Effective LR: {effective_lr.item():.6f}, Eq Norm: {eq_norm.item():.4f}")
        print(f"Chaotic Explosion (T=6): {accumulated_explosion.item():.4f}")

    generate_topic_22_artifacts()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr.safetensors")
    module = MetaSensitivityUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Topic 22 Invariant Parity Checks Passed: {errs}")
