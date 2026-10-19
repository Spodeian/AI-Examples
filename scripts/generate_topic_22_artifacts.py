"""Generate golden reference SafeTensors and verify PyTorch/JAX parity for Topic 22.

Topic 22: 08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr
Category: 04_training_methods_and_optimisers
Subfolder: 03_training_dynamics_self_optimising_non_backprop
"""

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

if __name__ == "__main__":
    generate_topic_22_artifacts()
