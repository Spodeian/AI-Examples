#!/usr/bin/env python3
"""Generate golden weights and inputs/outputs for Category 05 Topic 05:
Kolmogorov-Arnold Networks (KAN): B-Spline KAN, FastKAN (RBF), and ChebyKAN.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


def chebyshev_polynomials(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate Chebyshev polynomials T_0(x) to T_degree(x) for x in [-1, 1].
    Output shape: (..., degree + 1)
    """
    # Clamping or tanh-squashing to [-1, 1] domain
    u = torch.tanh(x)
    T = [torch.ones_like(u), u]
    for n in range(2, degree + 1):
        T.append(2.0 * u * T[-1] - T[-2])
    return torch.stack(T, dim=-1)


def rbf_basis(x: torch.Tensor, grid: torch.Tensor, denominator: float) -> torch.Tensor:
    """Evaluate Gaussian Radial Basis Functions for FastKAN.
    x: (..., in_dim)
    grid: (num_grids,)
    Output: (..., in_dim, num_grids)
    """
    # (..., in_dim, 1) - (num_grids,)
    diff = x.unsqueeze(-1) - grid
    return torch.exp(-torch.pow(diff / denominator, 2.0))


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    in_dim = 4
    out_dim = 4
    degree = 4        # Chebyshev polynomial degree (0..4, so 5 basis terms)
    num_grids = 8     # FastKAN RBF centers

    # Input tensor: (B, S, in_dim)
    x = torch.randn(batch_size, seq_len, in_dim, dtype=torch.float32)

    # 1. ChebyKAN Parameters
    # Weight tensor: c[in_dim, out_dim, degree + 1]
    cheby_c = torch.randn(in_dim, out_dim, degree + 1, dtype=torch.float32) * 0.2

    # ChebyKAN forward:
    # u = tanh(x): (B, S, in_dim)
    # T(u): (B, S, in_dim, degree + 1)
    # y[b, s, j] = sum_i sum_d cheby_c[i, j, d] * T(u)[b, s, i, d]
    T_x = chebyshev_polynomials(x, degree)  # (B, S, in_dim, degree + 1)
    # Einsum: b s i d, i j d -> b s j
    chebykan_output = torch.einsum("bsid,ijd->bsj", T_x, cheby_c)

    # 2. FastKAN (RBF) Parameters
    # Grid centers uniformly spaced in [-1, 1]
    grid_min, grid_max = -1.0, 1.0
    fastkan_grid = torch.linspace(grid_min, grid_max, num_grids, dtype=torch.float32)
    h = (grid_max - grid_min) / (num_grids - 1)
    fastkan_denominator = float(h)

    # Base linear weight: W_b (in_dim, out_dim)
    fastkan_base_w = torch.randn(in_dim, out_dim, dtype=torch.float32) * 0.2
    # RBF linear weight: W_rbf (in_dim, num_grids, out_dim)
    fastkan_rbf_w = torch.randn(in_dim, num_grids, out_dim, dtype=torch.float32) * 0.2

    # FastKAN forward:
    # base branch: silu(x) @ W_b
    base_out = F.silu(x) @ fastkan_base_w  # (B, S, out_dim)

    # RBF branch:
    phi = rbf_basis(x, fastkan_grid, fastkan_denominator)  # (B, S, in_dim, num_grids)
    # Einsum: b s i g, i g j -> b s j
    rbf_out = torch.einsum("bsig,igj->bsj", phi, fastkan_rbf_w)
    fastkan_output = base_out + rbf_out

    # 3. Standard linear anchor for cross-track parity verification
    D_core = 4
    test_input = torch.randn(8, D_core)
    weight = torch.randn(D_core, D_core)
    bias = torch.randn(D_core)
    test_output = test_input @ weight + bias

    # Output directory
    out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "05_kolmogorov_arnold_networks_kan.safetensors"

    tensors = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "input": x,
        "cheby_c": cheby_c,
        "chebykan_output": chebykan_output,
        "fastkan_grid": fastkan_grid,
        "fastkan_base_w": fastkan_base_w,
        "fastkan_rbf_w": fastkan_rbf_w,
        "fastkan_denominator": torch.tensor(fastkan_denominator, dtype=torch.float32),
        "fastkan_output": fastkan_output,
    }

    save_file(tensors, str(out_file))
    print(f"Generated {out_file} successfully.")
    print(f"chebykan_output shape: {chebykan_output.shape}, norm: {chebykan_output.norm():.6f}")
    print(f"fastkan_output shape: {fastkan_output.shape}, norm: {fastkan_output.norm():.6f}")


if __name__ == "__main__":
    main()
