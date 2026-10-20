"""PyTorch 2.x implementation for Category 05 Topic 05:
Kolmogorov-Arnold Networks (KAN): B-Spline KAN, FastKAN (RBF), and ChebyKAN
in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class ChebyKANLayer(nn.Module):
    """Chebyshev Kolmogorov-Arnold Network Layer (ChebyKAN).
    
    Replaces univariate spline evaluations with orthogonal Chebyshev polynomials:
    phi_ij(x) = sum_{n=0}^N c_{i, j, n} T_n(tanh(x))
    where T_0(u)=1, T_1(u)=u, T_{n+1}(u) = 2u T_n(u) - T_{n-1}(u).
    
    References:
        - Kolmogorov (1957); Arnold (1957)
        - Liu et al. (2024): KAN: Kolmogorov-Arnold Networks
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 4) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        # Coefficients c: (in_features, out_features, degree + 1)
        self.c = nn.Parameter(torch.empty(in_features, out_features, degree + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Squash inputs to [-1, 1] domain to maintain polynomial stability
        u = torch.tanh(x)
        
        # Compute Chebyshev polynomials T_0 .. T_degree
        T = [torch.ones_like(u), u]
        for _ in range(2, self.degree + 1):
            T.append(2.0 * u * T[-1] - T[-2])
        T_x = torch.stack(T, dim=-1)  # (..., in_features, degree + 1)

        # Contract across in_features and degree dimensions
        return torch.einsum("...id,ijd->...j", T_x, self.c)


class FastKANLayer(nn.Module):
    """Fast Kolmogorov-Arnold Network Layer with Radial Basis Functions (FastKAN).
    
    Replaces recursive Cox-de Boor B-splines with Gaussian RBF kernels and a SiLU base:
    phi_ij(x) = w_b * silu(x) + sum_k w_k * exp(-((x - mu_k)/h)^2)
    
    References:
        - Li (2024): FastKAN: Efficient Kolmogorov-Arnold Networks using Radial Basis Functions
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int = 8,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids

        # Grid centers
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)
        h = (grid_max - grid_min) / (num_grids - 1)
        self.denominator = float(h)

        # Base linear branch
        self.base_w = nn.Parameter(torch.empty(in_features, out_features))
        # RBF branch weights: (in_features, num_grids, out_features)
        self.rbf_w = nn.Parameter(torch.empty(in_features, num_grids, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base linear branch: silu(x) @ base_w
        base_out = F.silu(x) @ self.base_w

        # RBF branch: exp(-((x - grid) / h)^2)
        diff = x.unsqueeze(-1) - self.grid  # (..., in_features, num_grids)
        phi = torch.exp(-torch.pow(diff / self.denominator, 2.0))

        # Contract across in_features and num_grids
        rbf_out = torch.einsum("...ig,igj->...j", phi, self.rbf_w)

        return base_out + rbf_out


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def verify_against_golden() -> None:
    """Verifies PyTorch implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = torch.max(torch.abs(test_out - weights["test_output"])).item()
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]
    in_dim, out_dim = 4, 4

    # 2. ChebyKAN verification
    cheby = ChebyKANLayer(in_features=in_dim, out_features=out_dim, degree=4)
    with torch.no_grad():
        cheby.c.copy_(weights["cheby_c"])
    out_cheby = cheby(x)
    err_cheby = torch.max(torch.abs(out_cheby - weights["chebykan_output"])).item()
    assert err_cheby < 1e-6, f"ChebyKAN parity error {err_cheby:.2e} exceeds 1e-6"

    # 3. FastKAN verification
    fastkan = FastKANLayer(in_features=in_dim, out_features=out_dim, num_grids=8)
    with torch.no_grad():
        fastkan.base_w.copy_(weights["fastkan_base_w"])
        fastkan.rbf_w.copy_(weights["fastkan_rbf_w"])
        fastkan.grid.copy_(weights["fastkan_grid"])
        fastkan.denominator = float(weights["fastkan_denominator"])
    out_fastkan = fastkan(x)
    err_fastkan = torch.max(torch.abs(out_fastkan - weights["fastkan_output"])).item()
    assert err_fastkan < 1e-6, f"FastKAN parity error {err_fastkan:.2e} exceeds 1e-6"

    print("PyTorch 2.x KAN parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
