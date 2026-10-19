"""Optimal Transport, Wasserstein, Sinkhorn, and MMD Losses (C03 Topic 10).

Implements geometry-aware, distribution-matching, and kernel distance loss formulations:
1. Exact 1D Wasserstein Distance (Kantorovich 1942).
2. Sliced Wasserstein Distance (SWD) via hyperspherical random projections (Bonneel et al. 2015).
3. Entropic Regularized Optimal Transport / Sinkhorn Distance (Cuturi 2013).
4. Debiased Sinkhorn Divergence with self-transport correction (Feydy et al. 2019).
5. Maximum Mean Discrepancy (MMD) with multi-scale Gaussian RBF kernels (Gretton et al. 2012).
6. Energy / Cramer Distance (Szekely & Rizzo 2013).
7. Linear Anchor: Baseline affine transform.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file

try:
    from jaxtyping import Float
except ImportError:
    class _Float:
        def __getitem__(self, item: Any) -> Any:
            return Any
    Float = _Float()


def compute_wasserstein_1d(
    u: Float[torch.Tensor, "batch num_samples"],
    v: Float[torch.Tensor, "batch num_samples"],
) -> torch.Tensor:
    """Exact 1D Wasserstein Distance via sorted order statistics: W_1(u, v) = mean |u_(i) - v_(i)|."""
    u_sorted, _ = torch.sort(u, dim=-1)
    v_sorted, _ = torch.sort(v, dim=-1)
    return torch.mean(torch.abs(u_sorted - v_sorted))


def compute_sliced_wasserstein(
    x: Float[torch.Tensor, "batch num_samples dim"],
    y: Float[torch.Tensor, "batch num_samples dim"],
    projections: Float[torch.Tensor, "num_slices dim"],
) -> torch.Tensor:
    """Sliced Wasserstein Distance: average 1D Wasserstein distance across random 1D projections."""
    # x: [B, N, D], projections: [L, D] -> proj_x: [B, N, L] -> permute to [B, L, N]
    proj_x = torch.matmul(x, projections.T).permute(0, 2, 1)
    proj_y = torch.matmul(y, projections.T).permute(0, 2, 1)
    sorted_x, _ = torch.sort(proj_x, dim=-1)
    sorted_y, _ = torch.sort(proj_y, dim=-1)
    return torch.mean(torch.abs(sorted_x - sorted_y))


def compute_sinkhorn_cost(
    x: Float[torch.Tensor, "num_pts dim"],
    y: Float[torch.Tensor, "num_pts dim"],
    eps: float = 0.5,
    num_iters: int = 20,
) -> torch.Tensor:
    """Entropic Regularized Optimal Transport (Sinkhorn algorithm) under squared Euclidean cost."""
    n = x.shape[0]
    m = y.shape[0]
    # Cost matrix C_ij = ||x_i - y_j||^2
    c = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)
    k = torch.exp(-c / eps)
    a = torch.ones(n, dtype=x.dtype, device=x.device) / float(n)
    b = torch.ones(m, dtype=y.dtype, device=y.device) / float(m)
    v = torch.ones(m, dtype=y.dtype, device=y.device) / float(m)

    for _ in range(num_iters):
        u = a / (torch.matmul(k, v) + 1e-12)
        v = b / (torch.matmul(k.T, u) + 1e-12)

    coupling = u.unsqueeze(1) * k * v.unsqueeze(0)
    return torch.sum(coupling * c)


def compute_debiased_sinkhorn(
    x: Float[torch.Tensor, "num_pts dim"],
    y: Float[torch.Tensor, "num_pts dim"],
    eps: float = 0.5,
    num_iters: int = 20,
) -> torch.Tensor:
    """Debiased Sinkhorn Divergence: S_eps(X, Y) - 0.5 * S_eps(X, X) - 0.5 * S_eps(Y, Y)."""
    s_xy = compute_sinkhorn_cost(x, y, eps=eps, num_iters=num_iters)
    s_xx = compute_sinkhorn_cost(x, x, eps=eps, num_iters=num_iters)
    s_yy = compute_sinkhorn_cost(y, y, eps=eps, num_iters=num_iters)
    return s_xy - 0.5 * s_xx - 0.5 * s_yy


def compute_mmd(
    x: Float[torch.Tensor, "num_pts dim"],
    y: Float[torch.Tensor, "num_pts dim"],
    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Maximum Mean Discrepancy (MMD^2) with multi-scale Gaussian RBF kernels."""
    def rbf_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.sum((a.unsqueeze(1) - b.unsqueeze(0)) ** 2, dim=-1)
        k = torch.zeros_like(dist_sq)
        for s in sigmas:
            gamma = 1.0 / (2.0 * s * s)
            k = k + torch.exp(-gamma * dist_sq)
        return k

    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def compute_energy_distance(
    x: Float[torch.Tensor, "num_pts dim"],
    y: Float[torch.Tensor, "num_pts dim"],
) -> torch.Tensor:
    """Energy Distance / Cramer Distance: 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||."""
    dist_xy = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1)
    dist_xx = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=-1)
    dist_yy = torch.norm(y.unsqueeze(1) - y.unsqueeze(0), dim=-1)
    return 2.0 * dist_xy.mean() - dist_xx.mean() - dist_yy.mean()


class LinearAnchor(nn.Module):
    """Linear projection layer serving as parity anchor."""

    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class OptimalTransportUnifiedModule(nn.Module):
    """Unified module combining optimal transport losses with linear anchor."""

    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.anchor = LinearAnchor(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.anchor(x)


def load_checkpoint_tensors(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    return load_file(str(checkpoint_path))


if __name__ == "__main__":
    ckpt = Path("weights/03_building_blocks_and_losses/10_losses_optimal_transport_wasserstein_sinkhorn_and_mmd.safetensors")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt}")

    data = load_checkpoint_tensors(ckpt)
    print(f"Loaded {len(data)} tensors from {ckpt.name}")

    # 1. Linear Anchor parity
    mod = OptimalTransportUnifiedModule()
    mod.anchor.weight.data.copy_(data["weight"])
    mod.anchor.bias.data.copy_(data["bias"])
    pred = mod(data["test_input"])
    err_anchor = float(torch.max(torch.abs(pred - data["test_output"])).item())
    assert err_anchor < 1e-5, f"Anchor parity error {err_anchor:.2e} >= 1e-5"

    # 2. 1D Wasserstein parity
    w1_val = compute_wasserstein_1d(data["u_1d"], data["v_1d"])
    err_w1 = float(torch.abs(w1_val - data["expected_w1"]).item())
    assert err_w1 < 1e-5, f"1D Wasserstein parity error {err_w1:.2e} >= 1e-5"

    # 3. Sliced Wasserstein parity
    swd_val = compute_sliced_wasserstein(data["swd_x"], data["swd_y"], data["projections"])
    err_swd = float(torch.abs(swd_val - data["expected_swd"]).item())
    assert err_swd < 1e-5, f"SWD parity error {err_swd:.2e} >= 1e-5"

    # 4. Sinkhorn raw parity
    sink_raw = compute_sinkhorn_cost(data["cloud_p"], data["cloud_q"], eps=0.5, num_iters=20)
    err_sink_raw = float(torch.abs(sink_raw - data["expected_sinkhorn_raw"]).item())
    assert err_sink_raw < 1e-5, f"Sinkhorn raw parity error {err_sink_raw:.2e} >= 1e-5"

    # 5. Debiased Sinkhorn parity
    sink_deb = compute_debiased_sinkhorn(data["cloud_p"], data["cloud_q"], eps=0.5, num_iters=20)
    err_sink_deb = float(torch.abs(sink_deb - data["expected_sinkhorn_debiased"]).item())
    assert err_sink_deb < 1e-5, f"Debiased Sinkhorn parity error {err_sink_deb:.2e} >= 1e-5"

    # 6. MMD parity
    mmd_val = compute_mmd(data["cloud_p"], data["cloud_q"], sigmas=(0.5, 1.0, 2.0))
    err_mmd = float(torch.abs(mmd_val - data["expected_mmd"]).item())
    assert err_mmd < 1e-5, f"MMD parity error {err_mmd:.2e} >= 1e-5"

    # 7. Energy distance parity
    energy_val = compute_energy_distance(data["cloud_p"], data["cloud_q"])
    err_energy = float(torch.abs(energy_val - data["expected_energy"]).item())
    assert err_energy < 1e-5, f"Energy distance parity error {err_energy:.2e} >= 1e-5"

    print(
        f"All 7 invariants verified! Anchor: {err_anchor:.2e}, W1: {err_w1:.2e}, SWD: {err_swd:.2e}, "
        f"SinkRaw: {err_sink_raw:.2e}, SinkDeb: {err_sink_deb:.2e}, MMD: {err_mmd:.2e}, Energy: {err_energy:.2e}"
    )
