"""Generate deterministic reference safetensors checkpoint for Category 03 Topic 10:
10_losses_optimal_transport_wasserstein_sinkhorn_and_mmd
"""

from __future__ import annotations

from pathlib import Path
import torch
from safetensors.torch import save_file

import core


def sinkhorn_cost(X: torch.Tensor, Y: torch.Tensor, eps: float = 0.5, num_iters: int = 20) -> torch.Tensor:
    N, _ = X.shape
    M, _ = Y.shape
    C = torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1)
    K = torch.exp(-C / eps)
    a = torch.ones(N, dtype=torch.float32) / N
    b = torch.ones(M, dtype=torch.float32) / M
    v = torch.ones(M, dtype=torch.float32) / M
    for _ in range(num_iters):
        u = a / (torch.matmul(K, v) + 1e-12)
        v = b / (torch.matmul(K.T, u) + 1e-12)
    coupling = u.unsqueeze(1) * K * v.unsqueeze(0)
    return torch.sum(coupling * C)


def compute_mmd(X: torch.Tensor, Y: torch.Tensor, sigmas: list[float] = [0.5, 1.0, 2.0]) -> torch.Tensor:
    def rbf_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)) ** 2, dim=-1)
        k = torch.zeros_like(dist_sq)
        for s in sigmas:
            gamma = 1.0 / (2.0 * s * s)
            k = k + torch.exp(-gamma * dist_sq)
        return k

    K_xx = rbf_kernel(X, X)
    K_yy = rbf_kernel(Y, Y)
    K_xy = rbf_kernel(X, Y)
    return K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()


def compute_energy(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    dist_xy = torch.norm(X.unsqueeze(1) - Y.unsqueeze(0), dim=-1)
    dist_xx = torch.norm(X.unsqueeze(1) - X.unsqueeze(0), dim=-1)
    dist_yy = torch.norm(Y.unsqueeze(1) - Y.unsqueeze(0), dim=-1)
    return 2.0 * dist_xy.mean() - dist_xx.mean() - dist_yy.mean()


def generate_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    # 1. Linear Anchor parameters
    in_dim = 4
    out_dim = 4
    batch_size = 8
    weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    test_input = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
    test_output = test_input @ weight + bias

    # 2. 1D Wasserstein
    num_samples = 16
    u_1d = torch.randn(batch_size, num_samples, generator=gen, dtype=torch.float32)
    v_1d = torch.randn(batch_size, num_samples, generator=gen, dtype=torch.float32)
    u_sorted, _ = torch.sort(u_1d, dim=-1)
    v_sorted, _ = torch.sort(v_1d, dim=-1)
    expected_w1 = torch.mean(torch.abs(u_sorted - v_sorted))

    # 3. Sliced Wasserstein
    num_slices = 16
    swd_x = torch.randn(batch_size, num_samples, in_dim, generator=gen, dtype=torch.float32)
    swd_y = torch.randn(batch_size, num_samples, in_dim, generator=gen, dtype=torch.float32)
    raw_proj = torch.randn(num_slices, in_dim, generator=gen, dtype=torch.float32)
    projections = raw_proj / torch.norm(raw_proj, dim=-1, keepdim=True)
    x_proj = torch.matmul(swd_x, projections.T).permute(0, 2, 1)
    y_proj = torch.matmul(swd_y, projections.T).permute(0, 2, 1)
    x_p_sorted, _ = torch.sort(x_proj, dim=-1)
    y_p_sorted, _ = torch.sort(y_proj, dim=-1)
    expected_swd = torch.mean(torch.abs(x_p_sorted - y_p_sorted))

    # 4. Point clouds for Sinkhorn, MMD, Energy
    cloud_p = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
    cloud_q = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)

    # Sinkhorn
    expected_sinkhorn_raw = sinkhorn_cost(cloud_p, cloud_q, eps=0.5, num_iters=20)
    s_pp = sinkhorn_cost(cloud_p, cloud_p, eps=0.5, num_iters=20)
    s_qq = sinkhorn_cost(cloud_q, cloud_q, eps=0.5, num_iters=20)
    expected_sinkhorn_debiased = expected_sinkhorn_raw - 0.5 * s_pp - 0.5 * s_qq

    # MMD
    expected_mmd = compute_mmd(cloud_p, cloud_q, sigmas=[0.5, 1.0, 2.0])

    # Energy distance
    expected_energy = compute_energy(cloud_p, cloud_q)

    tensors = {
        "weight": weight,
        "bias": bias,
        "test_input": test_input,
        "test_output": test_output,
        "u_1d": u_1d,
        "v_1d": v_1d,
        "expected_w1": expected_w1.reshape(1),
        "swd_x": swd_x,
        "swd_y": swd_y,
        "projections": projections,
        "expected_swd": expected_swd.reshape(1),
        "cloud_p": cloud_p,
        "cloud_q": cloud_q,
        "expected_sinkhorn_raw": expected_sinkhorn_raw.reshape(1),
        "expected_sinkhorn_debiased": expected_sinkhorn_debiased.reshape(1),
        "expected_mmd": expected_mmd.reshape(1),
        "expected_energy": expected_energy.reshape(1),
    }

    out_dir = Path("weights/03_building_blocks_and_losses")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "10_losses_optimal_transport_wasserstein_sinkhorn_and_mmd.safetensors"
    save_file(tensors, out_path)
    print(f"Saved {len(tensors)} tensors to {out_path}")


if __name__ == "__main__":
    generate_weights()
