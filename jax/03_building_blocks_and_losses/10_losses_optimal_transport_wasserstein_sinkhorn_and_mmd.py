"""JAX / Functional NumPy: Optimal Transport, Wasserstein, Sinkhorn, and MMD Losses (C03 Topic 10).

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
from typing import Dict, Sequence, Tuple

import numpy as np
from safetensors.numpy import load_file


def wasserstein_1d_np(
    u: np.ndarray,
    v: np.ndarray,
) -> float:
    """Exact 1D Wasserstein Distance: mean |u_(i) - v_(i)| along sorted order statistics."""
    u_sorted = np.sort(u, axis=-1)
    v_sorted = np.sort(v, axis=-1)
    return float(np.mean(np.abs(u_sorted - v_sorted)))


def sliced_wasserstein_np(
    x: np.ndarray,
    y: np.ndarray,
    projections: np.ndarray,
) -> float:
    """Sliced Wasserstein Distance: average 1D Wasserstein distance across 1D projections."""
    # x: [B, N, D], projections: [L, D] -> proj_x: [B, N, L] -> transpose to [B, L, N]
    proj_x = np.matmul(x, projections.T).transpose(0, 2, 1)
    proj_y = np.matmul(y, projections.T).transpose(0, 2, 1)
    sorted_x = np.sort(proj_x, axis=-1)
    sorted_y = np.sort(proj_y, axis=-1)
    return float(np.mean(np.abs(sorted_x - sorted_y)))


def sinkhorn_cost_np(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.5,
    num_iters: int = 20,
) -> float:
    """Entropic Regularized Optimal Transport (Sinkhorn algorithm) under squared Euclidean cost."""
    n = x.shape[0]
    m = y.shape[0]
    # Cost matrix C_ij = ||x_i - y_j||^2
    diff = x[:, None, :] - y[None, :, :]
    c = np.sum(diff ** 2, axis=-1)
    k = np.exp(-c / eps)
    a = np.ones(n, dtype=np.float32) / float(n)
    b = np.ones(m, dtype=np.float32) / float(m)
    v = np.ones(m, dtype=np.float32) / float(m)

    for _ in range(num_iters):
        u = a / (np.dot(k, v) + 1e-12)
        v = b / (np.dot(k.T, u) + 1e-12)

    coupling = u[:, None] * k * v[None, :]
    return float(np.sum(coupling * c))


def debiased_sinkhorn_np(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.5,
    num_iters: int = 20,
) -> float:
    """Debiased Sinkhorn Divergence: S_eps(X, Y) - 0.5 * S_eps(X, X) - 0.5 * S_eps(Y, Y)."""
    s_xy = sinkhorn_cost_np(x, y, eps=eps, num_iters=num_iters)
    s_xx = sinkhorn_cost_np(x, x, eps=eps, num_iters=num_iters)
    s_yy = sinkhorn_cost_np(y, y, eps=eps, num_iters=num_iters)
    return float(s_xy - 0.5 * s_xx - 0.5 * s_yy)


def mmd_np(
    x: np.ndarray,
    y: np.ndarray,
    sigmas: Sequence[float] = (0.5, 1.0, 2.0),
) -> float:
    """Maximum Mean Discrepancy (MMD^2) with multi-scale Gaussian RBF kernels."""
    def rbf_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        k = np.zeros_like(dist_sq)
        for s in sigmas:
            gamma = 1.0 / (2.0 * s * s)
            k += np.exp(-gamma * dist_sq)
        return k

    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)
    return float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))


def energy_distance_np(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Energy Distance / Cramer Distance: 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||."""
    dist_xy = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    dist_xx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    dist_yy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    return float(2.0 * np.mean(dist_xy) - np.mean(dist_xx) - np.mean(dist_yy))


def linear_anchor_np(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Linear baseline mapping."""
    return np.dot(x, weight) + bias


def load_checkpoint_numpy(checkpoint_path: Path) -> Dict[str, np.ndarray]:
    return load_file(str(checkpoint_path))


if __name__ == "__main__":
    ckpt = Path("weights/03_building_blocks_and_losses/10_losses_optimal_transport_wasserstein_sinkhorn_and_mmd.safetensors")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt}")

    data = load_checkpoint_numpy(ckpt)
    print(f"Loaded {len(data)} tensors from {ckpt.name}")

    # 1. Linear Anchor
    pred = linear_anchor_np(data["test_input"], data["weight"], data["bias"])
    err_anchor = float(np.max(np.abs(pred - data["test_output"])))
    assert err_anchor < 1e-5, f"Anchor parity error {err_anchor:.2e} >= 1e-5"

    # 2. 1D Wasserstein
    w1_val = wasserstein_1d_np(data["u_1d"], data["v_1d"])
    err_w1 = abs(w1_val - float(data["expected_w1"][0]))
    assert err_w1 < 1e-5, f"W1 parity error {err_w1:.2e} >= 1e-5"

    # 3. Sliced Wasserstein
    swd_val = sliced_wasserstein_np(data["swd_x"], data["swd_y"], data["projections"])
    err_swd = abs(swd_val - float(data["expected_swd"][0]))
    assert err_swd < 1e-5, f"SWD parity error {err_swd:.2e} >= 1e-5"

    # 4. Sinkhorn Raw
    sink_raw = sinkhorn_cost_np(data["cloud_p"], data["cloud_q"], eps=0.5, num_iters=20)
    err_sink_raw = abs(sink_raw - float(data["expected_sinkhorn_raw"][0]))
    assert err_sink_raw < 1e-5, f"Sinkhorn raw parity error {err_sink_raw:.2e} >= 1e-5"

    # 5. Debiased Sinkhorn
    sink_deb = debiased_sinkhorn_np(data["cloud_p"], data["cloud_q"], eps=0.5, num_iters=20)
    err_sink_deb = abs(sink_deb - float(data["expected_sinkhorn_debiased"][0]))
    assert err_sink_deb < 1e-5, f"Debiased Sinkhorn parity error {err_sink_deb:.2e} >= 1e-5"

    # 6. MMD
    mmd_val = mmd_np(data["cloud_p"], data["cloud_q"], sigmas=(0.5, 1.0, 2.0))
    err_mmd = abs(mmd_val - float(data["expected_mmd"][0]))
    assert err_mmd < 1e-5, f"MMD parity error {err_mmd:.2e} >= 1e-5"

    # 7. Energy Distance
    energy_val = energy_distance_np(data["cloud_p"], data["cloud_q"])
    err_energy = abs(energy_val - float(data["expected_energy"][0]))
    assert err_energy < 1e-5, f"Energy parity error {err_energy:.2e} >= 1e-5"

    print(
        f"All 7 JAX/NumPy invariants verified! Anchor: {err_anchor:.2e}, W1: {err_w1:.2e}, SWD: {err_swd:.2e}, "
        f"SinkRaw: {err_sink_raw:.2e}, SinkDeb: {err_sink_deb:.2e}, MMD: {err_mmd:.2e}, Energy: {err_energy:.2e}"
    )
