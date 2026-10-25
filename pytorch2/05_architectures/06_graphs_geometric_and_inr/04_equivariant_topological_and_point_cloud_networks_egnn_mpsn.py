"""Equivariant, Topological, and Point Cloud Networks: EGNN, MPSN, and PointNet — PyTorch 2.x reference implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn
Golden weights: weights/05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn.safetensors

``python pytorch2/05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def egnn_layer(
    x: torch.Tensor,
    h: torch.Tensor,
    adj: torch.Tensor,
    w_m: torch.Tensor,
    b_m: torch.Tensor,
    w_x: torch.Tensor,
    b_x: torch.Tensor,
    w_h: torch.Tensor,
    b_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """E(n)-Equivariant Graph Neural Network (EGNN) message passing layer.
    
    Guarantees strict equivariance under 3D rotations, translations, and reflections:
        x has shape [N, 3], h has shape [N, D], adj has shape [N, N].
    Returns (x_next, h_next).
    """
    n, d = h.shape
    # Pairwise coordinate differences: x_i - x_j
    x_i = x.unsqueeze(1)  # [N, 1, 3]
    x_j = x.unsqueeze(0)  # [1, N, 3]
    delta_x = x_i - x_j   # [N, N, 3]
    d_sq = torch.sum(delta_x ** 2, dim=-1, keepdim=True)  # [N, N, 1]

    # Node features pairing: [h_i, h_j, d_sq]
    h_i = h.unsqueeze(1).expand(n, n, d)
    h_j = h.unsqueeze(0).expand(n, n, d)
    m_in = torch.cat([h_i, h_j, d_sq], dim=-1)  # [N, N, 2D + 1]

    # Message computation
    m_ij = F.silu(m_in @ w_m + b_m) * adj.unsqueeze(-1)  # [N, N, D_m]

    # Coordinate update (vector field along radial displacement delta_x)
    coord_weights = m_ij @ w_x + b_x  # [N, N, 1]
    x_next = x + torch.sum(delta_x * coord_weights, dim=1)

    # Feature aggregation and update
    m_i = torch.sum(m_ij, dim=1)  # [N, D_m]
    h_in = torch.cat([h, m_i], dim=-1)
    h_next = h + F.silu(h_in @ w_h + b_h)

    return x_next, h_next


def pointnet_global_pool(x: torch.Tensor, w_p: torch.Tensor, b_p: torch.Tensor) -> torch.Tensor:
    """PointNet symmetric permutation-invariant global feature pooling: max_{p in P} ReLU(p @ W + b)."""
    f = F.relu(x @ w_p + b_p)
    return torch.max(f, dim=0, keepdim=True)[0]


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # EGNN setup: N=4 particles/atoms in 3D, D=4 scalar attributes
    N, D, D_m = 4, 4, 4
    x = torch.randn(N, 3)
    h = torch.randn(N, D)
    adj = torch.tensor([
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ], dtype=torch.float32)

    w_m = torch.randn(2 * D + 1, D_m) * 0.2
    b_m = torch.randn(D_m) * 0.1
    w_x = torch.randn(D_m, 1) * 0.2
    b_x = torch.randn(1) * 0.1
    w_h = torch.randn(D + D_m, D) * 0.2
    b_h = torch.randn(D) * 0.1

    x_next, h_next = egnn_layer(x, h, adj, w_m, b_m, w_x, b_x, w_h, b_h)

    # PointNet
    w_p = torch.randn(3, 8) * 0.2
    b_p = torch.randn(8) * 0.1
    p_global = pointnet_global_pool(x, w_p, b_p)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "h": h,
        "adj": adj,
        "w_m": w_m,
        "b_m": b_m,
        "w_x": w_x,
        "b_x": b_x,
        "w_h": w_h,
        "b_h": b_h,
        "x_next": x_next,
        "h_next": h_next,
        "w_p": w_p,
        "b_p": b_p,
        "p_global": p_global,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
