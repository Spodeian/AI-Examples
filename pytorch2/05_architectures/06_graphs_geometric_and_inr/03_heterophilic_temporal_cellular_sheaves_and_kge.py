"""Heterophilic, Temporal, Cellular Sheaves, and Knowledge Graph Embeddings — PyTorch 2.x reference implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge
Golden weights: weights/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.safetensors

``python pytorch2/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def heterophilic_filter(
    h: torch.Tensor,
    norm_adj: torch.Tensor,
    w_self: torch.Tensor,
    w_low: torch.Tensor,
    w_high: torch.Tensor,
) -> torch.Tensor:
    """Decoupled high-pass / low-pass spatial filtering for heterophilic graphs:
    H_next = ReLU(H @ W_self + (A_hat @ H) @ W_low + ((I - A_hat) @ H) @ W_high).
    """
    n = norm_adj.shape[0]
    eye = torch.eye(n, device=norm_adj.device, dtype=norm_adj.dtype)
    high_pass_adj = eye - norm_adj

    h_self = h @ w_self
    h_low = (norm_adj @ h) @ w_low
    h_high = (high_pass_adj @ h) @ w_high
    return F.relu(h_self + h_low + h_high)


def cellular_sheaf_laplacian_1d(adj: torch.Tensor, edge_signs: torch.Tensor) -> torch.Tensor:
    """Constructs 1D orthogonal cellular sheaf Laplacian: L_F = D - S * A."""
    deg = torch.sum(adj, dim=-1)
    d_mat = torch.diag(deg)
    signed_adj = edge_signs * adj
    return d_mat - signed_adj


def sheaf_diffusion_step(h: torch.Tensor, sheaf_laplacian: torch.Tensor, dt: float) -> torch.Tensor:
    """Simulates one step of continuous cellular sheaf diffusion: H_next = H - dt * L_F @ H."""
    return h - dt * (sheaf_laplacian @ h)


def rotate_kge_score(
    h_re: torch.Tensor,
    h_im: torch.Tensor,
    theta: torch.Tensor,
    t_re: torch.Tensor,
    t_im: torch.Tensor,
) -> torch.Tensor:
    """RotatE Knowledge Graph Embedding score: -||(h o r) - t||_2 in complex space."""
    r_re = torch.cos(theta)
    r_im = torch.sin(theta)

    # Complex Hadamard multiplication: (h_re + i*h_im) * (r_re + i*r_im)
    rot_re = h_re * r_re - h_im * r_im
    rot_im = h_re * r_im + h_im * r_re

    diff_re = rot_re - t_re
    diff_im = rot_im - t_im

    dist_sq = torch.sum(diff_re ** 2 + diff_im ** 2, dim=-1, keepdim=True)
    return -torch.sqrt(dist_sq + 1e-12)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Heterophilic Graph: N=4 nodes, D=8 features
    N, D = 4, 8
    adj = torch.tensor([
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ], dtype=torch.float32)

    # Symmetric normalized adjacency
    deg = torch.sum(adj, dim=-1)
    d_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    norm_adj = d_inv_sqrt @ adj @ d_inv_sqrt

    h = torch.randn(N, D)
    w_self = torch.randn(D, D) * 0.2
    w_low = torch.randn(D, D) * 0.2
    w_high = torch.randn(D, D) * 0.2

    h_hetero = heterophilic_filter(h, norm_adj, w_self, w_low, w_high)

    # Cellular Sheaf with signed connections (+1 homophilic, -1 heterophilic)
    edge_signs = torch.tensor([
        [0.0, 1.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0],
    ], dtype=torch.float32)

    sheaf_laplacian = cellular_sheaf_laplacian_1d(adj, edge_signs)
    h_sheaf_diff = sheaf_diffusion_step(h, sheaf_laplacian, dt=0.1)

    # RotatE Knowledge Graph Embedding: B=2 triplets, D_kge=4
    B_kge, D_kge = 2, 4
    h_re = torch.randn(B_kge, D_kge)
    h_im = torch.randn(B_kge, D_kge)
    theta = torch.randn(B_kge, D_kge)
    t_re = torch.randn(B_kge, D_kge)
    t_im = torch.randn(B_kge, D_kge)

    rotate_score = rotate_kge_score(h_re, h_im, theta, t_re, t_im)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "adj": adj,
        "norm_adj": norm_adj,
        "h": h,
        "w_self": w_self,
        "w_low": w_low,
        "w_high": w_high,
        "h_hetero": h_hetero,
        "edge_signs": edge_signs,
        "sheaf_laplacian": sheaf_laplacian,
        "h_sheaf_diff": h_sheaf_diff,
        "h_re": h_re,
        "h_im": h_im,
        "theta": theta,
        "t_re": t_re,
        "t_im": t_im,
        "rotate_score": rotate_score,
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
