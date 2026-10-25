"""Graph Neural Networks GCN, GAT, GIN, Graphormer, and SDRF Rewiring — PyTorch 2.x reference implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring
Golden weights: weights/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.safetensors

``python pytorch2/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def gcn_normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Computes symmetric normalized adjacency with self-loops: A_hat = D^{-1/2} (A + I) D^{-1/2}."""
    n = adj.shape[0]
    a_tilde = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    deg = torch.sum(a_tilde, dim=-1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    d_mat = torch.diag(deg_inv_sqrt)
    return d_mat @ a_tilde @ d_mat


def gcn_layer(
    h: torch.Tensor,
    norm_adj: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Graph Convolutional layer: H_next = ReLU(A_hat @ H @ W + b)."""
    h_proj = h @ weight
    h_conv = norm_adj @ h_proj + bias
    return F.relu(h_conv)


def gin_layer(
    h: torch.Tensor,
    adj: torch.Tensor,
    eps: float,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Graph Isomorphism Network (GIN) layer: MLP((1 + eps) * h + sum_{j in N(i)} h_j)."""
    h_agg = adj @ h
    z = (1.0 + eps) * h + h_agg
    h1 = F.relu(z @ w1 + b1)
    return h1 @ w2 + b2


def graphormer_attention(
    h: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    spatial_bias: torch.Tensor,
) -> torch.Tensor:
    """Graphormer all-pairs attention with structural spatial bias."""
    d = h.shape[-1]
    q = h @ w_q
    k = h @ w_k
    v = h @ w_v

    scores = (q @ k.T) / math.sqrt(d) + spatial_bias
    attn = F.softmax(scores, dim=-1)
    return attn @ v


def sdrf_ricci_curvature(adj: torch.Tensor) -> torch.Tensor:
    """Balanced Forman-Ricci curvature proxy for edge oversquashing detection:
    Ric(i, j) = 4 - d_i - d_j + 3 * Triangles(i, j) for existing edges.
    """
    deg = torch.sum(adj, dim=-1)  # [N]
    deg_i = deg.unsqueeze(1)      # [N, 1]
    deg_j = deg.unsqueeze(0)      # [1, N]
    triangles = adj @ adj         # [N, N]
    ric = (4.0 - deg_i - deg_j + 3.0 * triangles) * adj
    return ric


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Graph structure: N=4 nodes, D=8 features
    N, D = 4, 8
    # Simple connected 4-node cycle with one chord
    adj = torch.tensor([
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ], dtype=torch.float32)

    h = torch.randn(N, D)

    # GCN
    norm_adj = gcn_normalize_adjacency(adj)
    w_gcn = torch.randn(D, D) * 0.2
    b_gcn = torch.randn(D) * 0.1
    h_gcn = gcn_layer(h, norm_adj, w_gcn, b_gcn)

    # GIN
    D_mid = 16
    w1_gin = torch.randn(D, D_mid) * 0.2
    b1_gin = torch.randn(D_mid) * 0.1
    w2_gin = torch.randn(D_mid, D) * 0.2
    b2_gin = torch.randn(D) * 0.1
    h_gin = gin_layer(h, adj, eps=0.05, w1=w1_gin, b1=b1_gin, w2=w2_gin, b2=b2_gin)

    # Graphormer
    w_q = torch.randn(D, D) * 0.2
    w_k = torch.randn(D, D) * 0.2
    w_v = torch.randn(D, D) * 0.2
    spatial_bias = torch.tensor([
        [0.0, -0.5, -0.5, -1.0],
        [-0.5, 0.0, -0.5, -0.5],
        [-0.5, -0.5, 0.0, -0.5],
        [-1.0, -0.5, -0.5, 0.0],
    ], dtype=torch.float32)
    h_graphormer = graphormer_attention(h, w_q, w_k, w_v, spatial_bias)

    # SDRF curvature
    ricci_curv = sdrf_ricci_curvature(adj)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "adj": adj,
        "h": h,
        "norm_adj": norm_adj,
        "w_gcn": w_gcn,
        "b_gcn": b_gcn,
        "h_gcn": h_gcn,
        "w1_gin": w1_gin,
        "b1_gin": b1_gin,
        "w2_gin": w2_gin,
        "b2_gin": b2_gin,
        "h_gin": h_gin,
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "spatial_bias": spatial_bias,
        "h_graphormer": h_graphormer,
        "ricci_curv": ricci_curv,
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
