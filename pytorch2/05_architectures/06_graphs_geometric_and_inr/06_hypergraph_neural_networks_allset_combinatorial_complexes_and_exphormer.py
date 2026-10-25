"""Hypergraph Neural Networks: AllSet, Combinatorial Complexes, and Exphormer — PyTorch 2.x reference implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer
Golden weights: weights/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.safetensors

``python pytorch2/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def hypergraph_conv(
    x: torch.Tensor,
    inc: torch.Tensor,
    w_feat: torch.Tensor,
    b_feat: torch.Tensor,
) -> torch.Tensor:
    """Hypergraph convolution via bipartite incidence matrix H in R^{V x E}:
    X_next = GELU(D_v^{-1} H D_e^{-1} H^T X @ W + b).
    """
    d_v = torch.sum(inc, dim=1, keepdim=True)  # [V, 1]
    d_e = torch.sum(inc, dim=0, keepdim=True)  # [1, E]

    # Node-to-Hyperedge aggregation: D_e^{-1} H^T X
    z_e = (inc.T @ x) / torch.clamp(d_e.T, min=1e-6)  # [E, D]

    # Hyperedge-to-Node aggregation: D_v^{-1} H Z_e
    x_agg = (inc @ z_e) / torch.clamp(d_v, min=1e-6)  # [V, D]

    # Feature transformation
    return F.gelu(x_agg @ w_feat + b_feat)


def allset_multiset_step(
    x: torch.Tensor,
    inc: torch.Tensor,
    w_v: torch.Tensor,
    b_v: torch.Tensor,
    w_e: torch.Tensor,
    b_e: torch.Tensor,
) -> torch.Tensor:
    """AllSet two-phase universal multiset propagation:
    Z_e = MLP_e(H^T @ MLP_v(X))
    X_next = X + MLP_v_out(H @ Z_e).
    """
    h_v = F.relu(x @ w_v + b_v)
    z_e = F.relu((inc.T @ h_v) @ w_e + b_e)
    x_next = x + F.relu((inc @ z_e) @ w_v + b_v)
    return x_next


def exphormer_spectral_mixing(
    h: torch.Tensor,
    expander_adj: torch.Tensor,
    w_mix: torch.Tensor,
    b_mix: torch.Tensor,
) -> torch.Tensor:
    """Exphormer expander graph sparse message mixing step: GELU((H + A_exp @ H) @ W + b)."""
    h_exp = expander_adj @ h
    return F.gelu((h + h_exp) @ w_mix + b_mix)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Hypergraph structure: V=4 nodes, E=3 hyperedges, D=8
    V, E, D = 4, 3, 8
    # e0 connects {0, 1, 2}, e1 connects {1, 2, 3}, e2 connects {0, 3}
    inc = torch.tensor([
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ], dtype=torch.float32)

    x = torch.randn(V, D)

    # Hypergraph convolution
    w_hg = torch.randn(D, D) * 0.2
    b_hg = torch.randn(D) * 0.1
    h_hg = hypergraph_conv(x, inc, w_hg, b_hg)

    # AllSet step
    w_v = torch.randn(D, D) * 0.2
    b_v = torch.randn(D) * 0.1
    w_e = torch.randn(D, D) * 0.2
    b_e = torch.randn(D) * 0.1
    h_allset = allset_multiset_step(x, inc, w_v, b_v, w_e, b_e)

    # Exphormer expander graph (2-regular cycle on 4 nodes)
    exp_adj = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float32)

    w_exp = torch.randn(D, D) * 0.2
    b_exp = torch.randn(D) * 0.1
    h_exphormer = exphormer_spectral_mixing(x, exp_adj, w_exp, b_exp)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "inc": inc,
        "x": x,
        "w_hg": w_hg,
        "b_hg": b_hg,
        "h_hg": h_hg,
        "w_v": w_v,
        "b_v": b_v,
        "w_e": w_e,
        "b_e": b_e,
        "h_allset": h_allset,
        "exp_adj": exp_adj,
        "w_exp": w_exp,
        "b_exp": b_exp,
        "h_exphormer": h_exphormer,
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
