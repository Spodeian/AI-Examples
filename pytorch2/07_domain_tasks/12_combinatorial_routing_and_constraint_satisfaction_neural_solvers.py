"""Combinatorial Routing and Constraint Satisfaction Neural Solvers — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers
Golden weights: weights/07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers.safetensors

``python pytorch2/07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def tsp_tour_cost(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """Computes closed-loop Euclidean tour length for TSP.
    coords: [B, N, 2]
    tour: [B, N] permutation indices
    Returns: [B] total length
    """
    b, n, _ = coords.shape
    # Gather coordinates in tour order: [B, N, 2]
    tour_expanded = tour.unsqueeze(-1).expand(-1, -1, 2)
    ordered_coords = torch.gather(coords, 1, tour_expanded)

    # Rolled coordinates to close the tour: [B, N, 2]
    next_coords = torch.roll(ordered_coords, shifts=-1, dims=1)
    segment_lengths = torch.sqrt(torch.sum((ordered_coords - next_coords) ** 2, dim=-1) + 1e-12)
    return torch.sum(segment_lengths, dim=-1)  # [B]


def pointer_network_step(
    node_embeds: torch.Tensor,
    decoder_state: torch.Tensor,
    visited_mask: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pointer Network attention step to select next node:
    node_embeds: [B, N, D]
    decoder_state: [B, D]
    visited_mask: [B, N] (1.0 for visited, 0.0 for unvisited)
    w1: [D, D], w2: [D, D], v: [D]
    Returns: [B, N] probability distribution over unvisited nodes
    """
    # e1: [B, N, D], e2: [B, 1, D]
    e1 = node_embeds @ w1
    e2 = (decoder_state @ w2).unsqueeze(1)
    scores = torch.tanh(e1 + e2) @ v  # [B, N]

    # Mask visited nodes with -1e9
    masked_scores = scores.masked_fill(visited_mask.bool(), -1e9)
    return F.softmax(masked_scores, dim=-1)


def gnn_edge_heatmap(
    h_nodes: torch.Tensor,
    edge_dists: torch.Tensor,
    w_edge: torch.Tensor,
    b_edge: torch.Tensor,
) -> torch.Tensor:
    """GNN Edge Classification / Tour Heatmap generation:
    h_nodes: [B, N, D]
    edge_dists: [B, N, N]
    w_edge: [2*D + 1, 1]
    b_edge: [1]
    Returns: [B, N, N] edge probability heatmap
    """
    b, n, d = h_nodes.shape
    # Pairwise node features: [B, N, 1, D] and [B, 1, N, D] -> [B, N, N, 2*D]
    hi = h_nodes.unsqueeze(2).expand(-1, -1, n, -1)
    hj = h_nodes.unsqueeze(1).expand(-1, n, -1, -1)
    dist_exp = edge_dists.unsqueeze(-1)  # [B, N, N, 1]

    edge_feats = torch.cat([hi, hj, dist_exp], dim=-1)  # [B, N, N, 2*D + 1]
    logits = edge_feats @ w_edge + b_edge  # [B, N, N, 1]
    return torch.sigmoid(logits.squeeze(-1))  # [B, N, N]


def relaxed_maxcut_cost(z_spins: torch.Tensor, w_adj: torch.Tensor) -> torch.Tensor:
    """Continuous relaxation of Max-Cut Hamiltonian:
    z_spins: [B, N]
    w_adj: [B, N, N]
    sigma = tanh(z_spins) in [-1, 1]
    Cut cost = 0.25 * sum_{i, j} w_{i, j} * (1 - sigma_i * sigma_j)
    Returns: [B] cut value
    """
    sigma = torch.tanh(z_spins)  # [B, N]
    # sigma_i * sigma_j: [B, N, N]
    spin_outer = sigma.unsqueeze(2) * sigma.unsqueeze(1)
    cut_matrix = 0.25 * w_adj * (1.0 - spin_outer)
    return torch.sum(cut_matrix, dim=(-2, -1))


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }

    # 1. TSP Tour Cost
    b, n = 2, 6
    coords = torch.rand(b, n, 2)
    # 2 sample tours
    tour = torch.stack([
        torch.tensor([0, 2, 4, 1, 3, 5], dtype=torch.long),
        torch.tensor([5, 4, 3, 2, 1, 0], dtype=torch.long),
    ])
    tsp_cost_ref = tsp_tour_cost(coords, tour)
    tensors["coords"] = coords
    tensors["tour"] = tour
    tensors["tsp_cost_ref"] = tsp_cost_ref

    # 2. Pointer Network Step
    d_ptr = 8
    node_embeds = torch.randn(b, n, d_ptr)
    decoder_state = torch.randn(b, d_ptr)
    visited_mask = torch.zeros(b, n)
    visited_mask[0, 0] = 1.0
    visited_mask[0, 2] = 1.0
    visited_mask[1, 5] = 1.0

    w1 = torch.randn(d_ptr, d_ptr) * 0.1
    w2 = torch.randn(d_ptr, d_ptr) * 0.1
    v = torch.randn(d_ptr) * 0.1
    ptr_probs_ref = pointer_network_step(node_embeds, decoder_state, visited_mask, w1, w2, v)

    tensors["node_embeds"] = node_embeds
    tensors["decoder_state"] = decoder_state
    tensors["visited_mask"] = visited_mask
    tensors["w1"] = w1
    tensors["w2"] = w2
    tensors["v"] = v
    tensors["ptr_probs_ref"] = ptr_probs_ref

    # 3. GNN Edge Heatmap
    n_gnn, d_gnn = 4, 4
    h_nodes = torch.randn(b, n_gnn, d_gnn)
    edge_dists = torch.abs(torch.randn(b, n_gnn, n_gnn))
    w_edge = torch.randn(2 * d_gnn + 1, 1) * 0.1
    b_edge = torch.randn(1) * 0.1
    heatmap_ref = gnn_edge_heatmap(h_nodes, edge_dists, w_edge, b_edge)

    tensors["h_nodes"] = h_nodes
    tensors["edge_dists"] = edge_dists
    tensors["w_edge"] = w_edge
    tensors["b_edge"] = b_edge
    tensors["heatmap_ref"] = heatmap_ref

    # 4. Relaxed Max-Cut Cost
    n_cut = 6
    z_spins = torch.randn(b, n_cut)
    # Random symmetric adjacency matrix
    rand_adj = torch.rand(b, n_cut, n_cut)
    w_adj = 0.5 * (rand_adj + rand_adj.transpose(1, 2))
    # Zero diagonal
    for bi in range(b):
        w_adj[bi].fill_diagonal_(0.0)

    maxcut_cost_ref = relaxed_maxcut_cost(z_spins, w_adj)
    tensors["z_spins"] = z_spins
    tensors["w_adj"] = w_adj
    tensors["maxcut_cost_ref"] = maxcut_cost_ref

    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
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
