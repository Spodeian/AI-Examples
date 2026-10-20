"""Generates reference SafeTensors checkpoint for Category 02 Topic 11:
11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.safetensors
"""

from __future__ import annotations
from pathlib import Path
import torch
import safetensors.torch


def sum_product_tree_marginals(
    unary: torch.Tensor,       # [4, 2]
    edges: torch.Tensor,       # [3, 2, 2] edges: (0-1), (1-2), (1-3)
) -> torch.Tensor:
    """Computes exact marginals for a 4-node tree rooted at node 1 (index 1):
    Nodes: 0 -- 1 -- 2
                |
                3
    """
    # Messages to root 1 from leaves 0, 2, 3:
    # m_{0 -> 1}(x1) = sum_{x0} psi_0(x0) * psi_01(x0, x1)
    # edges[0] is psi_01: shape [2, 2] where [x0, x1]
    m_0_1 = unary[0] @ edges[0] # [2]
    # edges[1] is psi_12: shape [2, 2] where [x1, x2].
    # m_{2 -> 1}(x1) = sum_{x2} psi_2(x2) * psi_12(x1, x2) = edges[1] @ psi_2
    m_2_1 = edges[1] @ unary[2] # [2]
    # edges[2] is psi_13: shape [2, 2] where [x1, x3].
    m_3_1 = edges[2] @ unary[3] # [2]

    # Belief at node 1:
    b1_unnorm = unary[1] * m_0_1 * m_2_1 * m_3_1
    b1 = b1_unnorm / b1_unnorm.sum()

    # Messages from root 1 to leaves:
    # m_{1 -> 0}(x0) = sum_{x1} psi_1(x1) * m_{2 -> 1}(x1) * m_{3 -> 1}(x1) * psi_01(x0, x1)
    prod_1_to_0 = unary[1] * m_2_1 * m_3_1 # [2] (over x1)
    m_1_0 = edges[0] @ prod_1_to_0 # [2] (over x0)
    b0_unnorm = unary[0] * m_1_0
    b0 = b0_unnorm / b0_unnorm.sum()

    # m_{1 -> 2}(x2) = sum_{x1} psi_1(x1) * m_{0 -> 1}(x1) * m_{3 -> 1}(x1) * psi_12(x1, x2)
    prod_1_to_2 = unary[1] * m_0_1 * m_3_1 # [2] (over x1)
    m_1_2 = prod_1_to_2 @ edges[1] # [2] (over x2)
    b2_unnorm = unary[2] * m_1_2
    b2 = b2_unnorm / b2_unnorm.sum()

    # m_{1 -> 3}(x3) = sum_{x1} psi_1(x1) * m_{0 -> 1}(x1) * m_{2 -> 1}(x1) * psi_13(x1, x3)
    prod_1_to_3 = unary[1] * m_0_1 * m_2_1 # [2] (over x1)
    m_1_3 = prod_1_to_3 @ edges[2] # [2] (over x3)
    b3_unnorm = unary[3] * m_1_3
    b3 = b3_unnorm / b3_unnorm.sum()

    return torch.stack([b0, b1, b2, b3])


def max_product_tree_map(
    unary: torch.Tensor,       # [4, 2]
    edges: torch.Tensor,       # [3, 2, 2]
) -> torch.Tensor:
    """Computes exact MAP state configuration using brute force over 2^4 = 16 states."""
    best_config = None
    max_p = -1.0
    for x0 in (0, 1):
        for x1 in (0, 1):
            for x2 in (0, 1):
                for x3 in (0, 1):
                    p = (
                        unary[0, x0] * unary[1, x1] * unary[2, x2] * unary[3, x3]
                        * edges[0, x0, x1] * edges[1, x1, x2] * edges[2, x1, x3]
                    )
                    if p > max_p:
                        max_p = p
                        best_config = [x0, x1, x2, x3]
    return torch.tensor(best_config, dtype=torch.long)


def damped_loopy_bp_triangle(
    unary: torch.Tensor,       # [3, 2]
    edges: torch.Tensor,       # [3, 2, 2]: (0-1), (1-2), (2-0)
    alpha: float = 0.5,
    iters: int = 20,
) -> torch.Tensor:
    """Damped Loopy BP on 3-node cycle: 0-1, 1-2, 2-0."""
    # 6 directed messages: (0->1, 1->0, 1->2, 2->1, 2->0, 0->2)
    # Initialize all messages uniformly:
    m = torch.ones(6, 2) * 0.5

    # Edge indices:
    # 0: 0->1, 1: 1->0
    # 2: 1->2, 3: 2->1
    # 4: 2->0, 5: 0->2

    for _ in range(iters):
        # 0->1: incoming to 0 from 2 is m[4] (2->0)
        raw_0_1 = (unary[0] * m[4]) @ edges[0]
        raw_0_1 = raw_0_1 / raw_0_1.sum()
        m[0] = (1 - alpha) * m[0] + alpha * raw_0_1

        # 1->0: incoming to 1 from 2 is m[3] (2->1)
        raw_1_0 = edges[0] @ (unary[1] * m[3])
        raw_1_0 = raw_1_0 / raw_1_0.sum()
        m[1] = (1 - alpha) * m[1] + alpha * raw_1_0

        # 1->2: incoming to 1 from 0 is m[0] (0->1)
        raw_1_2 = (unary[1] * m[0]) @ edges[1]
        raw_1_2 = raw_1_2 / raw_1_2.sum()
        m[2] = (1 - alpha) * m[2] + alpha * raw_1_2

        # 2->1: incoming to 2 from 0 is m[5] (0->2)
        raw_2_1 = edges[1] @ (unary[2] * m[5])
        raw_2_1 = raw_2_1 / raw_2_1.sum()
        m[3] = (1 - alpha) * m[3] + alpha * raw_2_1

        # 2->0: incoming to 2 from 1 is m[2] (1->2)
        raw_2_0 = (unary[2] * m[2]) @ edges[2]
        raw_2_0 = raw_2_0 / raw_2_0.sum()
        m[4] = (1 - alpha) * m[4] + alpha * raw_2_0

        # 0->2: incoming to 0 from 1 is m[1] (1->0)
        raw_0_2 = edges[2] @ (unary[0] * m[1])
        raw_0_2 = raw_0_2 / raw_0_2.sum()
        m[5] = (1 - alpha) * m[5] + alpha * raw_0_2

    # Compute beliefs:
    b0 = unary[0] * m[1] * m[4]
    b0 = b0 / b0.sum()

    b1 = unary[1] * m[0] * m[3]
    b1 = b1 / b1.sum()

    b2 = unary[2] * m[2] * m[5]
    b2 = b2 / b2.sum()

    return torch.stack([b0, b1, b2])


def compute_cavi_gmm(
    x: torch.Tensor,    # [N]
    mu: torch.Tensor,   # [K]
    sigma_sq: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes CAVI responsibilities and ELBO for fixed cluster means."""
    N = x.shape[0]
    K = mu.shape[0]
    diff = x.unsqueeze(1) - mu.unsqueeze(0) # [N, K]
    log_p = -0.5 * (diff ** 2) / sigma_sq
    # Normalize across K to get responsibilities r_{ik}
    r = torch.softmax(log_p, dim=1)
    
    # ELBO = sum_{i, k} r_{ik} [log_p - log(r_{ik})]
    elbo = torch.sum(r * (log_p - torch.log(torch.clamp(r, min=1e-8))))
    return r, elbo


def generate_checkpoint():
    torch.manual_seed(42)

    # 1. Tree potentials
    unary_tree = torch.tensor([
        [0.7, 0.3],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.5, 0.5],
    ], dtype=torch.float32)

    edge_tree = torch.tensor([
        [[0.8, 0.2], [0.1, 0.9]],  # 0-1
        [[0.7, 0.3], [0.4, 0.6]],  # 1-2
        [[0.6, 0.4], [0.3, 0.7]],  # 1-3
    ], dtype=torch.float32)

    exact_marginals = sum_product_tree_marginals(unary_tree, edge_tree)
    map_config = max_product_tree_map(unary_tree, edge_tree)

    # 2. Triangle graph for Damped Loopy BP
    unary_tri = torch.tensor([
        [0.6, 0.4],
        [0.3, 0.7],
        [0.5, 0.5],
    ], dtype=torch.float32)

    edge_tri = torch.tensor([
        [[0.8, 0.2], [0.2, 0.8]],  # 0-1
        [[0.7, 0.3], [0.3, 0.7]],  # 1-2
        [[0.9, 0.1], [0.1, 0.9]],  # 2-0
    ], dtype=torch.float32)

    loopy_marginals = damped_loopy_bp_triangle(unary_tri, edge_tri, alpha=0.5, iters=20)

    # 3. CAVI
    data_cavi = torch.tensor([-2.5, -2.1, -1.8, -2.0, 1.9, 2.3, 2.1, 2.5], dtype=torch.float32)
    mu_cavi = torch.tensor([-2.0, 2.0], dtype=torch.float32)
    resp, elbo = compute_cavi_gmm(data_cavi, mu_cavi)

    # Golden linear mapping for test_cross_track_parity.py
    D = 4
    test_input = torch.randn(8, D)
    weight = torch.randn(D, D)
    bias = torch.randn(D)
    test_output = test_input @ weight + bias

    checkpoint_data = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "unary_tree": unary_tree.clone().contiguous(),
        "edge_tree": edge_tree.clone().contiguous(),
        "exact_marginals": exact_marginals.clone().contiguous(),
        "map_config": map_config.clone().contiguous(),
        "unary_tri": unary_tri.clone().contiguous(),
        "edge_tri": edge_tri.clone().contiguous(),
        "loopy_marginals": loopy_marginals.clone().contiguous(),
        "data_cavi": data_cavi.clone().contiguous(),
        "mu_cavi": mu_cavi.clone().contiguous(),
        "responsibilities": resp.clone().contiguous(),
        "elbo_val": elbo.unsqueeze(0).clone().contiguous(),
    }

    out_dir = Path("weights/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.safetensors"
    safetensors.torch.save_file(checkpoint_data, out_path)
    print("Saved Topic 11 reference checkpoint to", out_path)


if __name__ == "__main__":
    generate_checkpoint()
