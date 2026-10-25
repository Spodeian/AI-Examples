"""Combinatorial Routing and Constraint Satisfaction Neural Solvers — JAX reference implementation.

Topic: 07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers
Golden weights: weights/07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def tsp_tour_cost(coords: jnp.ndarray, tour: jnp.ndarray) -> jnp.ndarray:
    """Computes closed-loop Euclidean tour length for TSP in JAX."""
    ordered = jnp.take_along_axis(coords, tour[:, :, None], axis=1)
    next_coords = jnp.roll(ordered, shift=-1, axis=1)
    seg_lengths = jnp.sqrt(jnp.sum((ordered - next_coords) ** 2, axis=-1) + 1e-12)
    return jnp.sum(seg_lengths, axis=-1)


def pointer_network_step(
    node_embeds: jnp.ndarray,
    decoder_state: jnp.ndarray,
    visited_mask: jnp.ndarray,
    w1: jnp.ndarray,
    w2: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Pointer Network attention step to select next node in JAX."""
    e1 = node_embeds @ w1
    e2 = jnp.expand_dims(decoder_state @ w2, axis=1)
    scores = jnp.tanh(e1 + e2) @ v
    masked_scores = jnp.where(visited_mask > 0.5, -1e9, scores)
    return jax.nn.softmax(masked_scores, axis=-1)


def gnn_edge_heatmap(
    h_nodes: jnp.ndarray,
    edge_dists: jnp.ndarray,
    w_edge: jnp.ndarray,
    b_edge: jnp.ndarray,
) -> jnp.ndarray:
    """GNN Edge Classification / Tour Heatmap generation in JAX."""
    b, n, d = h_nodes.shape
    hi = jnp.broadcast_to(h_nodes[:, :, None, :], (b, n, n, d))
    hj = jnp.broadcast_to(h_nodes[:, None, :, :], (b, n, n, d))
    dist_exp = edge_dists[:, :, :, None]
    edge_feats = jnp.concatenate([hi, hj, dist_exp], axis=-1)
    logits = edge_feats @ w_edge + b_edge
    return jax.nn.sigmoid(jnp.squeeze(logits, axis=-1))


def relaxed_maxcut_cost(z_spins: jnp.ndarray, w_adj: jnp.ndarray) -> jnp.ndarray:
    """Continuous relaxation of Max-Cut Hamiltonian in JAX."""
    sigma = jnp.tanh(z_spins)
    spin_outer = sigma[:, :, None] * sigma[:, None, :]
    cut_matrix = 0.25 * w_adj * (1.0 - spin_outer)
    return jnp.sum(cut_matrix, axis=(-2, -1))


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    raw = load_file(str(path))
    tensors = {k: jnp.array(v) for k, v in raw.items()}

    # Anchor check
    out = linear_anchor(tensors["test_input"], tensors["weight"], tensors["bias"])
    err = float(jnp.max(jnp.abs(out - tensors["test_output"])))
    assert err < 1e-5, f"test_output parity error: {err}"

    # 1. TSP Tour Cost
    cost = tsp_tour_cost(tensors["coords"], tensors["tour"])
    err_cost = float(jnp.max(jnp.abs(cost - tensors["tsp_cost_ref"])))
    assert err_cost < 1e-5, f"tsp cost parity error: {err_cost}"

    # 2. Pointer Network
    probs = pointer_network_step(
        tensors["node_embeds"],
        tensors["decoder_state"],
        tensors["visited_mask"],
        tensors["w1"],
        tensors["w2"],
        tensors["v"],
    )
    err_ptr = float(jnp.max(jnp.abs(probs - tensors["ptr_probs_ref"])))
    assert err_ptr < 1e-5, f"pointer probs parity error: {err_ptr}"

    # 3. GNN Edge Heatmap
    heatmap = gnn_edge_heatmap(
        tensors["h_nodes"], tensors["edge_dists"], tensors["w_edge"], tensors["b_edge"]
    )
    err_heat = float(jnp.max(jnp.abs(heatmap - tensors["heatmap_ref"])))
    assert err_heat < 1e-5, f"heatmap parity error: {err_heat}"

    # 4. Relaxed Max-Cut Cost
    cut = relaxed_maxcut_cost(tensors["z_spins"], tensors["w_adj"])
    err_cut = float(jnp.max(jnp.abs(cut - tensors["maxcut_cost_ref"])))
    assert err_cut < 1e-5, f"maxcut parity error: {err_cut}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
