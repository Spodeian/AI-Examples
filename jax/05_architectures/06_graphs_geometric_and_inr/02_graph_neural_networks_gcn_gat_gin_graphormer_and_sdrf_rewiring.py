"""Graph Neural Networks GCN, GAT, GIN, Graphormer, and SDRF Rewiring — functional JAX implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring. Run ``python jax/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def gcn_normalize_adjacency(adj: jnp.ndarray) -> jnp.ndarray:
    n = adj.shape[0]
    a_tilde = adj + jnp.eye(n)
    deg = jnp.sum(a_tilde, axis=-1)
    deg_inv_sqrt = jnp.where(deg > 0.0, jnp.power(deg, -0.5), 0.0)
    d_mat = jnp.diag(deg_inv_sqrt)
    return d_mat @ a_tilde @ d_mat


def gcn_layer(
    h: jnp.ndarray,
    norm_adj: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    h_proj = h @ weight
    h_conv = norm_adj @ h_proj + bias
    return jax.nn.relu(h_conv)


def gin_layer(
    h: jnp.ndarray,
    adj: jnp.ndarray,
    eps: float,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
) -> jnp.ndarray:
    h_agg = adj @ h
    z = (1.0 + eps) * h + h_agg
    h1 = jax.nn.relu(z @ w1 + b1)
    return h1 @ w2 + b2


def graphormer_attention(
    h: jnp.ndarray,
    w_q: jnp.ndarray,
    w_k: jnp.ndarray,
    w_v: jnp.ndarray,
    spatial_bias: jnp.ndarray,
) -> jnp.ndarray:
    d = h.shape[-1]
    q = h @ w_q
    k = h @ w_k
    v = h @ w_v

    scores = (q @ k.T) / math.sqrt(d) + spatial_bias
    attn = jax.nn.softmax(scores, axis=-1)
    return attn @ v


def sdrf_ricci_curvature(adj: jnp.ndarray) -> jnp.ndarray:
    deg = jnp.sum(adj, axis=-1)
    deg_i = jnp.expand_dims(deg, axis=1)
    deg_j = jnp.expand_dims(deg, axis=0)
    triangles = adj @ adj
    ric = (4.0 - deg_i - deg_j + 3.0 * triangles) * adj
    return ric


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    norm_adj = gcn_normalize_adjacency(g["adj"])
    assert float(jnp.max(jnp.abs(norm_adj - g["norm_adj"]))) < 1e-5, "norm_adj parity mismatch"

    h_gcn = gcn_layer(g["h"], norm_adj, g["w_gcn"], g["b_gcn"])
    assert float(jnp.max(jnp.abs(h_gcn - g["h_gcn"]))) < 1e-5, "h_gcn parity mismatch"

    h_gin = gin_layer(g["h"], g["adj"], 0.05, g["w1_gin"], g["b1_gin"], g["w2_gin"], g["b2_gin"])
    assert float(jnp.max(jnp.abs(h_gin - g["h_gin"]))) < 1e-5, "h_gin parity mismatch"

    h_graphormer = graphormer_attention(g["h"], g["w_q"], g["w_k"], g["w_v"], g["spatial_bias"])
    assert float(jnp.max(jnp.abs(h_graphormer - g["h_graphormer"]))) < 1e-5, "h_graphormer parity mismatch"

    ricci_curv = sdrf_ricci_curvature(g["adj"])
    assert float(jnp.max(jnp.abs(ricci_curv - g["ricci_curv"]))) < 1e-5, "ricci_curv parity mismatch"

    print("05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
