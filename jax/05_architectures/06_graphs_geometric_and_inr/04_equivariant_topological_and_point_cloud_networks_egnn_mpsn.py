"""Equivariant, Topological, and Point Cloud Networks: EGNN, MPSN, and PointNet — functional JAX implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn. Run ``python jax/05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def egnn_layer(
    x: jnp.ndarray,
    h: jnp.ndarray,
    adj: jnp.ndarray,
    w_m: jnp.ndarray,
    b_m: jnp.ndarray,
    w_x: jnp.ndarray,
    b_x: jnp.ndarray,
    w_h: jnp.ndarray,
    b_h: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n, d = h.shape
    x_i = jnp.expand_dims(x, axis=1)
    x_j = jnp.expand_dims(x, axis=0)
    delta_x = x_i - x_j
    d_sq = jnp.sum(delta_x ** 2, axis=-1, keepdims=True)

    h_i = jnp.broadcast_to(jnp.expand_dims(h, axis=1), (n, n, d))
    h_j = jnp.broadcast_to(jnp.expand_dims(h, axis=0), (n, n, d))
    m_in = jnp.concatenate([h_i, h_j, d_sq], axis=-1)

    m_ij = jax.nn.silu(m_in @ w_m + b_m) * jnp.expand_dims(adj, axis=-1)
    coord_weights = m_ij @ w_x + b_x
    x_next = x + jnp.sum(delta_x * coord_weights, axis=1)

    m_i = jnp.sum(m_ij, axis=1)
    h_in = jnp.concatenate([h, m_i], axis=-1)
    h_next = h + jax.nn.silu(h_in @ w_h + b_h)

    return x_next, h_next


def pointnet_global_pool(x: jnp.ndarray, w_p: jnp.ndarray, b_p: jnp.ndarray) -> jnp.ndarray:
    f = jax.nn.relu(x @ w_p + b_p)
    return jnp.max(f, axis=0, keepdims=True)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    x_next, h_next = egnn_layer(g["x"], g["h"], g["adj"], g["w_m"], g["b_m"], g["w_x"], g["b_x"], g["w_h"], g["b_h"])
    assert float(jnp.max(jnp.abs(x_next - g["x_next"]))) < 1e-5, "x_next parity mismatch"
    assert float(jnp.max(jnp.abs(h_next - g["h_next"]))) < 1e-5, "h_next parity mismatch"

    p_global = pointnet_global_pool(g["x"], g["w_p"], g["b_p"])
    assert float(jnp.max(jnp.abs(p_global - g["p_global"]))) < 1e-5, "p_global parity mismatch"

    print("05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
