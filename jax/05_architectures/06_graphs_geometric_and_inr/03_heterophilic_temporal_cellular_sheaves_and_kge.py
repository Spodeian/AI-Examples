"""Heterophilic, Temporal, Cellular Sheaves, and Knowledge Graph Embeddings — functional JAX implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge. Run ``python jax/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def heterophilic_filter(
    h: jnp.ndarray,
    norm_adj: jnp.ndarray,
    w_self: jnp.ndarray,
    w_low: jnp.ndarray,
    w_high: jnp.ndarray,
) -> jnp.ndarray:
    n = norm_adj.shape[0]
    high_pass_adj = jnp.eye(n) - norm_adj

    h_self = h @ w_self
    h_low = (norm_adj @ h) @ w_low
    h_high = (high_pass_adj @ h) @ w_high
    return jax.nn.relu(h_self + h_low + h_high)


def cellular_sheaf_laplacian_1d(adj: jnp.ndarray, edge_signs: jnp.ndarray) -> jnp.ndarray:
    deg = jnp.sum(adj, axis=-1)
    d_mat = jnp.diag(deg)
    signed_adj = edge_signs * adj
    return d_mat - signed_adj


def sheaf_diffusion_step(h: jnp.ndarray, sheaf_laplacian: jnp.ndarray, dt: float) -> jnp.ndarray:
    return h - dt * (sheaf_laplacian @ h)


def rotate_kge_score(
    h_re: jnp.ndarray,
    h_im: jnp.ndarray,
    theta: jnp.ndarray,
    t_re: jnp.ndarray,
    t_im: jnp.ndarray,
) -> jnp.ndarray:
    r_re = jnp.cos(theta)
    r_im = jnp.sin(theta)

    rot_re = h_re * r_re - h_im * r_im
    rot_im = h_re * r_im + h_im * r_re

    diff_re = rot_re - t_re
    diff_im = rot_im - t_im

    dist_sq = jnp.sum(diff_re ** 2 + diff_im ** 2, axis=-1, keepdims=True)
    return -jnp.sqrt(dist_sq + 1e-12)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    h_hetero = heterophilic_filter(g["h"], g["norm_adj"], g["w_self"], g["w_low"], g["w_high"])
    assert float(jnp.max(jnp.abs(h_hetero - g["h_hetero"]))) < 1e-5, "h_hetero parity mismatch"

    sheaf_laplacian = cellular_sheaf_laplacian_1d(g["adj"], g["edge_signs"])
    assert float(jnp.max(jnp.abs(sheaf_laplacian - g["sheaf_laplacian"]))) < 1e-5, "sheaf_laplacian parity mismatch"

    h_sheaf_diff = sheaf_diffusion_step(g["h"], sheaf_laplacian, dt=0.1)
    assert float(jnp.max(jnp.abs(h_sheaf_diff - g["h_sheaf_diff"]))) < 1e-5, "h_sheaf_diff parity mismatch"

    rotate_score = rotate_kge_score(g["h_re"], g["h_im"], g["theta"], g["t_re"], g["t_im"])
    assert float(jnp.max(jnp.abs(rotate_score - g["rotate_score"]))) < 1e-5, "rotate_score parity mismatch"

    print("05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
