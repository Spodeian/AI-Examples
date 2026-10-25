"""Hypergraph Neural Networks: AllSet, Combinatorial Complexes, and Exphormer — functional JAX implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer. Run ``python jax/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def hypergraph_conv(
    x: jnp.ndarray,
    inc: jnp.ndarray,
    w_feat: jnp.ndarray,
    b_feat: jnp.ndarray,
) -> jnp.ndarray:
    d_v = jnp.sum(inc, axis=1, keepdims=True)
    d_e = jnp.sum(inc, axis=0, keepdims=True)

    z_e = (inc.T @ x) / jnp.clip(d_e.T, a_min=1e-6)
    x_agg = (inc @ z_e) / jnp.clip(d_v, a_min=1e-6)

    return jax.nn.gelu(x_agg @ w_feat + b_feat, approximate=False)


def allset_multiset_step(
    x: jnp.ndarray,
    inc: jnp.ndarray,
    w_v: jnp.ndarray,
    b_v: jnp.ndarray,
    w_e: jnp.ndarray,
    b_e: jnp.ndarray,
) -> jnp.ndarray:
    h_v = jax.nn.relu(x @ w_v + b_v)
    z_e = jax.nn.relu((inc.T @ h_v) @ w_e + b_e)
    x_next = x + jax.nn.relu((inc @ z_e) @ w_v + b_v)
    return x_next


def exphormer_spectral_mixing(
    h: jnp.ndarray,
    expander_adj: jnp.ndarray,
    w_mix: jnp.ndarray,
    b_mix: jnp.ndarray,
) -> jnp.ndarray:
    h_exp = expander_adj @ h
    return jax.nn.gelu((h + h_exp) @ w_mix + b_mix, approximate=False)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    h_hg = hypergraph_conv(g["x"], g["inc"], g["w_hg"], g["b_hg"])
    assert float(jnp.max(jnp.abs(h_hg - g["h_hg"]))) < 1e-5, "h_hg parity mismatch"

    h_allset = allset_multiset_step(g["x"], g["inc"], g["w_v"], g["b_v"], g["w_e"], g["b_e"])
    assert float(jnp.max(jnp.abs(h_allset - g["h_allset"]))) < 1e-5, "h_allset parity mismatch"

    h_exphormer = exphormer_spectral_mixing(g["x"], g["exp_adj"], g["w_exp"], g["b_exp"])
    assert float(jnp.max(jnp.abs(h_exphormer - g["h_exphormer"]))) < 1e-5, "h_exphormer parity mismatch"

    print("05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
