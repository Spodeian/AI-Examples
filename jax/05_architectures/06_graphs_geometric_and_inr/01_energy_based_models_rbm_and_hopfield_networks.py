"""Energy-Based Models, RBM, and Hopfield Networks — functional JAX implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks. Run ``python jax/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def rbm_free_energy(v: jnp.ndarray, w: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    v_bias = jnp.expand_dims(v @ a, axis=1)
    h_act = jax.nn.softplus(v @ w + b)
    h_term = jnp.sum(h_act, axis=-1, keepdims=True)
    return -v_bias - h_term


def langevin_dynamics_step(
    x: jnp.ndarray,
    grad_energy: jnp.ndarray,
    step_size: float,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    return x - 0.5 * step_size * grad_energy + math.sqrt(step_size) * noise


def modern_hopfield_energy(x: jnp.ndarray, memories: jnp.ndarray, beta: float) -> jnp.ndarray:
    sim = x @ memories.T
    lse = jax.nn.logsumexp(beta * sim, axis=-1, keepdims=True) / beta
    norm_term = 0.5 * jnp.sum(x ** 2, axis=-1, keepdims=True)
    return -lse + norm_term


def modern_hopfield_retrieve(query: jnp.ndarray, memories: jnp.ndarray, beta: float) -> jnp.ndarray:
    scores = query @ memories.T
    weights = jax.nn.softmax(beta * scores, axis=-1)
    return weights @ memories


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    free_energy = rbm_free_energy(g["v"], g["w_rbm"], g["a_rbm"], g["b_rbm"])
    assert float(jnp.max(jnp.abs(free_energy - g["free_energy"]))) < 1e-5, "free_energy parity mismatch"

    x_langevin = langevin_dynamics_step(g["x_ebm"], g["grad_energy"], 0.01, g["noise"])
    assert float(jnp.max(jnp.abs(x_langevin - g["x_langevin"]))) < 1e-5, "x_langevin parity mismatch"

    hopfield_energy = modern_hopfield_energy(g["x_ebm"], g["memories"], beta=2.0)
    assert float(jnp.max(jnp.abs(hopfield_energy - g["hopfield_energy"]))) < 1e-5, "hopfield_energy parity mismatch"

    retrieved = modern_hopfield_retrieve(g["x_ebm"], g["memories"], beta=2.0)
    assert float(jnp.max(jnp.abs(retrieved - g["retrieved"]))) < 1e-5, "retrieved parity mismatch"

    print("05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
