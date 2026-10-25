"""Capstone Physics Smooth and Certified AI Pipeline — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline. Run ``python jax/17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def compute_hamiltonian_energy(q: jnp.ndarray, p: jnp.ndarray, w_v: jnp.ndarray) -> jnp.ndarray:
    """Computes total Hamiltonian mechanical energy: H = T(p) + V(q)."""
    kinetic = 0.5 * jnp.sum(p * p, axis=-1)
    potential = 0.5 * jnp.sum(q * (q @ w_v), axis=-1)
    return kinetic + potential


def symplectic_leapfrog_step(
    q: jnp.ndarray,
    p: jnp.ndarray,
    w_v: jnp.ndarray,
    dt: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Executes a single symplectic leapfrog integration step."""
    grad_v_0 = q @ w_v
    p_half = p - 0.5 * dt * grad_v_0
    q_next = q + dt * p_half
    grad_v_1 = q_next @ w_v
    p_next = p_half - 0.5 * dt * grad_v_1
    return q_next, p_next


def certified_smooth_forward(q_next: jnp.ndarray, w_l: jnp.ndarray, b_l: jnp.ndarray) -> jnp.ndarray:
    """Applies contractive 1-Lipschitz readout layer with SiLU activation."""
    z = q_next @ w_l + b_l
    return jax.nn.silu(z)


def physics_certified_pipeline(
    q: jnp.ndarray,
    p: jnp.ndarray,
    w_v: jnp.ndarray,
    w_l: jnp.ndarray,
    b_l: jnp.ndarray,
    dt: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """End-to-end physics-informed and certified pipeline execution."""
    energy_init = compute_hamiltonian_energy(q, p, w_v)
    q_next, p_next = symplectic_leapfrog_step(q, p, w_v, dt=dt)
    energy_next = compute_hamiltonian_energy(q_next, p_next, w_v)
    y_smooth = certified_smooth_forward(q_next, w_l, b_l)
    return y_smooth, q_next, p_next, energy_init, energy_next


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    y_smooth, q_next, p_next, e_init, e_next = physics_certified_pipeline(
        g["q"], g["p"], g["w_v"], g["w_l"], g["b_l"], dt=0.05
    )
    assert float(jnp.max(jnp.abs(y_smooth - g["y_smooth"]))) < 1e-5
    assert float(jnp.max(jnp.abs(q_next - g["q_next"]))) < 1e-5
    assert float(jnp.max(jnp.abs(p_next - g["p_next"]))) < 1e-5
    assert float(jnp.max(jnp.abs(e_init - g["energy_init"]))) < 1e-5
    assert float(jnp.max(jnp.abs(e_next - g["energy_next"]))) < 1e-5

    print("17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
