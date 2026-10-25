"""Structure Preserving Hamiltonian Port Generic and Divergence Free — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free.
Run ``python jax/06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def compute_potential_and_grad(
    q: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    z = q @ w1 + b1
    a = jnp.tanh(z)
    v = a @ w2 + b2
    temp = (1.0 - a**2) * w2.T
    grad_v = temp @ w1.T
    return v, grad_v


def compute_structure_preserving_all(
    z_state: jnp.ndarray,
    w_v1: jnp.ndarray,
    b_v1: jnp.ndarray,
    w_v2: jnp.ndarray,
    b_v2: jnp.ndarray,
    w_psi1: jnp.ndarray,
    b_psi1: jnp.ndarray,
    w_psi2: jnp.ndarray,
    b_psi2: jnp.ndarray,
    l0_skew: jnp.ndarray,
    m0_psd: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    q, p = z_state[:, :2], z_state[:, 2:]

    v_potential, grad_v = compute_potential_and_grad(q, w_v1, b_v1, w_v2, b_v2)
    t_kinetic = 0.5 * jnp.sum(p**2, axis=-1, keepdims=True)
    h_total = t_kinetic + v_potential

    z_dot_hnn = jnp.concatenate([p, -grad_v], axis=-1)

    gamma = 0.2
    z_dot_phnn = jnp.concatenate([p, -grad_v - gamma * p], axis=-1)

    h = 0.1
    p_half = p - 0.5 * h * grad_v
    q_next = q + h * p_half
    _, grad_v_next = compute_potential_and_grad(q_next, w_v1, b_v1, w_v2, b_v2)
    p_next = p_half - 0.5 * h * grad_v_next
    z_leapfrog = jnp.concatenate([q_next, p_next], axis=-1)

    xy = q
    psi_stream, grad_psi = compute_potential_and_grad(xy, w_psi1, b_psi1, w_psi2, b_psi2)
    v_solenoidal = jnp.stack([grad_psi[:, 1], -grad_psi[:, 0]], axis=-1)

    grad_E = jnp.concatenate([grad_v, p], axis=-1)
    grad_S = -z_state

    eye = jnp.eye(4)
    norm_s_sq = jnp.sum(grad_S**2, axis=-1, keepdims=True)[:, :, None] + 1e-8
    p_s = eye[None, :, :] - (grad_S[:, :, None] @ grad_S[:, None, :]) / norm_s_sq
    l_proj = p_s @ l0_skew[None, :, :] @ p_s

    norm_e_sq = jnp.sum(grad_E**2, axis=-1, keepdims=True)[:, :, None] + 1e-8
    p_e = eye[None, :, :] - (grad_E[:, :, None] @ grad_E[:, None, :]) / norm_e_sq
    m_proj = p_e @ m0_psd[None, :, :] @ p_e

    z_dot_generic = jnp.squeeze(l_proj @ grad_E[:, :, None] + m_proj @ grad_S[:, :, None], axis=-1)
    de_dt = jnp.sum(grad_E * z_dot_generic, axis=-1)
    ds_dt = jnp.sum(grad_S * z_dot_generic, axis=-1)

    return {
        "v_potential": v_potential,
        "grad_v": grad_v,
        "h_total": h_total,
        "z_dot_hnn": z_dot_hnn,
        "z_dot_phnn": z_dot_phnn,
        "z_leapfrog": z_leapfrog,
        "psi_stream": psi_stream,
        "v_solenoidal": v_solenoidal,
        "z_dot_generic": z_dot_generic,
        "de_dt": de_dt,
        "ds_dt": ds_dt,
    }


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    res = compute_structure_preserving_all(
        g["z_state"],
        g["w_v1"],
        g["b_v1"],
        g["w_v2"],
        g["b_v2"],
        g["w_psi1"],
        g["b_psi1"],
        g["w_psi2"],
        g["b_psi2"],
        g["l0_skew"],
        g["m0_psd"],
    )

    for name, computed in res.items():
        diff = float(jnp.max(jnp.abs(computed - g[name])))
        rel = diff / (float(jnp.max(jnp.abs(g[name]))) + 1e-8)
        assert diff < 1e-5 or rel < 1e-5, f"{name}: max abs diff {diff:.2e}, rel diff {rel:.2e}"

    print("06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
