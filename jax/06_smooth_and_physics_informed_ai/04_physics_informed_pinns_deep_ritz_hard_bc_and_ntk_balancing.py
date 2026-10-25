"""Physics Informed PINNs Deep Ritz Hard Bc and NTK Balancing — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing. Run ``python jax/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def compute_pinns_all(
    x_colloc: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Evaluates strong-form PINN, Hard BC, Deep Ritz energy, and NTK gradient balancing."""
    z = x_colloc @ w1 + b1
    a = jnp.tanh(z)
    u_raw = a @ w2 + b2

    psi = (1.0 - a**2) * w1
    u_x = psi @ w2
    phi = (-2.0 * a * (1.0 - a**2)) * (w1**2)
    u_xx = phi @ w2

    f_source = (jnp.pi**2) * jnp.sin(jnp.pi * x_colloc)
    pde_residual = -u_xx - f_source
    loss_pde = jnp.mean(pde_residual**2, keepdims=True)

    d = x_colloc * (1.0 - x_colloc)
    d_x = 1.0 - 2.0 * x_colloc
    u_hard = d * u_raw
    u_hard_x = d_x * u_raw + d * u_x

    ritz_energy = jnp.mean(0.5 * (u_hard_x**2) - f_source * u_hard, keepdims=True)

    x_bc = jnp.array([[0.0], [1.0]])
    z_bc = x_bc @ w1 + b1
    a_bc = jnp.tanh(z_bc)
    u_bc = a_bc @ w2 + b2
    g_bc_target = jnp.array([[0.0], [0.0]])
    loss_bc = jnp.mean((u_bc - g_bc_target) ** 2, keepdims=True)

    g_pde = (-2.0 / x_colloc.shape[0]) * (phi.T @ pde_residual)
    g_bc = (2.0 / x_bc.shape[0]) * (a_bc.T @ (u_bc - g_bc_target))

    norm_pde = jnp.linalg.norm(g_pde)
    norm_bc = jnp.linalg.norm(g_bc)
    lambda_ntk = jnp.array([norm_pde / (norm_bc + 1e-8)])
    g_balanced = g_pde + lambda_ntk * g_bc

    return {
        "u_raw": u_raw,
        "u_x": u_x,
        "u_xx": u_xx,
        "pde_residual": pde_residual,
        "loss_pde": loss_pde,
        "u_hard": u_hard,
        "u_hard_x": u_hard_x,
        "ritz_energy": ritz_energy,
        "loss_bc": loss_bc,
        "g_pde": g_pde,
        "g_bc": g_bc,
        "lambda_ntk": lambda_ntk,
        "g_balanced": g_balanced,
    }


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    res = compute_pinns_all(g["x_colloc"], g["w1"], g["b1"], g["w2"], g["b2"])
    for name, computed in res.items():
        diff = float(jnp.max(jnp.abs(computed - g[name])))
        rel = diff / (float(jnp.max(jnp.abs(g[name]))) + 1e-8)
        assert diff < 5e-5 or rel < 1e-5, f"{name}: max abs diff {diff:.2e}, rel diff {rel:.2e}"

    print("06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
