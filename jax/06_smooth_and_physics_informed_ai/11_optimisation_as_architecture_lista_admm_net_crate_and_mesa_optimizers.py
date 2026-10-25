"""Optimisation as Architecture LISTA ADMM-Net CRATE and Mesa-Optimizers — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers. Run ``python jax/06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def soft_threshold(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.clip(jnp.abs(x) - theta, a_min=0.0)


def lista_unroll(
    y: jnp.ndarray,
    w_e: jnp.ndarray,
    w_s: jnp.ndarray,
    theta: jnp.ndarray,
    num_layers: int = 3,
) -> jnp.ndarray:
    B = y.shape[0]
    N = w_e.shape[1]
    z = jnp.zeros((B, N), dtype=y.dtype)
    for _ in range(num_layers):
        lin = z @ w_s + y @ w_e
        z = soft_threshold(lin, theta)
    return z


def admm_net_step(
    y: jnp.ndarray,
    x: jnp.ndarray,
    z: jnp.ndarray,
    u: jnp.ndarray,
    w_inv: jnp.ndarray,
    rho: float = 1.0,
    theta: float = 0.08,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rhs = y + rho * (z - u)
    x_next = rhs @ w_inv
    z_next = soft_threshold(x_next + u, jnp.asarray(theta / rho))
    u_next = u + x_next - z_next
    return x_next, z_next, u_next


def crate_step(
    z: jnp.ndarray,
    d_dict: jnp.ndarray,
    step_size: float = 0.1,
    lambda_reg: float = 0.04,
) -> jnp.ndarray:
    dtd = d_dict.T @ d_dict
    gram_eye = dtd - jnp.eye(d_dict.shape[1], dtype=z.dtype)
    grad = z @ gram_eye
    z_step = z - step_size * grad
    return soft_threshold(z_step, jnp.asarray(lambda_reg))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"
    
    z_lista = lista_unroll(g["y_meas"], g["w_e"], g["w_s"], g["theta_lista"], num_layers=3)
    assert float(jnp.max(jnp.abs(z_lista - g["z_lista"]))) < 1e-5
    
    x0 = jnp.zeros_like(g["x1"])
    z0 = jnp.zeros_like(g["z1"])
    u0 = jnp.zeros_like(g["u1"])
    x1, z1, u1 = admm_net_step(g["y_admm"], x0, z0, u0, g["w_inv"], rho=1.0, theta=0.08)
    assert float(jnp.max(jnp.abs(x1 - g["x1"]))) < 1e-5
    assert float(jnp.max(jnp.abs(z1 - g["z1"]))) < 1e-5
    assert float(jnp.max(jnp.abs(u1 - g["u1"]))) < 1e-5
    
    z_crate = crate_step(g["z_crate_in"], g["d_dict"], step_size=0.1, lambda_reg=0.04)
    assert float(jnp.max(jnp.abs(z_crate - g["z_crate_out"]))) < 1e-5
    
    print("06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
