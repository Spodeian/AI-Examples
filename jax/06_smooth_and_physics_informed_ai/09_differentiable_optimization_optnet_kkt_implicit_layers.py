"""Differentiable Optimization OptNet KKT Implicit Layers — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers. Run ``python jax/06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def solve_equality_qp(
    Q: jnp.ndarray,
    p: jnp.ndarray,
    A: jnp.ndarray,
    b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    B, D = p.shape
    M = b.shape[1]
    
    KKT = jnp.zeros((D + M, D + M), dtype=Q.dtype)
    KKT = KKT.at[:D, :D].set(Q)
    KKT = KKT.at[:D, D:].set(A.T)
    KKT = KKT.at[D:, :D].set(A)
    
    rhs = jnp.concatenate([-p, b], axis=-1)
    sol = jnp.linalg.solve(KKT, rhs.T).T
    z_star = sol[:, :D]
    nu_star = sol[:, D:]
    return z_star, nu_star


def optnet_backward_step(
    Q: jnp.ndarray,
    A: jnp.ndarray,
    grad_z: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B, D = grad_z.shape
    M = A.shape[0]
    
    KKT = jnp.zeros((D + M, D + M), dtype=Q.dtype)
    KKT = KKT.at[:D, :D].set(Q)
    KKT = KKT.at[:D, D:].set(A.T)
    KKT = KKT.at[D:, :D].set(A)
    
    rhs_adj = jnp.concatenate([-grad_z, jnp.zeros((B, M), dtype=Q.dtype)], axis=-1)
    sol_adj = jnp.linalg.solve(KKT, rhs_adj.T).T
    d_z = sol_adj[:, :D]
    d_nu = sol_adj[:, D:]
    
    grad_p = -d_z
    grad_b = -d_nu
    return d_z, grad_p, grad_b


def barrier_qp_solve(
    Q: jnp.ndarray,
    p: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
    num_steps: int = 20,
    eta: float = 0.08,
    t_barrier: float = 5.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    B, D = p.shape
    z = jnp.zeros((B, D), dtype=Q.dtype)
    
    for _ in range(num_steps):
        slacks = h - z @ G.T
        safe_slacks = jnp.clip(slacks, a_min=1e-3)
        dual_lambda = 1.0 / (t_barrier * safe_slacks)
        grad_barrier = dual_lambda @ G
        grad_total = z @ Q + p + grad_barrier
        z = z - eta * grad_total
        
    final_slacks = jnp.clip(h - z @ G.T, a_min=1e-4)
    final_lambda = 1.0 / (t_barrier * final_slacks)
    return z, final_lambda


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"
    
    z_eq, nu_eq = solve_equality_qp(g["Q"], g["p"], g["A"], g["b"])
    assert float(jnp.max(jnp.abs(z_eq - g["z_eq"]))) < 1e-5
    assert float(jnp.max(jnp.abs(nu_eq - g["nu_eq"]))) < 1e-5
    
    d_z, grad_p, grad_b = optnet_backward_step(g["Q"], g["A"], g["grad_loss_z"])
    assert float(jnp.max(jnp.abs(d_z - g["d_z"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_p - g["grad_p"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_b - g["grad_b"]))) < 1e-5
    
    z_barrier, lambda_barrier = barrier_qp_solve(g["Q"], g["p"], g["G"], g["h_bound"], num_steps=20, eta=0.08, t_barrier=5.0)
    assert float(jnp.max(jnp.abs(z_barrier - g["z_barrier"]))) < 1e-5
    assert float(jnp.max(jnp.abs(lambda_barrier - g["lambda_barrier"]))) < 1e-5
    
    print("06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
