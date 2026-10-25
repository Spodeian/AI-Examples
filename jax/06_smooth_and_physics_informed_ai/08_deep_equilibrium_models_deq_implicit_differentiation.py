"""Deep Equilibrium Models DEQ Implicit Differentiation — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation. Run ``python jax/06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def f_theta(z: jnp.ndarray, x: jnp.ndarray, w_z: jnp.ndarray, w_x: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(z @ w_z + x @ w_x + b)


def fixed_point_iteration(
    x: jnp.ndarray,
    w_z: jnp.ndarray,
    w_x: jnp.ndarray,
    b: jnp.ndarray,
    num_iters: int = 20,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    z = jnp.zeros((x.shape[0], w_z.shape[0]), dtype=x.dtype)
    for _ in range(num_iters):
        z = f_theta(z, x, w_z, w_x, b)
    res = z - f_theta(z, x, w_z, w_x, b)
    return z, res


def anderson_acceleration(
    x: jnp.ndarray,
    w_z: jnp.ndarray,
    w_x: jnp.ndarray,
    b: jnp.ndarray,
    m: int = 2,
    num_iters: int = 15,
    beta: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    B, D = x.shape[0], w_z.shape[0]
    z = jnp.zeros((B, D), dtype=x.dtype)
    z_hist = []
    g_hist = []
    
    for k in range(num_iters):
        g = f_theta(z, x, w_z, w_x, b)
        z_hist.append(z)
        g_hist.append(g)
        if len(z_hist) > m + 1:
            z_hist.pop(0)
            g_hist.pop(0)
            
        k_hist = len(z_hist) - 1
        if k_hist == 0:
            z = g
        else:
            z_next_list = []
            for b_idx in range(B):
                R_mat = jnp.stack([g_hist[j][b_idx] - z_hist[j][b_idx] for j in range(k_hist + 1)], axis=1)
                d_R = R_mat[:, 1:] - R_mat[:, :-1]
                reg = 1e-5 * jnp.eye(k_hist, dtype=x.dtype)
                rhs = d_R.T @ R_mat[:, -1:]
                lhs = d_R.T @ d_R + reg
                alpha = jnp.linalg.solve(lhs, rhs).squeeze(1)
                
                g_diff = jnp.stack([g_hist[j+1][b_idx] - g_hist[j][b_idx] for j in range(k_hist)], axis=1)
                z_b = g_hist[-1][b_idx] - (g_diff @ alpha)
                z_prev_diff = jnp.stack([z_hist[j+1][b_idx] - z_hist[j][b_idx] for j in range(k_hist)], axis=1)
                z_elem = (1.0 - beta) * (z_hist[-1][b_idx] - (z_prev_diff @ alpha)) + beta * z_b
                z_next_list.append(z_elem)
            z = jnp.stack(z_next_list, axis=0)
            
    final_res = z - f_theta(z, x, w_z, w_x, b)
    return z, final_res


def implicit_jacobian_transpose_solve(
    z_star: jnp.ndarray,
    x: jnp.ndarray,
    w_z: jnp.ndarray,
    w_x: jnp.ndarray,
    b: jnp.ndarray,
    grad_z: jnp.ndarray,
    num_steps: int = 20,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u = z_star @ w_z + x @ w_x + b
    tanh_u = jnp.tanh(u)
    d_act = 1.0 - tanh_u**2
    
    v = jnp.zeros_like(grad_z)
    for _ in range(num_steps):
        v = grad_z + (v * d_act) @ w_z.T
        
    delta = v * d_act
    grad_x = delta @ w_x.T
    grad_wz = z_star.T @ delta
    return v, grad_x, grad_wz


def mon_operator_splitting(
    x: jnp.ndarray,
    w_sym: jnp.ndarray,
    w_skew: jnp.ndarray,
    w_x: jnp.ndarray,
    b: jnp.ndarray,
    alpha_step: float = 0.2,
    num_iters: int = 15,
) -> jnp.ndarray:
    W = w_sym + w_skew
    z = jnp.zeros((x.shape[0], W.shape[0]), dtype=x.dtype)
    for _ in range(num_iters):
        lin = z @ W + x @ w_x + b
        z = jnp.tanh(z - alpha_step * lin)
    return z


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"
    
    z_pic, res_pic = fixed_point_iteration(g["x_input"], g["w_z"], g["w_x"], g["b_z"], num_iters=20)
    assert float(jnp.max(jnp.abs(z_pic - g["z_picard"]))) < 1e-5
    assert float(jnp.max(jnp.abs(res_pic - g["res_picard"]))) < 1e-5
    
    z_and, res_and = anderson_acceleration(g["x_input"], g["w_z"], g["w_x"], g["b_z"], m=2, num_iters=15, beta=1.0)
    assert float(jnp.max(jnp.abs(z_and - g["z_anderson"]))) < 1e-5
    assert float(jnp.max(jnp.abs(res_and - g["res_anderson"]))) < 1e-5
    
    adj_v, grad_x, grad_wz = implicit_jacobian_transpose_solve(
        g["z_picard"], g["x_input"], g["w_z"], g["w_x"], g["b_z"], g["grad_loss_z"], num_steps=20
    )
    assert float(jnp.max(jnp.abs(adj_v - g["adj_v"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_x - g["grad_x"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_wz - g["grad_wz"]))) < 1e-5
    
    z_mon = mon_operator_splitting(g["x_input"], g["w_sym"], g["w_skew"], g["w_mon_x"], g["b_mon"], alpha_step=0.2, num_iters=15)
    assert float(jnp.max(jnp.abs(z_mon - g["z_mon"]))) < 1e-5
    
    print("06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
