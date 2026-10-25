"""Numerical Integration PCEC RK45 Neural ODE LTC and SDE — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde.
Run ``python jax/06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def eval_vector_field(x: jnp.ndarray, w1: jnp.ndarray, b1: jnp.ndarray, w2: jnp.ndarray, b2: jnp.ndarray) -> jnp.ndarray:
    z = x @ w1 + b1
    a = jnp.tanh(z)
    return a @ w2 + b2


def eval_jacobian(x: jnp.ndarray, w1: jnp.ndarray, b1: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
    z = x @ w1 + b1
    a = jnp.tanh(z)
    d_a = 1.0 - a**2
    return jnp.einsum("ik, bk, kj -> bij", w1, d_a, w2)


def compute_integrators_all(g: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    x0 = g["x0"]
    w1 = g["w1"]
    b1 = g["b1"]
    w2 = g["w2"]
    b2 = g["b2"]

    f = lambda x: eval_vector_field(x, w1, b1, w2, b2)
    h = 0.1
    f0 = f(x0)

    # 1. Euler
    x_euler = x0 + h * f0

    # 2. Heun
    k1 = f0
    k2 = f(x0 + h * k1)
    x_heun = x0 + 0.5 * h * (k1 + k2)

    # 3. PCEC
    x_p = x0 + h * f0
    f_p = f(x_p)
    x_c1 = x0 + 0.5 * h * (f0 + f_p)
    f_c = f(x_c1)
    x_pcec = x0 + 0.5 * h * (f0 + f_c)

    # 4. RK4
    rk_k1 = f0
    rk_k2 = f(x0 + 0.5 * h * rk_k1)
    rk_k3 = f(x0 + 0.5 * h * rk_k2)
    rk_k4 = f(x0 + h * rk_k3)
    x_rk4 = x0 + (h / 6.0) * (rk_k1 + 2.0 * rk_k2 + 2.0 * rk_k3 + rk_k4)

    # 5. Adjoint
    adj_T = g["adj_T"]
    j0 = eval_jacobian(x0, w1, b1, w2)
    adj_0 = adj_T + h * jnp.einsum("bj, bij -> bi", adj_T, j0)

    # 6. CfC
    w_cfc1 = g["w_cfc1"]
    w_cfc2 = g["w_cfc2"]
    gate = jax.nn.sigmoid(-jnp.abs(f0) * h)
    x_cfc = gate * (x0 @ w_cfc1) + (1.0 - gate) * (x0 @ w_cfc2)

    # 7. SDE
    sigma = 0.2
    dW = g["dW"]
    x_em = x0 + h * f0 + sigma * x0 * dW
    x_milstein = x_em + 0.5 * (sigma**2) * x0 * (dW**2 - h)

    return {
        "f0": f0,
        "x_euler": x_euler,
        "x_heun": x_heun,
        "x_pcec": x_pcec,
        "x_rk4": x_rk4,
        "j0": j0,
        "adj_0": adj_0,
        "x_cfc": x_cfc,
        "x_em": x_em,
        "x_milstein": x_milstein,
    }


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    res = compute_integrators_all(g)
    for name, computed in res.items():
        diff = float(jnp.max(jnp.abs(computed - g[name])))
        rel = diff / (float(jnp.max(jnp.abs(g[name]))) + 1e-8)
        assert diff < 1e-5 or rel < 1e-5, f"{name}: max abs diff {diff:.2e}, rel diff {rel:.2e}"

    print("06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
