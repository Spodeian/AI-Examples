"""Flow Matching Rectified Flow and Consistency Models — functional JAX implementation.

Topic: 05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models. Run ``python jax/05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def conditional_flow_matching(x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes straight-path interpolation x_t and target velocity u_t(x_1 | x_0) = x_1 - x_0."""
    x_t = (1.0 - t) * x_0 + t * x_1
    v_target = x_1 - x_0
    return x_t, v_target


def velocity_network(
    x: jnp.ndarray,
    t: jnp.ndarray,
    w_x: jnp.ndarray,
    b_x: jnp.ndarray,
    w_t: jnp.ndarray,
    b_t: jnp.ndarray,
    w_out: jnp.ndarray,
    b_out: jnp.ndarray,
) -> jnp.ndarray:
    """Predicts velocity v_theta(x, t) given position x and time t."""
    t_emb = jax.nn.silu(t @ w_t + b_t)
    h = jax.nn.silu(x @ w_x + b_x + t_emb)
    v_pred = h @ w_out + b_out
    return v_pred


def rectified_flow_step(x: jnp.ndarray, v_pred: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Simulates one Euler ODE step along the rectified vector field."""
    return x + dt * v_pred


def consistency_coefficients(
    t: jnp.ndarray,
    epsilon: float = 0.002,
    sigma_data: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes skip and output scaling coefficients for boundary condition enforcement."""
    dt = t - epsilon
    c_skip = (sigma_data ** 2) / (dt ** 2 + sigma_data ** 2)
    c_out = (sigma_data * dt) / jnp.sqrt(sigma_data ** 2 + t ** 2)
    return c_skip, c_out


def consistency_forward(
    x: jnp.ndarray,
    t: jnp.ndarray,
    f_x: jnp.ndarray,
    epsilon: float = 0.002,
    sigma_data: float = 0.5,
) -> jnp.ndarray:
    """Evaluates consistency function f_theta(x, t) = c_skip(t)*x + c_out(t)*F_theta(x, t)."""
    c_skip, c_out = consistency_coefficients(t, epsilon, sigma_data)
    return c_skip * x + c_out * f_x


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    x_t, v_target = conditional_flow_matching(g["x_0"], g["x_1"], g["t"])
    assert float(jnp.max(jnp.abs(x_t - g["x_t"]))) < 1e-5, "x_t parity mismatch"
    assert float(jnp.max(jnp.abs(v_target - g["v_target"]))) < 1e-5, "v_target parity mismatch"

    v_pred = velocity_network(x_t, g["t"], g["w_x"], g["b_x"], g["w_t"], g["b_t"], g["w_out"], g["b_out"])
    assert float(jnp.max(jnp.abs(v_pred - g["v_pred"]))) < 1e-5, "v_pred parity mismatch"

    cfm_loss = jnp.mean((v_pred - v_target) ** 2).reshape((1,))
    assert float(jnp.max(jnp.abs(cfm_loss - g["cfm_loss"]))) < 1e-5, "cfm_loss parity mismatch"

    x_next = rectified_flow_step(x_t, v_pred, dt=0.1)
    assert float(jnp.max(jnp.abs(x_next - g["x_next"]))) < 1e-5, "x_next parity mismatch"

    c_skip, c_out = consistency_coefficients(g["t"], epsilon=0.002, sigma_data=0.5)
    assert float(jnp.max(jnp.abs(c_skip - g["c_skip"]))) < 1e-5, "c_skip parity mismatch"
    assert float(jnp.max(jnp.abs(c_out - g["c_out"]))) < 1e-5, "c_out parity mismatch"

    cons_out = consistency_forward(x_t, g["t"], v_pred, epsilon=0.002, sigma_data=0.5)
    assert float(jnp.max(jnp.abs(cons_out - g["cons_out"]))) < 1e-5, "cons_out parity mismatch"

    print("05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
