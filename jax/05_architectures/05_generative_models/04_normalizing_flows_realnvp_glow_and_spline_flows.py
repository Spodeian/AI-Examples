"""Normalizing Flows Realnvp Glow and Spline Flows — functional JAX implementation.

Topic: 05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows. Run ``python jax/05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def realnvp_affine_coupling_forward(
    x_a: jnp.ndarray,
    x_b: jnp.ndarray,
    s_w: jnp.ndarray,
    s_b: jnp.ndarray,
    t_w: jnp.ndarray,
    t_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = jnp.tanh(x_a @ s_w + s_b)
    t = x_a @ t_w + t_b
    y_a = x_a
    y_b = x_b * jnp.exp(s) + t
    log_det = jnp.sum(s, axis=-1)
    return y_a, y_b, log_det


def realnvp_affine_coupling_inverse(
    y_a: jnp.ndarray,
    y_b: jnp.ndarray,
    s_w: jnp.ndarray,
    s_b: jnp.ndarray,
    t_w: jnp.ndarray,
    t_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = jnp.tanh(y_a @ s_w + s_b)
    t = y_a @ t_w + t_b
    x_a = y_a
    x_b = (y_b - t) * jnp.exp(-s)
    log_det = -jnp.sum(s, axis=-1)
    return x_a, x_b, log_det


def glow_invertible_1x1_conv(
    x: jnp.ndarray,
    w: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    y = x @ w
    sign, logabsdet = jnp.linalg.slogdet(w)
    log_det = jnp.broadcast_to(logabsdet, (x.shape[0],))
    w_inv = jnp.linalg.inv(w)
    x_recon = y @ w_inv
    return y, log_det, x_recon


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. RealNVP
    y_a, y_b, log_det = realnvp_affine_coupling_forward(g["x_a"], g["x_b"], g["s_w"], g["s_b"], g["t_w"], g["t_b"])
    err_ya = float(jnp.max(jnp.abs(y_a - g["y_a"])))
    err_yb = float(jnp.max(jnp.abs(y_b - g["y_b"])))
    err_ld = float(jnp.max(jnp.abs(log_det - g["log_det"])))
    assert err_ya < 1e-5, f"y_a error {err_ya:.2e}"
    assert err_yb < 1e-5, f"y_b error {err_yb:.2e}"
    assert err_ld < 1e-5, f"log_det error {err_ld:.2e}"

    x_a_rec, x_b_rec, _ = realnvp_affine_coupling_inverse(y_a, y_b, g["s_w"], g["s_b"], g["t_w"], g["t_b"])
    err_xar = float(jnp.max(jnp.abs(x_a_rec - g["x_a_rec"])))
    err_xbr = float(jnp.max(jnp.abs(x_b_rec - g["x_b_rec"])))
    assert err_xar < 1e-5, f"x_a_rec error {err_xar:.2e}"
    assert err_xbr < 1e-5, f"x_b_rec error {err_xbr:.2e}"

    # 3. Glow
    y_glow, log_det_glow, x_glow_rec = glow_invertible_1x1_conv(g["x_glow"], g["w_glow"])
    err_yg = float(jnp.max(jnp.abs(y_glow - g["y_glow"])))
    err_ldg = float(jnp.max(jnp.abs(log_det_glow - g["log_det_glow"])))
    err_xgr = float(jnp.max(jnp.abs(x_glow_rec - g["x_glow_rec"])))
    assert err_yg < 1e-5, f"y_glow error {err_yg:.2e}"
    assert err_ldg < 1e-5, f"log_det_glow error {err_ldg:.2e}"
    assert err_xgr < 1e-5, f"x_glow_rec error {err_xgr:.2e}"

    print("05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
