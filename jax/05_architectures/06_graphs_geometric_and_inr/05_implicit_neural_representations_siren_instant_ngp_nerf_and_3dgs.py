"""Implicit Neural Representations: SIREN, Instant-NGP, NeRF, and 3DGS — functional JAX implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs. Run ``python jax/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def siren_layer(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray, omega_0: float = 30.0) -> jnp.ndarray:
    return jnp.sin(omega_0 * (x @ weight + bias))


def volume_rendering_quadrature(
    densities: jnp.ndarray,
    colors: jnp.ndarray,
    deltas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    b, s = densities.shape
    tau = jax.nn.relu(densities) * deltas
    alpha = 1.0 - jnp.exp(-tau)

    cum_tau = jnp.cumsum(tau, axis=-1)
    zeros = jnp.zeros((b, 1))
    prev_tau = jnp.concatenate([zeros, cum_tau[:, :-1]], axis=-1)
    transmittance = jnp.exp(-prev_tau)

    weights = jnp.expand_dims(transmittance * alpha, axis=-1)
    rendered_color = jnp.sum(weights * colors, axis=1)
    accum_opacity = jnp.sum(weights, axis=1)

    return rendered_color, accum_opacity


def gaussian_splat_2d_eval(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    cov_inv: jnp.ndarray,
) -> jnp.ndarray:
    delta = x - mu
    proj = delta @ cov_inv
    mahalanobis = jnp.sum(proj * delta, axis=-1, keepdims=True)
    return jnp.exp(-0.5 * mahalanobis)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    h_siren = siren_layer(g["x_siren"], g["w_siren"], g["b_siren"], omega_0=30.0)
    assert float(jnp.max(jnp.abs(h_siren - g["h_siren"]))) < 1e-5, "h_siren parity mismatch"

    rendered_color, accum_opacity = volume_rendering_quadrature(g["densities"], g["colors"], g["deltas"])
    assert float(jnp.max(jnp.abs(rendered_color - g["rendered_color"]))) < 1e-5, "rendered_color parity mismatch"
    assert float(jnp.max(jnp.abs(accum_opacity - g["accum_opacity"]))) < 1e-5, "accum_opacity parity mismatch"

    splat_alpha = gaussian_splat_2d_eval(g["x_pixels"], g["mu"], g["cov_inv"])
    assert float(jnp.max(jnp.abs(splat_alpha - g["splat_alpha"]))) < 1e-5, "splat_alpha parity mismatch"

    print("05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
