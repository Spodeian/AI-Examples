"""Input Convex Neural Networks (ICNN), Brenier Optimal Transport, and Certified Monotonic Networks — JAX reference implementation.

Topic: 06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic
Golden weights: weights/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.safetensors

``python jax/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.matmul(x, weight) + bias


def icnn_potential(
    x: jax.Array,
    w_x0: jax.Array,
    b0: jax.Array,
    w_z1: jax.Array,
    w_x1: jax.Array,
    b1: jax.Array,
) -> jax.Array:
    """Convex potential evaluation using Input Convex Neural Network in JAX."""
    w_z1_pos = jax.nn.softplus(w_z1)
    z0 = jax.nn.relu(jnp.matmul(x, w_x0) + b0)
    return jnp.matmul(z0, w_z1_pos) + jnp.matmul(x, w_x1) + b1


def brenier_ot_gradient_map(
    x: jax.Array,
    w_x0: jax.Array,
    b0: jax.Array,
    w_z1: jax.Array,
    w_x1: jax.Array,
) -> jax.Array:
    """Brenier optimal transport gradient map T(x) = x + nabla_x psi(x) in JAX."""
    w_z1_pos = jax.nn.softplus(w_z1)
    pre_act = jnp.matmul(x, w_x0) + b0
    mask = (pre_act > 0.0).astype(x.dtype)
    effective_z = mask * w_z1_pos.T
    grad_psi = jnp.matmul(effective_z, w_x0.T) + w_x1.T
    return x + grad_psi


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation reproduces the golden checkpoint."""
    golden = {k: jnp.asarray(v) for k, v in load_file(str(path)).items()}

    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear_anchor parity failed: max abs error {err:.2e}"

    psi = icnn_potential(
        golden["x_input"],
        golden["w_x0"],
        golden["b0"],
        golden["w_z1"],
        golden["w_x1"],
        golden["b1"],
    )
    assert float(jnp.max(jnp.abs(psi - golden["icnn_psi"]))) < 1e-5

    t_map = brenier_ot_gradient_map(
        golden["x_input"],
        golden["w_x0"],
        golden["b0"],
        golden["w_z1"],
        golden["w_x1"],
    )
    assert float(jnp.max(jnp.abs(t_map - golden["brenier_t_map"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
