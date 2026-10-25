"""Lipschitz Networks: Björck Orthonormalization, Cayley Transform, and GroupSort — JAX reference implementation.

Topic: 06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort
Golden weights: weights/06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort.safetensors

``python jax/06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.matmul(x, weight) + bias


def bjorck_orthonormalize(w: jax.Array, iters: int = 5) -> jax.Array:
    """Iterative Björck orthonormalization in JAX."""
    norm = jnp.linalg.norm(w, ord="fro") + 1e-6
    w_k = w / norm
    d_out = w.shape[1]
    eye = jnp.eye(d_out)
    for _ in range(iters):
        gram = jnp.matmul(w_k.T, w_k)
        w_k = jnp.matmul(w_k, 1.5 * eye - 0.5 * gram)
    return w_k


def cayley_transform(m: jax.Array) -> jax.Array:
    """Exact orthogonal matrix construction via Cayley transform in JAX."""
    d = m.shape[0]
    eye = jnp.eye(d)
    a = m - m.T
    return jnp.matmul(eye - a, jnp.linalg.inv(eye + a))


def groupsort2(x: jax.Array) -> jax.Array:
    """GroupSort-2 (PairSort) activation function in JAX."""
    b, d = x.shape
    pairs = jnp.reshape(x, (b, d // 2, 2))
    a = pairs[:, :, 0]
    b_val = pairs[:, :, 1]
    y_max = jnp.maximum(a, b_val)
    y_min = jnp.minimum(a, b_val)
    sorted_pairs = jnp.stack([y_max, y_min], axis=-1)
    return jnp.reshape(sorted_pairs, (b, d))


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation reproduces the golden checkpoint."""
    golden = {k: jnp.asarray(v) for k, v in load_file(str(path)).items()}

    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear_anchor parity failed: max abs error {err:.2e}"

    w_ortho = bjorck_orthonormalize(golden["w_raw"], iters=5)
    assert float(jnp.max(jnp.abs(w_ortho - golden["w_ortho"]))) < 1e-5

    q_cayley = cayley_transform(golden["m_raw"])
    assert float(jnp.max(jnp.abs(q_cayley - golden["q_cayley"]))) < 1e-5

    h_bjorck = jnp.matmul(golden["x_feat"], golden["w_ortho"])
    assert float(jnp.max(jnp.abs(h_bjorck - golden["h_bjorck"]))) < 1e-5

    y_groupsort = groupsort2(golden["h_bjorck"])
    assert float(jnp.max(jnp.abs(y_groupsort - golden["y_groupsort"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
