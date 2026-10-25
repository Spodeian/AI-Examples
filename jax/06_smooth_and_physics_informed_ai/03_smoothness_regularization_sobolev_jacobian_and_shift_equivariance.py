"""Smoothness Regularization: Sobolev Norms, Jacobian Penalties, and Shift Equivariance — JAX reference implementation.

Topic: 06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance
Golden weights: weights/06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance.safetensors

``python jax/06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.matmul(x, weight) + bias


def sobolev_gradient(x: jax.Array, w1: jax.Array, b1: jax.Array, w2: jax.Array) -> jax.Array:
    """Computes exact analytical input gradient nabla_x y in JAX."""
    def f_single(x_row: jax.Array) -> jax.Array:
        h = jax.nn.gelu(jnp.matmul(x_row, w1) + b1, approximate=False)
        return jnp.sum(jnp.matmul(h, w2))

    return jax.vmap(jax.grad(f_single))(x)


def jacobian_frobenius_sq(
    x: jax.Array,
    w1: jax.Array,
    b1: jax.Array,
    w2: jax.Array,
) -> jax.Array:
    """Computes squared Frobenius norm of the Jacobian in JAX."""
    def f_vec(x_row: jax.Array) -> jax.Array:
        h = jax.nn.gelu(jnp.matmul(x_row, w1) + b1, approximate=False)
        return jnp.matmul(h, w2)

    vmap_jac = jax.vmap(jax.jacobian(f_vec))(x)  # [B, D_out, D_in]
    return jnp.sum(vmap_jac ** 2, axis=(-2, -1))


def blur_downsample_1d(x: jax.Array, kernel: jax.Array) -> jax.Array:
    """Anti-aliased 1D BlurPool downsampling in JAX."""
    pad_width = ((0, 0), (0, 0), (1, 1))
    x_pad = jnp.pad(x, pad_width, mode="constant")
    b, c, l = x.shape
    out_len = l // 2
    steps = jnp.arange(out_len) * 2
    y0 = x_pad[:, :, steps] * kernel[0]
    y1 = x_pad[:, :, steps + 1] * kernel[1]
    y2 = x_pad[:, :, steps + 2] * kernel[2]
    return y0 + y1 + y2


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation reproduces the golden checkpoint."""
    golden = {k: jnp.asarray(v) for k, v in load_file(str(path)).items()}

    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear_anchor parity failed: max abs error {err:.2e}"

    grad_x = sobolev_gradient(golden["x_feat"], golden["w1"], golden["b1"], golden["w2_scalar"])
    assert float(jnp.max(jnp.abs(grad_x - golden["sobolev_grad"]))) < 1e-5

    j_sq = jacobian_frobenius_sq(golden["x_feat"], golden["w1"], golden["b1"], golden["w2_multi"])
    assert float(jnp.max(jnp.abs(j_sq - golden["j_frob_sq"]))) < 1e-5

    y_blur = blur_downsample_1d(golden["x_signal"], golden["kernel_1d"])
    assert float(jnp.max(jnp.abs(y_blur - golden["y_blur"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
