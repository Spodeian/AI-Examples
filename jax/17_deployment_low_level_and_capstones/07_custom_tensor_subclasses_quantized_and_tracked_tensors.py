"""Custom Tensor Subclasses Quantized and Tracked Tensors — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors. Run ``python jax/17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def dequantize_affine(int_weight: jnp.ndarray, scale: jnp.ndarray, zero_point: jnp.ndarray) -> jnp.ndarray:
    """Simulates transparent dequantization: W_fp = (W_int - zp) * scale."""
    return (int_weight.astype(jnp.float32) - zero_point) * scale


def forward_dual_affine(
    x_primal: jnp.ndarray,
    x_tangent: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward affine step computing both primal output and JVP tangent."""
    y_primal = x_primal @ w + b
    y_tangent = x_tangent @ w
    return y_primal, y_tangent


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    w = dequantize_affine(g["int_w"], g["scale"], g["zero_point"])
    assert float(jnp.max(jnp.abs(w - g["w"]))) < 1e-5

    y_primal, y_tangent = forward_dual_affine(g["x_primal"], g["x_tangent"], w, g["b"])
    assert float(jnp.max(jnp.abs(y_primal - g["y_primal"]))) < 1e-5
    assert float(jnp.max(jnp.abs(y_tangent - g["y_tangent"]))) < 1e-5

    print("17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
