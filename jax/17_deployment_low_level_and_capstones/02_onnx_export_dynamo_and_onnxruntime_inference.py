"""ONNX Export Dynamo and Onnxruntime Inference — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference.
Run ``python jax/17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def onnx_dynamo_forward(
    x: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
) -> jnp.ndarray:
    """Dynamo-compatible feedforward block in JAX."""
    h = jax.nn.gelu(x @ w1 + b1, approximate=False)
    return h @ w2 + b2


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Dynamic shape 1: [1, 4, 4]
    y1 = onnx_dynamo_forward(g["x1"], g["w1"], g["b1"], g["w2"], g["b2"])
    err = float(jnp.max(jnp.abs(y1 - g["y1"])))
    assert err < 1e-5, f"y1 dynamic shape error {err:.2e}"

    # Dynamic shape 2: [2, 2, 4]
    y2 = onnx_dynamo_forward(g["x2"], g["w1"], g["b1"], g["w2"], g["b2"])
    err = float(jnp.max(jnp.abs(y2 - g["y2"])))
    assert err < 1e-5, f"y2 dynamic shape error {err:.2e}"

    print("17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
