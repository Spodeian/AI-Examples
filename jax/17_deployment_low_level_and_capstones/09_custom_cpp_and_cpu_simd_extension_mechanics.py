"""Custom Cpp and CPU SIMD Extension Mechanics — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics. Run ``python jax/17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def fused_simd_fma_leaky_relu(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    negative_slope: float = 0.1,
) -> jnp.ndarray:
    """Simulates vector FMA followed by LeakyReLU."""
    z = x * w + b
    return jnp.where(z >= 0.0, z, z * negative_slope)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    y = fused_simd_fma_leaky_relu(g["x"], g["w"], g["b"], negative_slope=0.1)
    assert float(jnp.max(jnp.abs(y - g["y"]))) < 1e-5

    print("17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
