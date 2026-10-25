"""Torch Dispatch Modes Flop Counter and Memory Tracker — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker. Run ``python jax/17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def mlp_forward(
    x: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure functional forward pass of 2-layer residual MLP."""
    z1 = jax.nn.silu(x @ w1 + b1)
    z2 = z1 @ w2 + b2
    y = z2 + x
    return y, z1


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    y, z1 = mlp_forward(g["x"], g["w1"], g["b1"], g["w2"], g["b2"])
    err_y = float(jnp.max(jnp.abs(y - g["y"])))
    assert err_y < 1e-5, f"y error {err_y:.2e}"

    err_z1 = float(jnp.max(jnp.abs(z1 - g["z1"])))
    assert err_z1 < 1e-5, f"z1 error {err_z1:.2e}"

    print("17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
