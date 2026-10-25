"""AOT Inductor and Executorch Edge Concepts — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts.
Run ``python jax/17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def aot_edge_forward(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    z = x @ w + b
    mu = jnp.mean(z, axis=-1, keepdims=True)
    var = jnp.var(z, axis=-1, keepdims=True)
    hat_z = ((z - mu) / jnp.sqrt(var + 1e-5)) * gamma + beta
    y = jax.nn.silu(hat_z)
    return y, z, hat_z


def plan_static_memory_arena(
    tensor_bytes: list[int],
    reuse_pairs: list[tuple[int, int]],
) -> tuple[jnp.ndarray, int]:
    offsets = [0] * len(tensor_bytes)
    current_offset = 0
    reused = {}

    for src, dst in reuse_pairs:
        reused[dst] = src

    for i, size in enumerate(tensor_bytes):
        if i in reused:
            offsets[i] = offsets[reused[i]]
        else:
            offsets[i] = current_offset
            current_offset += size

    return jnp.array(offsets, dtype=jnp.int64), current_offset


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Forward
    y, z, hat_z = aot_edge_forward(g["x"], g["w"], g["b"], g["gamma"], g["beta"])
    err = float(jnp.max(jnp.abs(y - g["y_out"])))
    assert err < 1e-5, f"y_out error {err:.2e}"
    err = float(jnp.max(jnp.abs(z - g["linear_z"])))
    assert err < 1e-5, f"linear_z error {err:.2e}"
    err = float(jnp.max(jnp.abs(hat_z - g["norm_z"])))
    assert err < 1e-5, f"norm_z error {err:.2e}"

    # Arena plan
    offsets, _ = plan_static_memory_arena([64, 64, 64, 64], [(1, 3)])
    err = float(jnp.max(jnp.abs(offsets - g["arena_offsets"])))
    assert err < 1e-5, f"arena_offsets error {err:.2e}"

    print("17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
