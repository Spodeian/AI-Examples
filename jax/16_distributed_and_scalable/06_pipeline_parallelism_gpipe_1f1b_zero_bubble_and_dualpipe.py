"""Pipeline Parallelism Gpipe 1f1b Zero Bubble and Dualpipe — functional JAX implementation.

Topic: 16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe.
Run ``python jax/16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def compute_pipeline_bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    return (num_stages - 1) / (num_microbatches + num_stages - 1)


def pipeline_forward_step(
    x_mb: jnp.ndarray,
    w0: jnp.ndarray,
    b0: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    m = x_mb.shape[0]
    h_list = []
    y_list = []

    for i in range(m):
        h_i = x_mb[i] @ w0 + b0
        y_i = h_i @ w1 + b1
        h_list.append(h_i)
        y_list.append(y_i)

    h_mb = jnp.stack(h_list, axis=0)
    y_mb = jnp.stack(y_list, axis=0)
    return h_mb, y_mb


def dualpipe_backward_step(
    x_mb: jnp.ndarray,
    h_mb: jnp.ndarray,
    grad_y_mb: jnp.ndarray,
    w0: jnp.ndarray,
    w1: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    m = x_mb.shape[0]
    grad_w0 = jnp.zeros_like(w0)
    grad_w1 = jnp.zeros_like(w1)
    grad_x_list = []

    for i in range(m):
        b_a_1 = grad_y_mb[i] @ w1.T
        b_w_1 = h_mb[i].T @ grad_y_mb[i]
        grad_w1 = grad_w1 + b_w_1

        b_a_0 = b_a_1 @ w0.T
        b_w_0 = x_mb[i].T @ b_a_1
        grad_w0 = grad_w0 + b_w_0

        grad_x_list.append(b_a_0)

    grad_x_mb = jnp.stack(grad_x_list, axis=0)
    return grad_w0, grad_w1, grad_x_mb


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Forward
    h_mb, y_mb = pipeline_forward_step(g["x_mb"], g["w0"], g["b0"], g["w1"], g["b1"])
    err = float(jnp.max(jnp.abs(h_mb - g["stage0_fwd_outs"])))
    assert err < 1e-5, f"stage 0 forward error {err:.2e}"
    err = float(jnp.max(jnp.abs(y_mb - g["stage1_fwd_outs"])))
    assert err < 1e-5, f"stage 1 forward error {err:.2e}"

    # Backward
    grad_w0, grad_w1, grad_x = dualpipe_backward_step(g["x_mb"], g["stage0_fwd_outs"], g["grad_y_mb"], g["w0"], g["w1"])
    err = float(jnp.max(jnp.abs(grad_w0 - g["grad_w0"])))
    assert err < 1e-5, f"grad_w0 error {err:.2e}"
    err = float(jnp.max(jnp.abs(grad_w1 - g["grad_w1"])))
    assert err < 1e-5, f"grad_w1 error {err:.2e}"
    err = float(jnp.max(jnp.abs(grad_x - g["grad_x_outs"])))
    assert err < 1e-5, f"grad_x error {err:.2e}"

    print("16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
