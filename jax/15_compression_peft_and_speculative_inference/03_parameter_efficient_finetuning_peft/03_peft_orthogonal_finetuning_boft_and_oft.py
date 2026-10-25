"""PEFT Orthogonal Finetuning BOFT and Oft — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft. Run ``python jax/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def cayley_orthogonal(q_skew: jnp.ndarray) -> jnp.ndarray:
    """Cayley transform: R = (I + Q)^(-1) (I - Q) or solve(I + Q, I - Q)."""
    b = q_skew.shape[0]
    eye = jnp.eye(b, dtype=q_skew.dtype)
    return jnp.linalg.solve(eye + q_skew, eye - q_skew)


def block_oft_forward(
    x: jnp.ndarray,
    w0: jnp.ndarray,
    q_blocks: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Block-Diagonal Orthogonal Finetuning forward pass."""
    num_blocks, block_size, _ = q_blocks.shape
    d_in = num_blocks * block_size

    r_blocks = []
    for k in range(num_blocks):
        skew_k = 0.5 * (q_blocks[k] - q_blocks[k].T)
        r_k = cayley_orthogonal(skew_k)
        r_blocks.append(r_k)

    # Construct block diagonal matrix
    r_full = jnp.zeros((d_in, d_in), dtype=q_blocks.dtype)
    for k, r_k in enumerate(r_blocks):
        start = k * block_size
        end = start + block_size
        r_full = r_full.at[start:end, start:end].set(r_k)

    w_oft = r_full @ w0
    h = x @ w_oft
    return h, r_full


def butterfly_boft_forward(
    x: jnp.ndarray,
    w0: jnp.ndarray,
    q_blocks_layer1: jnp.ndarray,
    q_blocks_layer2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Butterfly Orthogonal Finetuning forward pass."""
    _, r1 = block_oft_forward(x, w0, q_blocks_layer1)
    _, r2 = block_oft_forward(x, w0, q_blocks_layer2)
    r_total = r2 @ r1
    w_boft = r_total @ w0
    h = x @ w_boft
    return h, r_total


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Cayley orthogonal
    r_c = cayley_orthogonal(g["q_skew"])
    err = float(jnp.max(jnp.abs(r_c - g["r_cayley"])))
    assert err < 1e-5, f"cayley error {err:.2e}"

    # 2. Block OFT
    oft_out, r_full = block_oft_forward(g["x"], g["w0"], g["q_blocks1"])
    err = float(jnp.max(jnp.abs(oft_out - g["oft_out"])))
    assert err < 1e-5, f"oft_out error {err:.2e}"
    err = float(jnp.max(jnp.abs(r_full - g["r_full"])))
    assert err < 1e-5, f"r_full error {err:.2e}"

    # 3. Butterfly BOFT
    boft_out, r_total = butterfly_boft_forward(g["x"], g["w0"], g["q_blocks1"], g["q_blocks2"])
    err = float(jnp.max(jnp.abs(boft_out - g["boft_out"])))
    assert err < 1e-5, f"boft_out error {err:.2e}"
    err = float(jnp.max(jnp.abs(r_total - g["r_total"])))
    assert err < 1e-5, f"r_total error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
