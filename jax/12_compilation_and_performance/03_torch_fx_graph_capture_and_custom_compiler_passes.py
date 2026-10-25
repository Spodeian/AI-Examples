"""Torch Fx Graph Capture and Custom Compiler Passes — functional JAX implementation.

Topic: 12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes. Run ``python jax/12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def algebraic_linear_bn_folding(
    w: jnp.ndarray,
    b: jnp.ndarray,
    running_mean: jnp.ndarray,
    running_var: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Algebraic folding of BatchNorm into Linear parameters:
    W_fold = W * (gamma / sqrt(var + eps))
    b_fold = (b - mean) * (gamma / sqrt(var + eps)) + beta
    """
    scale = gamma / jnp.sqrt(running_var + float(eps))
    w_fold = w * jnp.expand_dims(scale, 0)
    b_fold = (b - running_mean) * scale + beta
    return w_fold, b_fold


def sequential_linear_bn(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    running_mean: jnp.ndarray,
    running_var: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Unfolded sequential Linear -> BatchNorm forward pass."""
    z = x @ w + b
    return ((z - running_mean) / jnp.sqrt(running_var + float(eps))) * gamma + beta


def parallel_gemm_fusion(
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
    w3: jnp.ndarray,
    b3: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Horizontal GEMM fusion pass (e.g. Q, K, V projection concatenation)."""
    w_fused = jnp.concatenate([w1, w2, w3], axis=-1)
    b_fused = jnp.concatenate([b1, b2, b3], axis=-1)
    return w_fused, b_fused


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Verify Linear-BN folding
    w_fold, b_fold = algebraic_linear_bn_folding(
        g["lin_w"], g["lin_b"], g["bn_mean"], g["bn_var"], g["bn_gamma"], g["bn_beta"], eps=1e-5
    )
    assert float(jnp.max(jnp.abs(w_fold - g["w_fold_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(b_fold - g["b_fold_ref"]))) < 1e-5

    folded_out = g["lin_x"] @ w_fold + b_fold
    assert float(jnp.max(jnp.abs(folded_out - g["folded_out_ref"]))) < 1e-5

    unfolded_out = sequential_linear_bn(
        g["lin_x"], g["lin_w"], g["lin_b"], g["bn_mean"], g["bn_var"], g["bn_gamma"], g["bn_beta"], eps=1e-5
    )
    assert float(jnp.max(jnp.abs(unfolded_out - folded_out))) < 1e-5

    # 2. Verify Parallel GEMM fusion
    w_fused, b_fused = parallel_gemm_fusion(g["w1"], g["b1"], g["w2"], g["b2"], g["w3"], g["b3"])
    assert float(jnp.max(jnp.abs(w_fused - g["w_fused_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(b_fused - g["b_fused_ref"]))) < 1e-5

    fused_out = g["lin_x"] @ w_fused + b_fused
    assert float(jnp.max(jnp.abs(fused_out - g["fused_out_ref"]))) < 1e-5

    print("12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
