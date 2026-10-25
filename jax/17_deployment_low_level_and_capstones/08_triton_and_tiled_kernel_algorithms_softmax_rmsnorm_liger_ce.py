"""Triton and Tiled Kernel Algorithms Softmax RMSNorm Liger Ce — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce. Run ``python jax/17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def fused_rmsnorm(x: jnp.ndarray, gamma: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Fused RMSNorm: x / sqrt(mean(x^2) + eps) * gamma."""
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def fused_swiglu(x: jnp.ndarray, w_gate: jnp.ndarray, w_up: jnp.ndarray) -> jnp.ndarray:
    """Fused SwiGLU: silu(x @ w_gate) * (x @ w_up)."""
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return gate * up


def tiled_linear_cross_entropy(
    h: jnp.ndarray,
    w_vocab: jnp.ndarray,
    labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tiled linear + cross-entropy computing loss and input gradients."""
    b, _ = h.shape
    v = w_vocab.shape[1]
    logits = h @ w_vocab
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, v)
    loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    probs = jax.nn.softmax(logits, axis=-1)
    dh = ((probs - one_hot) @ w_vocab.T) / b
    return loss, dh


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    normed_h = fused_rmsnorm(g["h"], g["gamma"])
    assert float(jnp.max(jnp.abs(normed_h - g["normed_h"]))) < 1e-5

    swiglu_out = fused_swiglu(g["h"], g["w_gate"], g["w_up"])
    assert float(jnp.max(jnp.abs(swiglu_out - g["swiglu_out"]))) < 1e-5

    loss, dh = tiled_linear_cross_entropy(g["h"], g["w_vocab"], g["labels"])
    assert float(jnp.max(jnp.abs(loss - g["loss"]))) < 1e-5
    assert float(jnp.max(jnp.abs(dh - g["dh"]))) < 1e-5

    print("17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
