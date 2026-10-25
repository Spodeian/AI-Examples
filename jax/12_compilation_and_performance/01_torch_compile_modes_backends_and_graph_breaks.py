"""Torch Compile Modes Backends and Graph Breaks — functional JAX implementation.

Topic: 12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks. Run ``python jax/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def fused_swiglu(x: jnp.ndarray, gate: jnp.ndarray) -> jnp.ndarray:
    """Fused SwiGLU activation (Shazeer 2020): y = (x * sigmoid(x)) * gate."""
    return (x * jax.nn.sigmoid(x)) * gate


def fused_add_layernorm(
    x: jnp.ndarray,
    residual: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Fused residual addition and Layer Normalization (Ba et al. 2016)."""
    h = x + residual
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean((h - mean) ** 2, axis=-1, keepdims=True)
    norm = (h - mean) / jnp.sqrt(var + float(eps))
    return norm * gamma + beta


def tensor_cond_select(
    x: jnp.ndarray,
    threshold: float,
    scale_high: float,
    scale_low: float,
) -> jnp.ndarray:
    """Branchless conditional selection (torch.cond / jax.lax.cond equivalent)."""
    return jnp.where(x > float(threshold), x * float(scale_high), x * float(scale_low))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. SwiGLU
    swiglu = fused_swiglu(g["swiglu_x"], g["swiglu_gate"])
    err = float(jnp.max(jnp.abs(swiglu - g["swiglu_ref"])))
    assert err < 1e-5, f"swiglu error {err:.2e}"

    # 3. Add + LayerNorm
    add_ln = fused_add_layernorm(g["ln_x"], g["ln_residual"], g["ln_gamma"], g["ln_beta"], eps=1e-5)
    err = float(jnp.max(jnp.abs(add_ln - g["add_ln_ref"])))
    assert err < 1e-5, f"add_ln error {err:.2e}"

    # 4. Conditional select
    cond = tensor_cond_select(g["cond_x"], threshold=0.0, scale_high=2.0, scale_low=0.5)
    err = float(jnp.max(jnp.abs(cond - g["cond_ref"])))
    assert err < 1e-5, f"cond error {err:.2e}"

    print("12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
