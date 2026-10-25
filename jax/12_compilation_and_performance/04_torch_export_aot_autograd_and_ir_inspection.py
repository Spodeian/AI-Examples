"""Torch Export AOT Autograd and IR Inspection — functional JAX implementation.

Topic: 12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection. Run ``python jax/12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.scipy.special
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def aten_core_softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Decomposed Core ATen operator representation of Softmax."""
    m = jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x - m)
    s = jnp.sum(e, axis=axis, keepdims=True)
    return e / s


def aten_core_layernorm(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Decomposed Core ATen operator representation of LayerNorm."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    rsqrt_var = 1.0 / jnp.sqrt(var + float(eps))
    norm = (x - mean) * rsqrt_var
    return norm * gamma + beta


def aot_forward_backward_affine_gelu(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    grad_out: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """AOTAutograd forward and backward VJP adjoint graph."""
    z = x @ w + b
    sqrt_2 = 1.4142135623730951
    sqrt_2_pi = 2.5066282746310002

    a = 0.5 * z * (1.0 + jax.scipy.special.erf(z / sqrt_2))
    d_gelu_dz = 0.5 * (1.0 + jax.scipy.special.erf(z / sqrt_2)) + (z / sqrt_2_pi) * jnp.exp(-0.5 * z * z)
    gz = grad_out * d_gelu_dz
    grad_w = x.T @ gz
    grad_b = jnp.sum(gz, axis=0)
    grad_x = gz @ w.T
    return a, grad_w, grad_b, grad_x


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Softmax
    sm = aten_core_softmax(g["sm_x"], axis=-1)
    assert float(jnp.max(jnp.abs(sm - g["sm_ref"]))) < 1e-5

    # 2. LayerNorm
    ln = aten_core_layernorm(g["ln_x"], g["ln_gamma"], g["ln_beta"], eps=1e-5)
    assert float(jnp.max(jnp.abs(ln - g["ln_ref"]))) < 1e-5

    # 3. AOT Forward-Backward VJP
    fwd_a, grad_w, grad_b, grad_x = aot_forward_backward_affine_gelu(
        g["mlp_x"], g["mlp_w"], g["mlp_b"], g["mlp_grad_out"]
    )
    assert float(jnp.max(jnp.abs(fwd_a - g["fwd_a_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_w - g["grad_w_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_b - g["grad_b_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_x - g["grad_x_ref"]))) < 1e-5

    print("12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
