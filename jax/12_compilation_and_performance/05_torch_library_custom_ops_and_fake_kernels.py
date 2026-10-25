"""Torch Library Custom Ops and Fake Kernels — functional JAX implementation.

Topic: 12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels. Run ``python jax/12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def custom_soft_shrink(x: jnp.ndarray, lambd: float = 0.5) -> jnp.ndarray:
    return jnp.sign(x) * jnp.clip(jnp.abs(x) - float(lambd), a_min=0.0)


def custom_asymmetric_quantize(
    x: jnp.ndarray,
    eps: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_min = jnp.min(x, axis=-1, keepdims=True)
    x_max = jnp.max(x, axis=-1, keepdims=True)
    scale = jnp.clip((x_max - x_min) / 255.0, a_min=float(eps))
    zero_point = jnp.clip(jnp.round(-x_min / scale), a_min=0.0, a_max=255.0)
    q = jnp.clip(jnp.round(x / scale) + zero_point, a_min=0.0, a_max=255.0)
    x_hat = (q - zero_point) * scale
    return x_hat, scale, zero_point


def custom_depthwise_conv1d(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    b, c, l = x.shape
    c_w, k = weight.shape
    out_len = l - k + 1
    slices = []
    for t in range(out_len):
        sub_x = x[:, :, t : t + k]
        prod = jnp.sum(sub_x * jnp.expand_dims(weight, 0), axis=-1) + jnp.expand_dims(bias, 0)
        slices.append(prod)
    return jnp.stack(slices, axis=-1)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Soft Shrink
    shrink = custom_soft_shrink(g["shrink_x"], lambd=0.5)
    assert float(jnp.max(jnp.abs(shrink - g["shrink_ref"]))) < 1e-5

    # 2. Asymmetric Quantize
    x_hat, scale, zero_point = custom_asymmetric_quantize(g["quant_x"], eps=1e-7)
    assert float(jnp.max(jnp.abs(x_hat - g["quant_x_hat"]))) < 1e-5
    assert float(jnp.max(jnp.abs(scale - g["quant_scale"]))) < 1e-5
    assert float(jnp.max(jnp.abs(zero_point - g["quant_zero"]))) < 1e-5

    # 3. Depthwise Conv1D
    conv = custom_depthwise_conv1d(g["conv_x"], g["conv_w"], g["conv_b"])
    assert float(jnp.max(jnp.abs(conv - g["conv_ref"]))) < 1e-5

    print("12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
