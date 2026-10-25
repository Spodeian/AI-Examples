"""TorchAO Modern API and Affine Quantized Tensor — JAX implementation.

Topic: 14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor
Golden weights: weights/14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor.safetensors
"""

from __future__ import annotations

from pathlib import Path
import jax.numpy as jnp
from safetensors.torch import load_file

TOPIC = "14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(x, weight) + bias


def affine_quantize_blockwise(
    weight: jnp.ndarray,
    group_size: int = 16,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    orig_shape = weight.shape
    flat = weight.flatten()
    n_groups = len(flat) // group_size
    scales = []
    q_ints = []
    dequants = []

    for g in range(n_groups):
        chunk = flat[g * group_size : (g + 1) * group_size]
        amax = float(jnp.max(jnp.abs(chunk)))
        scale = amax / qmax if amax > 0.0 else 1.0
        q = jnp.clip(jnp.round(chunk / scale), qmin, qmax)
        deq = q * scale
        scales.append(scale)
        q_ints.append(q)
        dequants.append(deq)

    scales_array = jnp.array(scales, dtype=jnp.float32)
    q_int_array = jnp.concatenate(q_ints).reshape(orig_shape)
    dequant_array = jnp.concatenate(dequants).reshape(orig_shape)
    return dequant_array, scales_array, q_int_array


def dynamic_w8a8_linear(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s_x = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / 127.0
    s_x = jnp.where(s_x == 0.0, jnp.ones_like(s_x), s_x)
    x_q = jnp.clip(jnp.round(x / s_x), -128.0, 127.0)

    s_w = jnp.max(jnp.abs(weight), axis=0, keepdims=True) / 127.0
    s_w = jnp.where(s_w == 0.0, jnp.ones_like(s_w), s_w)
    w_q = jnp.clip(jnp.round(weight / s_w), -128.0, 127.0)

    y_int = jnp.matmul(x_q, w_q)
    y = y_int * (s_x * s_w) + bias
    return y, s_x, s_w


def int4_weight_only_linear(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    group_size: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    w_deq, scales, q_int = affine_quantize_blockwise(weight, group_size=group_size, qmin=-8.0, qmax=7.0)
    y = jnp.matmul(x, w_deq) + bias
    return y, w_deq, scales


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. Block-wise INT8 AQT
    w_deq, w_scales, w_int = affine_quantize_blockwise(g["w_linear"], group_size=16, qmin=-128.0, qmax=127.0)
    assert float(jnp.max(jnp.abs(w_deq - g["w_block_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(w_scales - g["w_block_scales"]))) < 1e-5
    assert float(jnp.max(jnp.abs(w_int - g["w_block_int"]))) < 1e-5

    out_w = jnp.matmul(g["x_act"], w_deq) + g["b_linear"]
    assert float(jnp.max(jnp.abs(out_w - g["out_w_only"]))) < 1e-5

    # 3. Dynamic W8A8 Linear
    y_w8a8, s_x, s_w = dynamic_w8a8_linear(g["x_act"], g["w_linear"], g["b_linear"])
    assert float(jnp.max(jnp.abs(y_w8a8 - g["w8a8_out"]))) < 1e-5
    assert float(jnp.max(jnp.abs(s_x - g["w8a8_s_x"]))) < 1e-5
    assert float(jnp.max(jnp.abs(s_w - g["w8a8_s_w"]))) < 1e-5

    # 4. INT4 Weight-Only Linear
    out_w4, w4_deq, w4_scales = int4_weight_only_linear(g["x_act"], g["w_linear"], g["b_linear"], group_size=16)
    assert float(jnp.max(jnp.abs(out_w4 - g["out_w4_only"]))) < 1e-5
    assert float(jnp.max(jnp.abs(w4_deq - g["w4_block_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(w4_scales - g["w4_block_scales"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
