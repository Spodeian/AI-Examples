"""PT2E Export Quantization Workflow — JAX implementation.

Topic: 14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow
Golden weights: weights/14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow.safetensors
"""

from __future__ import annotations

from pathlib import Path
import jax.numpy as jnp
from jax import lax
from safetensors.torch import load_file

TOPIC = "14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(x, weight) + bias


def conv2d_bn_fold(
    w_conv: jnp.ndarray,
    b_conv: jnp.ndarray,
    bn_mean: jnp.ndarray,
    bn_var: jnp.ndarray,
    bn_gamma: jnp.ndarray,
    bn_beta: jnp.ndarray,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = bn_gamma / jnp.sqrt(bn_var + eps)
    w_folded = w_conv * scale.reshape(-1, 1, 1, 1)
    b_folded = (b_conv - bn_mean) * scale + bn_beta
    return w_folded, b_folded


def linear_bn_fold(
    w_linear: jnp.ndarray,
    b_linear: jnp.ndarray,
    bn_mean: jnp.ndarray,
    bn_var: jnp.ndarray,
    bn_gamma: jnp.ndarray,
    bn_beta: jnp.ndarray,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = bn_gamma / jnp.sqrt(bn_var + eps)
    w_folded = w_linear * scale.reshape(1, -1)
    b_folded = (b_linear - bn_mean) * scale + bn_beta
    return w_folded, b_folded


def quantize_symmetric_int8(
    x: jnp.ndarray,
    scale: float,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> jnp.ndarray:
    if scale == 0.0:
        return jnp.zeros_like(x)
    return jnp.clip(jnp.round(x / scale), qmin, qmax) * scale


def pt2e_qdq_conv2d(
    x: jnp.ndarray,
    w_folded: jnp.ndarray,
    b_folded: jnp.ndarray,
    s_x: float,
    s_w: float,
) -> jnp.ndarray:
    x_q = quantize_symmetric_int8(x, s_x)
    w_q = quantize_symmetric_int8(w_folded, s_w)
    conv_out = lax.conv_general_dilated(
        x_q, w_q, (1, 1), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW")
    )
    return conv_out + b_folded.reshape(1, -1, 1, 1)


def pt2e_qdq_linear(
    x: jnp.ndarray,
    w_folded: jnp.ndarray,
    b_folded: jnp.ndarray,
    s_x: float,
    s_w: float,
) -> jnp.ndarray:
    x_q = quantize_symmetric_int8(x, s_x)
    w_q = quantize_symmetric_int8(w_folded, s_w)
    return jnp.matmul(x_q, w_q) + b_folded


def unfused_conv_bn_quantize(
    x: jnp.ndarray,
    w_conv: jnp.ndarray,
    b_conv: jnp.ndarray,
    bn_mean: jnp.ndarray,
    bn_var: jnp.ndarray,
    bn_gamma: jnp.ndarray,
    bn_beta: jnp.ndarray,
    s_w: float,
    eps: float = 1e-5,
) -> jnp.ndarray:
    w_q = quantize_symmetric_int8(w_conv, s_w)
    conv_out = lax.conv_general_dilated(
        x, w_q, (1, 1), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW")
    ) + b_conv.reshape(1, -1, 1, 1)
    scale = bn_gamma / jnp.sqrt(bn_var + eps)
    return (conv_out - bn_mean.reshape(1, -1, 1, 1)) * scale.reshape(1, -1, 1, 1) + bn_beta.reshape(1, -1, 1, 1)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. Conv-BN fold
    wf, bf = conv2d_bn_fold(g["w_conv"], g["b_conv"], g["bn_mean"], g["bn_var"], g["bn_gamma"], g["bn_beta"])
    assert float(jnp.max(jnp.abs(wf - g["w_folded"]))) < 1e-5
    assert float(jnp.max(jnp.abs(bf - g["b_folded"]))) < 1e-5

    # 3. Linear-BN fold
    wlf, blf = linear_bn_fold(g["w_lin"], g["b_lin"], g["lin_mean"], g["lin_var"], g["lin_gamma"], g["lin_beta"])
    assert float(jnp.max(jnp.abs(wlf - g["w_lin_folded"]))) < 1e-5
    assert float(jnp.max(jnp.abs(blf - g["b_lin_folded"]))) < 1e-5

    # 4. QDQ Conv2d
    s_x_conv = float(jnp.max(jnp.abs(g["x_conv"]))) / 127.0
    s_w_conv = float(jnp.max(jnp.abs(g["w_folded"]))) / 127.0
    out_qdq_conv = pt2e_qdq_conv2d(g["x_conv"], g["w_folded"], g["b_folded"], s_x_conv, s_w_conv)
    assert float(jnp.max(jnp.abs(out_qdq_conv - g["out_qdq_conv"]))) < 1e-5

    # 5. QDQ Linear
    s_x_lin = float(jnp.max(jnp.abs(g["x_lin"]))) / 127.0
    s_w_lin = float(jnp.max(jnp.abs(g["w_lin_folded"]))) / 127.0
    out_qdq_lin = pt2e_qdq_linear(g["x_lin"], g["w_lin_folded"], g["b_lin_folded"], s_x_lin, s_w_lin)
    assert float(jnp.max(jnp.abs(out_qdq_lin - g["out_qdq_lin"]))) < 1e-5

    # 6. Unfused Conv-BN
    out_unf = unfused_conv_bn_quantize(g["x_conv"], g["w_conv"], g["b_conv"], g["bn_mean"], g["bn_var"], g["bn_gamma"], g["bn_beta"], s_w_conv)
    assert float(jnp.max(jnp.abs(out_unf - g["out_unfused"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
