"""LLM INT8 Mixed Precision Outlier Decomposition — functional JAX implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition. Run ``python jax/14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def find_outlier_channels(x: jnp.ndarray, threshold: float = 6.0) -> jnp.ndarray:
    channel_max = jnp.max(jnp.abs(x), axis=0)
    return jnp.where(channel_max >= threshold)[0]


def quantize_row_int8(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / 127.0
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    q = jnp.clip(jnp.round(x / scale), -128.0, 127.0)
    return q, scale


def quantize_col_int8(w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = jnp.max(jnp.abs(w), axis=0, keepdims=True) / 127.0
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    q = jnp.clip(jnp.round(w / scale), -128.0, 127.0)
    return q, scale


def llm_int8_linear(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    threshold: float = 6.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    channel_max = jnp.max(jnp.abs(x), axis=0)
    outlier_mask = channel_max >= threshold
    outlier_cols = jnp.where(outlier_mask)[0]
    normal_cols = jnp.where(~outlier_mask)[0]

    if len(outlier_cols) > 0:
        x_outlier = x[:, outlier_cols]
        w_outlier = weight[outlier_cols, :]
        y_fp = x_outlier @ w_outlier
    else:
        y_fp = jnp.zeros((x.shape[0], weight.shape[1]), dtype=x.dtype)

    if len(normal_cols) > 0:
        x_normal = x[:, normal_cols]
        w_normal = weight[normal_cols, :]
        q_x, s_x = quantize_row_int8(x_normal)
        q_w, s_w = quantize_col_int8(w_normal)
        deq_x = q_x * s_x
        deq_w = q_w * s_w
        y_int8 = deq_x @ deq_w
    else:
        y_int8 = jnp.zeros((x.shape[0], weight.shape[1]), dtype=x.dtype)

    y_total = y_fp + y_int8
    if bias is not None:
        y_total = y_total + bias

    return y_total, y_fp, y_int8, outlier_cols.astype(jnp.float32)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Pure INT8 baseline
    q_x_pure, s_x_pure = quantize_row_int8(g["x_calib"])
    q_w_pure, s_w_pure = quantize_col_int8(g["w_orig"])
    out_pure_int8 = (q_x_pure * s_x_pure) @ (q_w_pure * s_w_pure) + g["b_orig"]
    assert float(jnp.max(jnp.abs(out_pure_int8 - g["out_pure_int8"]))) < 1e-5

    # 3. LLM.int8 decomposition
    out_total, out_fp, out_int8, outlier_indices = llm_int8_linear(
        g["x_calib"], g["w_orig"], g["b_orig"], threshold=6.0
    )
    assert float(jnp.max(jnp.abs(out_total - g["out_total"]))) < 1e-5
    assert float(jnp.max(jnp.abs(out_fp - g["out_fp"]))) < 1e-5
    assert float(jnp.max(jnp.abs(out_int8 - g["out_int8"]))) < 1e-5
    assert float(jnp.max(jnp.abs(outlier_indices - g["outlier_indices"]))) < 1e-5

    print("14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
