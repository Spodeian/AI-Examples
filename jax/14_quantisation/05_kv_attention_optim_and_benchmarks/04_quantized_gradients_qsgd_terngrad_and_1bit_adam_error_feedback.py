"""Quantized Gradients QSGD Terngrad and 1bit Adam Error Feedback — functional JAX implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback. Run ``python jax/14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def qsgd_quantize(grad: jnp.ndarray, num_levels: int = 7) -> tuple[jnp.ndarray, jnp.ndarray]:
    """QSGD with L2 norm scaling and uniform coordinate quantization in JAX."""
    norm = jnp.linalg.norm(grad, axis=-1, keepdims=True)
    xi = jnp.abs(grad) / (norm + 1e-8)
    tilde_xi = jnp.round(num_levels * xi) / float(num_levels)
    quant_grad = jnp.sign(grad) * norm * tilde_xi
    return quant_grad, norm


def terngrad_quantize(grad: jnp.ndarray, clip_factor: float = 2.5) -> tuple[jnp.ndarray, jnp.ndarray]:
    """TernGrad 3-level {-scale, 0, +scale} gradient quantization with outlier clipping in JAX."""
    mean = jnp.mean(grad, axis=-1, keepdims=True)
    var = jnp.mean((grad - mean) ** 2, axis=-1, keepdims=True)
    std = jnp.sqrt(var + 1e-8)
    c = clip_factor * std
    g_clip = jnp.clip(grad, -c, c)
    scale = jnp.max(jnp.abs(g_clip), axis=-1, keepdims=True)
    norm_g = jnp.abs(g_clip) / (scale + 1e-8)
    quant_grad = jnp.sign(g_clip) * (norm_g >= 0.5).astype(grad.dtype) * scale
    return quant_grad, scale


def onebit_adam_compress(
    momentum: jnp.ndarray, error_buffer: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """1-bit Adam momentum compression with Error Feedback (EF) in JAX."""
    m_tilde = momentum + error_buffer
    scale = jnp.mean(jnp.abs(m_tilde), axis=-1, keepdims=True)
    signs = jnp.where(m_tilde >= 0, jnp.ones_like(m_tilde), -jnp.ones_like(m_tilde))
    m_hat = scale * signs
    new_error = m_tilde - m_hat
    return m_hat, new_error, scale


def onebit_adam_update_step(
    param: jnp.ndarray,
    m_hat: jnp.ndarray,
    v: jnp.ndarray,
    lr: float = 1e-3,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> jnp.ndarray:
    """Applies parameter update using 1-bit compressed momentum in JAX."""
    denom = jnp.sqrt(v) + eps
    step = m_hat / denom + weight_decay * param
    return param - lr * step


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # QSGD
    q_grad, q_norm = qsgd_quantize(g["grad"], num_levels=7)
    assert float(jnp.max(jnp.abs(q_norm - g["qsgd_norm"]))) < 1e-5
    assert float(jnp.max(jnp.abs(q_grad - g["qsgd_grad"]))) < 1e-5

    # TernGrad
    t_grad, t_scale = terngrad_quantize(g["grad"], clip_factor=2.5)
    assert float(jnp.max(jnp.abs(t_scale - g["terngrad_scale"]))) < 1e-5
    assert float(jnp.max(jnp.abs(t_grad - g["terngrad_grad"]))) < 1e-5

    # 1-bit Adam step 1
    m_hat_1, err_1, s_1 = onebit_adam_compress(g["grad"], g["error_0"])
    assert float(jnp.max(jnp.abs(s_1 - g["adam_scale_1"]))) < 1e-5
    assert float(jnp.max(jnp.abs(m_hat_1 - g["adam_m_hat_1"]))) < 1e-5
    assert float(jnp.max(jnp.abs(err_1 - g["adam_error_1"]))) < 1e-5

    # 1-bit Adam step 2
    m_hat_2, err_2, s_2 = onebit_adam_compress(g["grad_2"], err_1)
    assert float(jnp.max(jnp.abs(s_2 - g["adam_scale_2"]))) < 1e-5
    assert float(jnp.max(jnp.abs(m_hat_2 - g["adam_m_hat_2"]))) < 1e-5
    assert float(jnp.max(jnp.abs(err_2 - g["adam_error_2"]))) < 1e-5

    # Update step
    new_param = onebit_adam_update_step(g["param"], m_hat_1, g["v"], lr=1e-3, eps=1e-8, weight_decay=0.01)
    assert float(jnp.max(jnp.abs(new_param - g["new_param"]))) < 1e-5

    print("14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
