"""Profiling Roofline Analysis CUDA Graphs and Fused Optimizers — functional JAX implementation.

Topic: 12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers. Run ``python jax/12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def roofline_metrics(
    flops: float,
    bytes_transferred: float,
    peak_tflops: float = 312.0,
    peak_bandwidth_tbs: float = 2.039,
) -> tuple[float, float, float]:
    intensity = float(flops) / max(float(bytes_transferred), 1.0)
    ridge_point = float(peak_tflops) / float(peak_bandwidth_tbs)
    attainable_tflops = min(float(peak_tflops), intensity * float(peak_bandwidth_tbs))
    return intensity, ridge_point, attainable_tflops


def fused_adamw_step(
    w: jnp.ndarray,
    g: jnp.ndarray,
    m: jnp.ndarray,
    v: jnp.ndarray,
    step: int = 1,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    b1 = float(beta1)
    b2 = float(beta2)
    t = float(step)

    m_t = b1 * m + (1.0 - b1) * g
    v_t = b2 * v + (1.0 - b2) * (g * g)

    bias_corr1 = 1.0 - (b1 ** t)
    bias_corr2 = 1.0 - (b2 ** t)

    m_hat = m_t / bias_corr1
    v_hat = v_t / bias_corr2

    w_decayed = w * (1.0 - float(lr) * float(weight_decay))
    w_t = w_decayed - float(lr) * (m_hat / (jnp.sqrt(v_hat) + float(eps)))
    return w_t, m_t, v_t


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Verify Fused AdamW
    w_t, m_t, v_t = fused_adamw_step(
        g["adam_w"], g["adam_g"], g["adam_m"], g["adam_v"],
        step=5, lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=1e-2, eps=1e-8
    )
    assert float(jnp.max(jnp.abs(w_t - g["adam_w_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(m_t - g["adam_m_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(v_t - g["adam_v_ref"]))) < 1e-5

    print("12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
