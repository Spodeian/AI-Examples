"""JAX / NumPy implementation for Category 04 Topic 03:
03_large_batch_and_layerwise_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def step_lars_np(
    param: np.ndarray,
    grad: np.ndarray,
    v_buf: np.ndarray,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    trust_coef: float = 0.001,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    w_norm = float(np.linalg.norm(param))
    g_norm = float(np.linalg.norm(grad))

    if w_norm > 0 and g_norm > 0:
        trust_ratio = trust_coef * w_norm / (g_norm + weight_decay * w_norm + eps)
    else:
        trust_ratio = 1.0

    local_lr = lr * trust_ratio
    scaled_g = local_lr * (grad + weight_decay * param)
    new_v = momentum * v_buf + scaled_g
    new_param = param - new_v
    return new_param, new_v, trust_ratio


def step_lamb_np(
    param: np.ndarray,
    grad: np.ndarray,
    m_buf: np.ndarray,
    v_buf: np.ndarray,
    step: int = 2,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-6,
    weight_decay: float = 0.01,
    trust_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    new_m = beta1 * m_buf + (1.0 - beta1) * grad
    new_v = beta2 * v_buf + (1.0 - beta2) * (grad * grad)

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step

    m_hat = new_m / bc1
    v_hat = new_v / bc2

    adam_step = m_hat / (np.sqrt(v_hat) + eps)
    r = adam_step + weight_decay * param

    w_norm = float(np.linalg.norm(param))
    r_norm = float(np.linalg.norm(r))

    if w_norm > 0 and r_norm > 0:
        trust_ratio = w_norm / r_norm
        if trust_clip:
            trust_ratio = min(trust_ratio, 10.0)
    else:
        trust_ratio = 1.0

    new_param = param - lr * trust_ratio * r
    return new_param, new_m, new_v, trust_ratio


def compute_gradient_noise_scale_np(
    per_sample_grads: np.ndarray,
) -> tuple[float, float, float]:
    B, D = per_sample_grads.shape
    g_mean = np.mean(per_sample_grads, axis=0)
    g_norm_sq = float(np.sum(g_mean * g_mean))

    diff = per_sample_grads - g_mean[None, :]
    trace_sigma = float(np.sum(diff * diff) / float(B - 1))

    b_simple = trace_sigma / (g_norm_sq + 1e-12)
    return g_norm_sq, trace_sigma, b_simple


def verify_against_checkpoint(ckpt_path: str | Path) -> None:
    weights = load_file(str(ckpt_path))

    # 1. LARS
    param = weights["param"]
    grad = weights["grad"]
    v_init_lars = weights["v_init_lars"]
    p_lars, v_lars, ratio_lars = step_lars_np(
        param, grad, v_init_lars, lr=0.1, momentum=0.9, weight_decay=0.0001, trust_coef=0.001
    )
    err_p_lars = float(np.max(np.abs(p_lars - weights["p_lars"])))
    err_v_lars = float(np.max(np.abs(v_lars - weights["v_lars"])))
    err_ratio_lars = float(np.max(np.abs(ratio_lars - weights["ratio_lars"])))
    assert err_p_lars < 1e-5 and err_v_lars < 1e-5 and err_ratio_lars < 1e-5, f"LARS error: {err_p_lars}"

    # 2. LAMB
    m_init_lamb = weights["m_init_lamb"]
    v_init_lamb = weights["v_init_lamb"]
    p_lamb, m_lamb, v_lamb, ratio_lamb = step_lamb_np(
        param, grad, m_init_lamb, v_init_lamb, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.01
    )
    err_p_lamb = float(np.max(np.abs(p_lamb - weights["p_lamb"])))
    err_m_lamb = float(np.max(np.abs(m_lamb - weights["m_lamb"])))
    err_v_lamb = float(np.max(np.abs(v_lamb - weights["v_lamb"])))
    err_ratio_lamb = float(np.max(np.abs(ratio_lamb - weights["ratio_lamb"])))
    assert err_p_lamb < 1e-5 and err_m_lamb < 1e-5 and err_v_lamb < 1e-5 and err_ratio_lamb < 1e-5, f"LAMB error: {err_p_lamb}"

    # 3. Gradient Noise Scale
    per_sample_grads = weights["per_sample_grads"]
    g_norm_sq, trace_sigma, b_simple = compute_gradient_noise_scale_np(per_sample_grads)
    err_g_sq = float(np.max(np.abs(g_norm_sq - weights["g_norm_sq"])))
    err_tr = float(np.max(np.abs(trace_sigma - weights["trace_sigma"])))
    err_b = float(np.max(np.abs(b_simple - weights["b_simple"])))
    assert err_g_sq < 1e-5 and err_tr < 1e-5 and err_b < 1e-5, f"GNS error: {err_b}"

    # 4. Linear Anchor
    test_in = weights["test_input"]
    test_out = weights["test_output"]
    weight = weights["weight"]
    bias = weights["bias"]
    pred_y = test_in @ weight + bias
    err_anchor = float(np.max(np.abs(pred_y - test_out)))
    assert err_anchor < 1e-5, f"Linear anchor error: {err_anchor}"

    print(f"All 4 invariants verified in JAX/NumPy! Max error: {max(err_p_lars, err_p_lamb, err_b, err_anchor):.2e}")


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.safetensors")
    verify_against_checkpoint(p)
