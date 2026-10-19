"""JAX / NumPy implementation for Category 04 Topic 02:
02_modern_sign_and_factored_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def step_lion_np(
    param: np.ndarray,
    grad: np.ndarray,
    exp_avg: np.ndarray,
    lr: float = 0.0001,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    update = np.sign(beta1 * exp_avg + (1.0 - beta1) * grad)
    new_param = param * (1.0 - lr * weight_decay) - lr * update
    new_exp_avg = beta2 * exp_avg + (1.0 - beta2) * grad
    return new_param, new_exp_avg


def step_adafactor_2d_np(
    param: np.ndarray,
    grad: np.ndarray,
    r_buf: np.ndarray,
    c_buf: np.ndarray,
    step: int = 2,
    lr: float = 0.001,
    beta2: float = 0.8,
    eps1: float = 1e-30,
    eps2: float = 1e-3,
    clip_threshold: float = 1.0,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decay_rate = 1.0 - (step ** (-beta2))
    s = grad * grad + eps1
    new_r = decay_rate * r_buf + (1.0 - decay_rate) * np.mean(s, axis=1)
    new_c = decay_rate * c_buf + (1.0 - decay_rate) * np.mean(s, axis=0)

    r_mean = np.mean(new_r)
    v_approx = (new_r[:, None] * new_c[None, :]) / (r_mean + eps1)

    u = grad / (np.sqrt(v_approx) + eps2)
    rms_u = np.sqrt(np.mean(u * u))
    clip_factor = np.maximum(1.0, rms_u / clip_threshold)
    u_clipped = u / clip_factor

    new_param = param * (1.0 - lr * weight_decay) - lr * u_clipped
    return new_param, new_r, new_c


def step_ademamix_np(
    param: np.ndarray,
    grad: np.ndarray,
    m1_buf: np.ndarray,
    m2_buf: np.ndarray,
    v_buf: np.ndarray,
    step: int = 2,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta3: float = 0.9999,
    alpha: float = 5.0,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    new_m1 = beta1 * m1_buf + (1.0 - beta1) * grad
    new_m2 = beta3 * m2_buf + (1.0 - beta3) * grad
    new_v = beta2 * v_buf + (1.0 - beta2) * (grad * grad)

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    bc3 = 1.0 - beta3**step

    m1_hat = new_m1 / bc1
    m2_hat = new_m2 / bc3
    v_hat = new_v / bc2

    combined_m = m1_hat + alpha * m2_hat
    u = combined_m / (np.sqrt(v_hat) + eps)

    new_param = param * (1.0 - lr * weight_decay) - lr * u
    return new_param, new_m1, new_m2, new_v


def verify_against_checkpoint(ckpt_path: str | Path) -> None:
    weights = load_file(str(ckpt_path))

    # 1. Lion (1D)
    param_1d = weights["param_1d"]
    grad_1d = weights["grad_1d"]
    m_1d = weights["m_1d"]
    p_lion, m_lion = step_lion_np(param_1d, grad_1d, m_1d, lr=0.0001, beta1=0.9, beta2=0.99, weight_decay=0.1)
    err_p_lion = float(np.max(np.abs(p_lion - weights["p_lion"])))
    err_m_lion = float(np.max(np.abs(m_lion - weights["m_lion"])))
    assert err_p_lion < 1e-5 and err_m_lion < 1e-5, f"Lion error: {err_p_lion}, {err_m_lion}"

    # 2. Adafactor (2D)
    param_2d = weights["param_2d"]
    grad_2d = weights["grad_2d"]
    r_buf = weights["r_buf"]
    c_buf = weights["c_buf"]
    p_adafactor, r_adafactor, c_adafactor = step_adafactor_2d_np(
        param_2d, grad_2d, r_buf, c_buf, step=2, lr=0.001, beta2=0.8, weight_decay=0.01
    )
    err_p_ada = float(np.max(np.abs(p_adafactor - weights["p_adafactor"])))
    err_r_ada = float(np.max(np.abs(r_adafactor - weights["r_adafactor"])))
    err_c_ada = float(np.max(np.abs(c_adafactor - weights["c_adafactor"])))
    assert err_p_ada < 1e-5 and err_r_ada < 1e-5 and err_c_ada < 1e-5, f"Adafactor error: {err_p_ada}"

    # 3. AdEMAMix (1D)
    m2_1d = weights["m2_1d"]
    v_1d = weights["v_1d"]
    p_ademamix, m1_ademamix, m2_ademamix, v_ademamix = step_ademamix_np(
        param_1d, grad_1d, m_1d, m2_1d, v_1d, step=2, lr=0.001, beta1=0.9, beta2=0.999, beta3=0.9999, alpha=5.0, eps=1e-8, weight_decay=0.01
    )
    err_p_adem = float(np.max(np.abs(p_ademamix - weights["p_ademamix"])))
    err_m1_adem = float(np.max(np.abs(m1_ademamix - weights["m1_ademamix"])))
    err_m2_adem = float(np.max(np.abs(m2_ademamix - weights["m2_ademamix"])))
    err_v_adem = float(np.max(np.abs(v_ademamix - weights["v_ademamix"])))
    assert err_p_adem < 1e-5 and err_m1_adem < 1e-5 and err_m2_adem < 1e-5 and err_v_adem < 1e-5, f"AdEMAMix error: {err_p_adem}"

    # 4. Linear Anchor
    test_in = weights["test_input"]
    test_out = weights["test_output"]
    weight = weights["weight"]
    bias = weights["bias"]
    pred_y = test_in @ weight + bias
    err_anchor = float(np.max(np.abs(pred_y - test_out)))
    assert err_anchor < 1e-5, f"Linear anchor error: {err_anchor}"

    print(f"All 4 invariants verified in JAX/NumPy! Max error: {max(err_p_lion, err_p_ada, err_p_adem, err_anchor):.2e}")


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.safetensors")
    verify_against_checkpoint(p)
