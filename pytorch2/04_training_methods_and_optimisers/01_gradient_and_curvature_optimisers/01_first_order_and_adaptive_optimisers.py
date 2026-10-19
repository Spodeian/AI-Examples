"""PyTorch 2.x implementation for Category 04 Topic 01:
01_first_order_and_adaptive_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file

import core


def step_sgd_momentum(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buf: torch.Tensor,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = grad + weight_decay * param
    new_v = momentum * momentum_buf + g
    new_param = param - lr * new_v
    return new_param, new_v


def step_adagrad(
    param: torch.Tensor,
    grad: torch.Tensor,
    accum_buf: torch.Tensor,
    lr: float = 0.05,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = grad + weight_decay * param
    new_accum = accum_buf + g * g
    new_param = param - lr * g / (torch.sqrt(new_accum) + eps)
    return new_param, new_accum


def step_rmsprop(
    param: torch.Tensor,
    grad: torch.Tensor,
    square_buf: torch.Tensor,
    lr: float = 0.01,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = grad + weight_decay * param
    new_v = alpha * square_buf + (1.0 - alpha) * (g * g)
    new_param = param - lr * g / (torch.sqrt(new_v) + eps)
    return new_param, new_v


def step_adam(
    param: torch.Tensor,
    grad: torch.Tensor,
    m_buf: torch.Tensor,
    v_buf: torch.Tensor,
    step: int = 1,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = grad + weight_decay * param
    new_m = beta1 * m_buf + (1.0 - beta1) * g
    new_v = beta2 * v_buf + (1.0 - beta2) * (g * g)
    m_hat = new_m / (1.0 - beta1**step)
    v_hat = new_v / (1.0 - beta2**step)
    new_param = param - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return new_param, new_m, new_v


def step_adamw(
    param: torch.Tensor,
    grad: torch.Tensor,
    m_buf: torch.Tensor,
    v_buf: torch.Tensor,
    step: int = 1,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_m = beta1 * m_buf + (1.0 - beta1) * grad
    new_v = beta2 * v_buf + (1.0 - beta2) * (grad * grad)
    m_hat = new_m / (1.0 - beta1**step)
    v_hat = new_v / (1.0 - beta2**step)
    new_param = param * (1.0 - lr * weight_decay) - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return new_param, new_m, new_v


def step_adabelief(
    param: torch.Tensor,
    grad: torch.Tensor,
    m_buf: torch.Tensor,
    s_buf: torch.Tensor,
    step: int = 1,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-16,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_m = beta1 * m_buf + (1.0 - beta1) * grad
    diff = grad - new_m
    new_s = beta2 * s_buf + (1.0 - beta2) * (diff * diff) + eps
    m_hat = new_m / (1.0 - beta1**step)
    s_hat = new_s / (1.0 - beta2**step)
    new_param = param * (1.0 - lr * weight_decay) - lr * m_hat / (torch.sqrt(s_hat) + eps)
    return new_param, new_m, new_s


class FirstOrderOptimisersUnifiedModule(nn.Module):
    """Unified module validating first-order and adaptive optimiser update steps."""

    def __init__(self, ckpt_path: str | Path | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=True)
        if ckpt_path is not None:
            self.load_reference_weights(ckpt_path)

    def load_reference_weights(self, ckpt_path: str | Path) -> None:
        weights = load_file(str(ckpt_path))
        with torch.no_grad():
            self.linear.weight.copy_(weights["core_weight"])
            self.linear.bias.copy_(weights["core_bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def verify_against_checkpoint(ckpt_path: str | Path) -> None:
    weights = load_file(str(ckpt_path))
    param = weights["param"]
    grad = weights["grad"]
    m_init = weights["m_init"]
    v_init = weights["v_init"]

    # 1. SGD Momentum
    p_sgd, v_sgd = step_sgd_momentum(param, grad, m_init, lr=0.05, momentum=0.9, weight_decay=0.01)
    err_p_sgd = float(torch.max(torch.abs(p_sgd - weights["p_sgd"])))
    err_v_sgd = float(torch.max(torch.abs(v_sgd - weights["v_sgd"])))
    assert err_p_sgd < 1e-6 and err_v_sgd < 1e-6, f"SGD Momentum error: {err_p_sgd}, {err_v_sgd}"

    # 2. Adagrad
    p_adagrad, g_adagrad = step_adagrad(param, grad, v_init, lr=0.05, eps=1e-8, weight_decay=0.01)
    err_p_ada = float(torch.max(torch.abs(p_adagrad - weights["p_adagrad"])))
    err_g_ada = float(torch.max(torch.abs(g_adagrad - weights["g_adagrad"])))
    assert err_p_ada < 1e-6 and err_g_ada < 1e-6, f"Adagrad error: {err_p_ada}, {err_g_ada}"

    # 3. RMSprop
    p_rms, v_rms = step_rmsprop(param, grad, v_init, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.01)
    err_p_rms = float(torch.max(torch.abs(p_rms - weights["p_rmsprop"])))
    err_v_rms = float(torch.max(torch.abs(v_rms - weights["v_rmsprop"])))
    assert err_p_rms < 1e-6 and err_v_rms < 1e-6, f"RMSprop error: {err_p_rms}, {err_v_rms}"

    # 4. Adam
    p_adam, m_adam, v_adam = step_adam(param, grad, m_init, v_init, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    err_p_adam = float(torch.max(torch.abs(p_adam - weights["p_adam"])))
    err_m_adam = float(torch.max(torch.abs(m_adam - weights["m_adam"])))
    err_v_adam = float(torch.max(torch.abs(v_adam - weights["v_adam"])))
    assert err_p_adam < 1e-6 and err_m_adam < 1e-6 and err_v_adam < 1e-6, f"Adam error: {err_p_adam}"

    # 5. AdamW
    p_adamw, m_adamw, v_adamw = step_adamw(param, grad, m_init, v_init, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    err_p_adamw = float(torch.max(torch.abs(p_adamw - weights["p_adamw"])))
    err_m_adamw = float(torch.max(torch.abs(m_adamw - weights["m_adamw"])))
    err_v_adamw = float(torch.max(torch.abs(v_adamw - weights["v_adamw"])))
    assert err_p_adamw < 1e-6 and err_m_adamw < 1e-6 and err_v_adamw < 1e-6, f"AdamW error: {err_p_adamw}"

    # 6. AdaBelief
    p_adab, m_adab, s_adab = step_adabelief(param, grad, m_init, v_init, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-16, weight_decay=0.01)
    err_p_adab = float(torch.max(torch.abs(p_adab - weights["p_adabelief"])))
    err_m_adab = float(torch.max(torch.abs(m_adab - weights["m_adabelief"])))
    err_s_adab = float(torch.max(torch.abs(s_adab - weights["s_adabelief"])))
    assert err_p_adab < 1e-6 and err_m_adab < 1e-6 and err_s_adab < 1e-6, f"AdaBelief error: {err_p_adab}"

    # 7. Linear Anchor
    model = FirstOrderOptimisersUnifiedModule(ckpt_path)
    anchor_y = model(weights["anchor_x"])
    err_anchor = float(torch.max(torch.abs(anchor_y.detach() - weights["anchor_y"])))
    assert err_anchor < 1e-6, f"Anchor error: {err_anchor}"

    print(f"All 7 invariants verified successfully! Max error: {max(err_p_sgd, err_p_ada, err_p_rms, err_p_adam, err_p_adamw, err_p_adab, err_anchor):.2e}")


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.safetensors")
    verify_against_checkpoint(p)
