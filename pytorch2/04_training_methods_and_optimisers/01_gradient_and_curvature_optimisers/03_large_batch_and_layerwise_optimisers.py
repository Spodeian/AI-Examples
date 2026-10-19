"""PyTorch 2.x implementation for Category 04 Topic 03:
03_large_batch_and_layerwise_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file

import core


def step_lars(
    param: torch.Tensor,
    grad: torch.Tensor,
    v_buf: torch.Tensor,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    trust_coef: float = 0.001,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_norm = torch.norm(param)
    g_norm = torch.norm(grad)

    if w_norm > 0 and g_norm > 0:
        trust_ratio = trust_coef * w_norm / (g_norm + weight_decay * w_norm + eps)
    else:
        trust_ratio = torch.tensor(1.0, device=param.device, dtype=param.dtype)

    local_lr = lr * trust_ratio
    scaled_g = local_lr * (grad + weight_decay * param)
    new_v = momentum * v_buf + scaled_g
    new_param = param - new_v
    return new_param, new_v, trust_ratio


def step_lamb(
    param: torch.Tensor,
    grad: torch.Tensor,
    m_buf: torch.Tensor,
    v_buf: torch.Tensor,
    step: int = 2,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-6,
    weight_decay: float = 0.01,
    trust_clip: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_m = beta1 * m_buf + (1.0 - beta1) * grad
    new_v = beta2 * v_buf + (1.0 - beta2) * (grad * grad)

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step

    m_hat = new_m / bc1
    v_hat = new_v / bc2

    adam_step = m_hat / (torch.sqrt(v_hat) + eps)
    r = adam_step + weight_decay * param

    w_norm = torch.norm(param)
    r_norm = torch.norm(r)

    if w_norm > 0 and r_norm > 0:
        trust_ratio = w_norm / r_norm
        if trust_clip:
            trust_ratio = torch.clamp(trust_ratio, max=10.0)
    else:
        trust_ratio = torch.tensor(1.0, device=param.device, dtype=param.dtype)

    new_param = param - lr * trust_ratio * r
    return new_param, new_m, new_v, trust_ratio


def compute_gradient_noise_scale(
    per_sample_grads: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, D = per_sample_grads.shape
    g_mean = torch.mean(per_sample_grads, dim=0)
    g_norm_sq = torch.sum(g_mean * g_mean)

    diff = per_sample_grads - g_mean.unsqueeze(0)
    trace_sigma = torch.sum(diff * diff) / float(B - 1)

    b_simple = trace_sigma / (g_norm_sq + 1e-12)
    return g_norm_sq, trace_sigma, b_simple


class LargeBatchOptimisersUnifiedModule(nn.Module):
    """Unified module validating LARS, LAMB, and Gradient Noise Scale."""

    def __init__(self, in_dim: int = 8, out_dim: int = 8, ckpt_path: str | Path | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        if ckpt_path is not None:
            self.load_reference_weights(ckpt_path)

    def load_reference_weights(self, ckpt_path: str | Path) -> None:
        weights = load_file(str(ckpt_path))
        with torch.no_grad():
            self.linear.weight.copy_(weights["weight"].t())
            self.linear.bias.copy_(weights["bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def verify_against_checkpoint(ckpt_path: str | Path) -> None:
    weights = load_file(str(ckpt_path))

    # 1. LARS
    param = weights["param"]
    grad = weights["grad"]
    v_init_lars = weights["v_init_lars"]
    p_lars, v_lars, ratio_lars = step_lars(
        param, grad, v_init_lars, lr=0.1, momentum=0.9, weight_decay=0.0001, trust_coef=0.001
    )
    err_p_lars = float(torch.max(torch.abs(p_lars - weights["p_lars"])))
    err_v_lars = float(torch.max(torch.abs(v_lars - weights["v_lars"])))
    err_ratio_lars = float(torch.max(torch.abs(ratio_lars - weights["ratio_lars"])))
    assert err_p_lars < 1e-6 and err_v_lars < 1e-6 and err_ratio_lars < 1e-6, f"LARS error: {err_p_lars}"

    # 2. LAMB
    m_init_lamb = weights["m_init_lamb"]
    v_init_lamb = weights["v_init_lamb"]
    p_lamb, m_lamb, v_lamb, ratio_lamb = step_lamb(
        param, grad, m_init_lamb, v_init_lamb, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.01
    )
    err_p_lamb = float(torch.max(torch.abs(p_lamb - weights["p_lamb"])))
    err_m_lamb = float(torch.max(torch.abs(m_lamb - weights["m_lamb"])))
    err_v_lamb = float(torch.max(torch.abs(v_lamb - weights["v_lamb"])))
    err_ratio_lamb = float(torch.max(torch.abs(ratio_lamb - weights["ratio_lamb"])))
    assert err_p_lamb < 1e-6 and err_m_lamb < 1e-6 and err_v_lamb < 1e-6 and err_ratio_lamb < 1e-6, f"LAMB error: {err_p_lamb}"

    # 3. Gradient Noise Scale
    per_sample_grads = weights["per_sample_grads"]
    g_norm_sq, trace_sigma, b_simple = compute_gradient_noise_scale(per_sample_grads)
    err_g_sq = float(torch.max(torch.abs(g_norm_sq - weights["g_norm_sq"])))
    err_tr = float(torch.max(torch.abs(trace_sigma - weights["trace_sigma"])))
    err_b = float(torch.max(torch.abs(b_simple - weights["b_simple"])))
    assert err_g_sq < 1e-6 and err_tr < 1e-6 and err_b < 1e-6, f"GNS error: {err_b}"

    # 4. Linear Anchor
    test_in = weights["test_input"]
    test_out = weights["test_output"]
    model = LargeBatchOptimisersUnifiedModule(ckpt_path=ckpt_path)
    pred_y = model(test_in)
    err_anchor = float(torch.max(torch.abs(pred_y.detach() - test_out)))
    assert err_anchor < 1e-6, f"Linear anchor error: {err_anchor}"

    print(f"All 4 invariants verified successfully! Max error: {max(err_p_lars, err_p_lamb, err_b, err_anchor):.2e}")


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.safetensors")
    verify_against_checkpoint(p)
