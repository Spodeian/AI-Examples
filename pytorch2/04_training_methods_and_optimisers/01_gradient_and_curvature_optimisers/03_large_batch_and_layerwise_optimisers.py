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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import torch
    from safetensors.torch import save_file

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
        # LARS (You et al. 2017):
        # Layer-wise Adaptive Rate Scaling
        # w_norm = ||w||_2
        # g_norm = ||g||_2
        # If w_norm > 0 and g_norm > 0:
        #   trust_ratio = trust_coef * w_norm / (g_norm + weight_decay * w_norm + eps)
        # else:
        #   trust_ratio = 1.0
        # local_lr = lr * trust_ratio
        # scaled_g = local_lr * (grad + weight_decay * param)
        # v_{t+1} = momentum * v_t + scaled_g
        # p_{t+1} = param - v_{t+1}
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
        # LAMB (You et al. 2019):
        # Layer-wise Adaptive Moments for Batch training
        # m_{t+1} = beta1 * m_t + (1 - beta1) * grad
        # v_{t+1} = beta2 * v_t + (1 - beta2) * (grad^2)
        # m_hat = m_{t+1} / (1 - beta1^t)
        # v_hat = v_{t+1} / (1 - beta2^t)
        # r = m_hat / (sqrt(v_hat) + eps) + weight_decay * param
        # r_norm = ||r||_2, w_norm = ||w||_2
        # If w_norm > 0 and r_norm > 0:
        #   trust_ratio = w_norm / r_norm
        # else:
        #   trust_ratio = 1.0
        # p_{t+1} = param - lr * trust_ratio * r
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
        # McCandlish et al. (2018):
        # per_sample_grads: [B, D]
        # G = mean(per_sample_grads, dim=0) -> [D]
        # ||G||^2 = sum(G_i^2)
        # Var(g) = 1/(B - 1) sum_b ||g_b - G||^2
        # B_simple = Tr(Sigma) / ||G||^2
        B, D = per_sample_grads.shape
        g_mean = torch.mean(per_sample_grads, dim=0) # [D]
        g_norm_sq = torch.sum(g_mean * g_mean) # ||G||^2

        diff = per_sample_grads - g_mean.unsqueeze(0) # [B, D]
        trace_sigma = torch.sum(diff * diff) / float(B - 1)

        b_simple = trace_sigma / (g_norm_sq + 1e-12)
        return g_norm_sq, trace_sigma, b_simple


    def main() -> None:
        core.setup_env(seed=42)
        device = torch.device("cpu")

        dim = 8
        # 1. Linear Anchor (Test standard: test_input @ weight + bias = test_output)
        gen = torch.Generator().manual_seed(42)
        weight = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
        bias = torch.randn(dim, generator=gen, dtype=torch.float32)
        test_input = torch.randn(4, dim, generator=gen, dtype=torch.float32)
        test_output = test_input @ weight + bias

        # 2. Layerwise Parameters & Gradients (1D layer slice of length 8)
        param = torch.tensor([1.2, -0.8, 2.5, -1.4, 0.6, -3.0, 1.8, -0.5], dtype=torch.float32)
        grad = torch.tensor([0.25, -0.15, 0.40, -0.30, 0.10, -0.60, 0.35, -0.10], dtype=torch.float32)
        v_init_lars = torch.tensor([0.02, -0.01, 0.04, -0.03, 0.01, -0.05, 0.03, -0.01], dtype=torch.float32)

        # 3. Step LARS
        p_lars, v_lars, ratio_lars = step_lars(
            param, grad, v_init_lars, lr=0.1, momentum=0.9, weight_decay=0.0001, trust_coef=0.001
        )

        # 4. Step LAMB
        m_init_lamb = torch.tensor([0.03, -0.02, 0.05, -0.04, 0.01, -0.08, 0.04, -0.02], dtype=torch.float32)
        v_init_lamb = torch.tensor([0.01, 0.005, 0.02, 0.015, 0.003, 0.045, 0.018, 0.005], dtype=torch.float32)

        p_lamb, m_lamb, v_lamb, ratio_lamb = step_lamb(
            param, grad, m_init_lamb, v_init_lamb, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.01
        )

        # 5. Gradient Noise Scale (Batch B=8, Dim D=4)
        per_sample_grads = torch.tensor([
            [0.10, -0.05, 0.20, -0.15],
            [0.12, -0.04, 0.18, -0.14],
            [0.09, -0.06, 0.22, -0.16],
            [0.11, -0.05, 0.19, -0.15],
            [0.08, -0.07, 0.21, -0.13],
            [0.13, -0.03, 0.17, -0.17],
            [0.10, -0.05, 0.20, -0.15],
            [0.11, -0.04, 0.21, -0.16],
        ], dtype=torch.float32)
        g_norm_sq, trace_sigma, b_simple = compute_gradient_noise_scale(per_sample_grads)

        tensors = {
            # Linear Anchor
            "weight": weight,
            "bias": bias,
            "test_input": test_input,
            "test_output": test_output,
            # Inputs
            "param": param,
            "grad": grad,
            "v_init_lars": v_init_lars,
            "m_init_lamb": m_init_lamb,
            "v_init_lamb": v_init_lamb,
            # LARS outputs
            "p_lars": p_lars,
            "v_lars": v_lars,
            "ratio_lars": ratio_lars.unsqueeze(0),
            # LAMB outputs
            "p_lamb": p_lamb,
            "m_lamb": m_lamb,
            "v_lamb": v_lamb,
            "ratio_lamb": ratio_lamb.unsqueeze(0),
            # Gradient Noise Scale
            "per_sample_grads": per_sample_grads,
            "g_norm_sq": g_norm_sq.unsqueeze(0),
            "trace_sigma": trace_sigma.unsqueeze(0),
            "b_simple": b_simple.unsqueeze(0),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "03_large_batch_and_layerwise_optimisers.safetensors"
        save_file(tensors, str(out_path))
        print(f"Saved {len(tensors)} tensors to {out_path}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.safetensors")
    verify_against_checkpoint(p)
