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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import torch
    from safetensors.torch import save_file

    import core


    def step_sgd_momentum(
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum_buf: torch.Tensor,
        lr: float = 0.05,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # g = grad + weight_decay * param
        # v_{t+1} = momentum * v_t + g
        # p_{t+1} = param - lr * v_{t+1}
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
        # g = grad + weight_decay * param
        # G_{t+1} = G_t + g^2
        # p_{t+1} = param - lr * g / (sqrt(G_{t+1}) + eps)
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
        # g = grad + weight_decay * param
        # v_{t+1} = alpha * v_t + (1 - alpha) * g^2
        # p_{t+1} = param - lr * g / (sqrt(v_{t+1}) + eps)
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
        # Adam with coupled L2 regularization: g = grad + weight_decay * param
        # m_{t+1} = beta1 * m_t + (1 - beta1) * g
        # v_{t+1} = beta2 * v_t + (1 - beta2) * g^2
        # m_hat = m_{t+1} / (1 - beta1^t)
        # v_hat = v_{t+1} / (1 - beta2^t)
        # p_{t+1} = param - lr * m_hat / (sqrt(v_hat) + eps)
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
        # Decoupled AdamW (Loshchilov & Hutter 2017):
        # m_{t+1} = beta1 * m_t + (1 - beta1) * grad (pure gradient!)
        # v_{t+1} = beta2 * v_t + (1 - beta2) * grad^2
        # m_hat = m_{t+1} / (1 - beta1^t)
        # v_hat = v_{t+1} / (1 - beta2^t)
        # p_{t+1} = param - lr * weight_decay * param - lr * m_hat / (sqrt(v_hat) + eps)
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
        # AdaBelief (Zhuang et al. 2020):
        # Adapts step-size according to the "belief" in current gradient direction:
        # m_{t+1} = beta1 * m_t + (1 - beta1) * grad
        # diff = grad - m_{t+1}
        # s_{t+1} = beta2 * s_t + (1 - beta2) * (diff^2) + eps
        # m_hat = m_{t+1} / (1 - beta1^t)
        # s_hat = s_{t+1} / (1 - beta2^t)
        # p_{t+1} = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(s_hat) + eps)
        new_m = beta1 * m_buf + (1.0 - beta1) * grad
        diff = grad - new_m
        new_s = beta2 * s_buf + (1.0 - beta2) * (diff * diff) + eps
        m_hat = new_m / (1.0 - beta1**step)
        s_hat = new_s / (1.0 - beta2**step)
        new_param = param * (1.0 - lr * weight_decay) - lr * m_hat / (torch.sqrt(s_hat) + eps)
        return new_param, new_m, new_s


    def main() -> None:
        core.setup_env(seed=42)
        device = torch.device("cpu")

        # Dimensions
        dim = 8
        # Anchor inputs & params
        core_linear = torch.nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            core_linear.weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.1, 0.4, 0.05, -0.2, 0.15, -0.3, 0.25],
                        [-0.15, 0.3, -0.05, 0.2, 0.1, -0.25, 0.05, -0.1],
                        [0.1, -0.2, 0.3, -0.15, 0.2, 0.05, -0.1, 0.3],
                        [0.25, 0.05, -0.3, 0.1, -0.05, 0.2, -0.15, 0.1],
                        [-0.2, 0.15, 0.05, -0.1, 0.3, -0.05, 0.2, -0.25],
                        [0.05, -0.1, 0.2, 0.25, -0.15, 0.3, -0.05, 0.1],
                        [-0.3, 0.25, -0.1, 0.05, 0.2, -0.15, 0.1, -0.2],
                        [0.15, -0.05, 0.25, -0.2, 0.1, 0.05, -0.3, 0.15],
                    ],
                    dtype=torch.float32,
                )
            )
            core_linear.bias.copy_(
                torch.tensor([0.05, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02, -0.04], dtype=torch.float32)
            )

        # Deterministic test parameters & gradients (1D vector of length 8)
        param = torch.tensor([0.5, -0.3, 1.2, -0.8, 0.1, -1.5, 0.7, -0.2], dtype=torch.float32)
        grad = torch.tensor([0.04, -0.02, 0.15, -0.08, 0.01, -0.25, 0.12, -0.05], dtype=torch.float32)
        m_init = torch.tensor([0.01, -0.005, 0.03, -0.02, 0.005, -0.05, 0.02, -0.01], dtype=torch.float32)
        v_init = torch.tensor([0.002, 0.001, 0.01, 0.005, 0.001, 0.025, 0.008, 0.003], dtype=torch.float32)

        # 1. SGD Momentum
        p_sgd, v_sgd = step_sgd_momentum(param, grad, m_init, lr=0.05, momentum=0.9, weight_decay=0.01)

        # 2. Adagrad
        p_adagrad, g_adagrad = step_adagrad(param, grad, v_init, lr=0.05, eps=1e-8, weight_decay=0.01)

        # 3. RMSprop
        p_rmsprop, v_rmsprop = step_rmsprop(param, grad, v_init, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.01)

        # 4. Adam (Coupled L2)
        p_adam, m_adam, v_adam = step_adam(param, grad, m_init, v_init, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)

        # 5. AdamW (Decoupled Weight Decay)
        p_adamw, m_adamw, v_adamw = step_adamw(param, grad, m_init, v_init, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)

        # 6. AdaBelief
        p_adabelief, m_adabelief, s_adabelief = step_adabelief(param, grad, m_init, v_init, step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-16, weight_decay=0.01)

        # Anchor forward pass
        anchor_x = torch.tensor(
            [
                [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
                [-0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        weight = core_linear.weight.t().contiguous()
        bias = core_linear.bias
        anchor_y = anchor_x @ weight + bias

        tensors = {
            # Initial states
            "param": param,
            "grad": grad,
            "m_init": m_init,
            "v_init": v_init,
            # SGD Momentum
            "p_sgd": p_sgd,
            "v_sgd": v_sgd,
            # Adagrad
            "p_adagrad": p_adagrad,
            "g_adagrad": g_adagrad,
            # RMSprop
            "p_rmsprop": p_rmsprop,
            "v_rmsprop": v_rmsprop,
            # Adam
            "p_adam": p_adam,
            "m_adam": m_adam,
            "v_adam": v_adam,
            # AdamW
            "p_adamw": p_adamw,
            "m_adamw": m_adamw,
            "v_adamw": v_adamw,
            # AdaBelief
            "p_adabelief": p_adabelief,
            "m_adabelief": m_adabelief,
            "s_adabelief": s_adabelief,
            # Linear Anchor
            "core_weight": core_linear.weight,
            "core_bias": core_linear.bias,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "weight": weight,
            "bias": bias.clone(),
            "test_input": anchor_x.clone(),
            "test_output": anchor_y.clone(),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "01_first_order_and_adaptive_optimisers.safetensors"
        save_file(tensors, str(out_path))
        print(f"Saved {len(tensors)} tensors to {out_path}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.safetensors")
    verify_against_checkpoint(p)
