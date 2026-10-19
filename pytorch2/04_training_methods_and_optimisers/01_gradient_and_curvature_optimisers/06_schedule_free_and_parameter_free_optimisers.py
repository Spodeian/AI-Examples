"""Schedule-Free and Parameter-Free Optimisers in PyTorch 2.x.

Implements learning-rate schedule-free and adaptive parameter-free optimization algorithms:
1. Schedule-Free AdamW (Defazio et al. 2024): Continuous iterate averaging eliminating LR schedules.
2. Polyak Step-Size: Exact optimal step scaling using function value gaps.
3. Barzilai-Borwein Spectral Steps (BB1): Non-monotonic two-point gradient step estimation.
4. D-Adaptation: Automatic distance-to-solution tracking for parameter-free learning rates.

Enforces strict transitive parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file


def step_schedule_free_adamw(
    x: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    step: int = 1,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one Schedule-Free AdamW update step.

    Maintains base buffer z_t, running average x_t, and training iterate y_t:
        m_{t+1} = beta1 * m_t + (1 - beta1) * g_t
        v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2
        v_hat = v_{t+1} / (1 - beta2^step)
        u_t = m_{t+1} / (sqrt(v_hat) + eps) + weight_decay * y_t
        z_{t+1} = z_t - lr * u_t
        c_prime = 1 / (step + 1)
        x_{t+1} = (1 - c_prime) * x_t + c_prime * z_{t+1}
        c_next = 1 / (step + 2)
        y_{t+1} = (1 - c_next) * z_{t+1} + c_next * x_{t+1}
    """
    m_next = beta1 * m + (1.0 - beta1) * grad
    v_next = beta2 * v + (1.0 - beta2) * (grad ** 2)

    bias_v = float(1.0 - beta2 ** step)
    v_hat = v_next / bias_v

    u_t = (m_next / (torch.sqrt(v_hat) + eps)) + weight_decay * y
    z_next = z - lr * u_t

    c_prime = float(1.0 / (step + 1))
    x_next = (1.0 - c_prime) * x + c_prime * z_next

    c_next = float(1.0 / (step + 2))
    y_next = (1.0 - c_next) * z_next + c_next * x_next

    return x_next, z_next, y_next, m_next, v_next


def step_polyak(
    param: torch.Tensor,
    grad: torch.Tensor,
    f_val: float,
    f_opt: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Perform one step using Polyak's optimal step-size: gamma = (f(x) - f*) / ||g||^2."""
    grad_norm_sq = torch.sum(grad ** 2)
    gamma = (f_val - f_opt) / (grad_norm_sq + eps)
    return param - gamma * grad


def step_barzilai_borwein(
    param: torch.Tensor,
    grad: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Perform one step using Barzilai-Borwein BB1 spectral step-size: gamma = (s^T s) / (s^T y)."""
    s_dot_s = torch.dot(s, s)
    s_dot_y = torch.dot(s, y)
    gamma = s_dot_s / (s_dot_y + eps)
    return param - gamma * grad


def step_d_adaptation(
    param: torch.Tensor,
    grad: torch.Tensor,
    d_est: float,
    lr0: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """Perform one D-Adaptation step with distance-to-solution tracking."""
    dim = param.numel()
    grad_norm_l1 = float(torch.sum(torch.abs(grad)))
    grad_norm_l2 = torch.norm(grad)

    d_next = max(0.1, d_est + 0.05 * (grad_norm_l1 / (dim ** 0.5) - 1.0))
    eff_lr = lr0 * d_next
    param_next = param - eff_lr * (grad / (grad_norm_l2 + eps))
    return param_next, d_next


class LinearAnchor(nn.Module):
    """Linear projection layer serving as deterministic anchor across all tracks."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class ScheduleFreeUnifiedModule(nn.Module):
    """Unified module encapsulating schedule-free and parameter-free optimisers."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        tensors = load_file(str(checkpoint_path))
        self.sf_x = tensors["sf_x"]
        self.sf_z = tensors["sf_z"]
        self.sf_y = tensors["sf_y"]
        self.sf_grad = tensors["sf_grad"]
        self.sf_m = tensors["sf_m"]
        self.sf_v = tensors["sf_v"]

        self.polyak_param = tensors["polyak_param"]
        self.polyak_grad = tensors["polyak_grad"]
        self.f_val = float(tensors["f_val"][0])
        self.f_opt = float(tensors["f_opt"][0])

        self.bb_param = tensors["bb_param"]
        self.bb_grad = tensors["bb_grad"]
        self.bb_s = tensors["bb_s"]
        self.bb_y = tensors["bb_y"]

        self.d_param = tensors["d_param"]
        self.d_grad = tensors["d_grad"]
        self.d_est = float(tensors["d_est"][0])

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(tensors["anchor_w"].T)
        self.anchor.bias.data.copy_(tensors["anchor_b"])

        self.golden = {
            "sf_x": tensors["golden_sf_x"],
            "sf_z": tensors["golden_sf_z"],
            "sf_y": tensors["golden_sf_y"],
            "sf_m": tensors["golden_sf_m"],
            "sf_v": tensors["golden_sf_v"],
            "polyak_param": tensors["golden_polyak_param"],
            "bb_param": tensors["golden_bb_param"],
            "d_adapt_param": tensors["golden_d_adapt_param"],
            "anchor_out": tensors["golden_anchor_out"],
            "anchor_x": tensors["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: Schedule-Free AdamW
        x_next, z_next, y_next, m_next, v_next = step_schedule_free_adamw(
            self.sf_x, self.sf_z, self.sf_y, self.sf_grad, self.sf_m, self.sf_v,
            lr=0.01, beta1=0.9, beta2=0.999, weight_decay=0.01, step=1, eps=1e-8,
        )
        err_sf_x = float(torch.max(torch.abs(x_next - self.golden["sf_x"])))
        err_sf_z = float(torch.max(torch.abs(z_next - self.golden["sf_z"])))
        err_sf_y = float(torch.max(torch.abs(y_next - self.golden["sf_y"])))
        err_sf = max(err_sf_x, err_sf_z, err_sf_y)
        assert err_sf < atol, f"Schedule-Free invariant failed: {err_sf:.2e}"

        # Invariant 2: Polyak Step-Size
        p_polyak = step_polyak(self.polyak_param, self.polyak_grad, self.f_val, self.f_opt)
        err_polyak = float(torch.max(torch.abs(p_polyak - self.golden["polyak_param"])))
        assert err_polyak < atol, f"Polyak invariant failed: {err_polyak:.2e}"

        # Invariant 3: Barzilai-Borwein Spectral Step
        p_bb = step_barzilai_borwein(self.bb_param, self.bb_grad, self.bb_s, self.bb_y)
        err_bb = float(torch.max(torch.abs(p_bb - self.golden["bb_param"])))
        assert err_bb < atol, f"Barzilai-Borwein invariant failed: {err_bb:.2e}"

        # Invariant 4: D-Adaptation
        p_dadapt, _ = step_d_adaptation(self.d_param, self.d_grad, self.d_est, lr0=0.01)
        err_dadapt = float(torch.max(torch.abs(p_dadapt - self.golden["d_adapt_param"])))
        assert err_dadapt < atol, f"D-Adaptation invariant failed: {err_dadapt:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor(self.golden["anchor_x"]).detach()
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden["anchor_out"])))
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"

        return {
            "schedule_free_err": err_sf,
            "polyak_err": err_polyak,
            "bb_err": err_bb,
            "d_adapt_err": err_dadapt,
            "anchor_err": err_anchor,
        }


def main():
    ckpt_path = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.safetensors")
    module = ScheduleFreeUnifiedModule(ckpt_path)
    errors = module.verify_invariants()
    print("PyTorch 2.x Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


if __name__ == "__main__":
    main()
