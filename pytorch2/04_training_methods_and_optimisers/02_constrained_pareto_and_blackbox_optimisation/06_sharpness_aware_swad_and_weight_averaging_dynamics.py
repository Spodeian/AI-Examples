"""PyTorch 2.x implementation for Category 04 Topic 13:
Sharpness-Aware Minimization (SAM), SWAD, and Weight Averaging Dynamics.

Provides production-grade implementations of:
1. Sharpness-Aware Minimization (SAM) dual ascent gradient perturbation and update.
2. Lookahead optimizer synchronization between slow and fast weights.
3. Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA).
4. Uniform Model Soups weight-space averaging.
"""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


def step_sam(
    param: torch.Tensor,
    grad_at_w: torch.Tensor,
    grad_at_pert: torch.Tensor,
    rho: float = 0.05,
    lr: float = 0.01,
    weight_decay: float = 0.001,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sharpness-Aware Minimization (SAM) update step."""
    grad_norm = torch.norm(grad_at_w, p=2) + eps
    epsilon = rho * (grad_at_w / grad_norm)
    update = grad_at_pert + weight_decay * param
    param_next = param - lr * update
    return param_next, epsilon


def step_lookahead(
    slow_param: torch.Tensor,
    fast_param: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lookahead synchronization step."""
    slow_next = slow_param + alpha * (fast_param - slow_param)
    fast_next = slow_next.clone()
    return slow_next, fast_next


def step_swa_ema(
    swa_avg: torch.Tensor,
    ema_avg: torch.Tensor,
    current_param: torch.Tensor,
    n_models: int,
    decay: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SWA running average and EMA step."""
    new_swa = (swa_avg * n_models + current_param) / (n_models + 1)
    new_ema = decay * ema_avg + (1.0 - decay) * current_param
    return new_swa, new_ema


def uniform_model_soup(models: torch.Tensor) -> torch.Tensor:
    """Average a stack of M models: models shape [M, d]."""
    return torch.mean(models, dim=0)


class LinearAnchor(nn.Module):
    """Linear anchor layer for cross-track parity verification."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class SharpnessAwareUnifiedModule(nn.Module):
    """Unified module encapsulating sharpness-aware and weight-averaging algorithms."""

    def __init__(self, weights_path: str | Path):
        super().__init__()
        weights = load_file(str(weights_path))

        self.param_sam = weights["param_sam"]
        self.grad_at_w = weights["grad_at_w"]
        self.grad_at_pert = weights["grad_at_pert"]
        self.golden_param_sam = weights["golden_param_sam"]
        self.golden_epsilon_sam = weights["golden_epsilon_sam"]

        self.slow_p = weights["slow_p"]
        self.fast_p = weights["fast_p"]
        self.golden_slow_p = weights["golden_slow_p"]
        self.golden_fast_p = weights["golden_fast_p"]

        self.swa_avg = weights["swa_avg"]
        self.ema_avg = weights["ema_avg"]
        self.curr_param = weights["curr_param"]
        self.golden_swa = weights["golden_swa"]
        self.golden_ema = weights["golden_ema"]

        self.soup_models = weights["soup_models"]
        self.golden_soup = weights["golden_soup"]

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verify parity against golden reference tensors."""
        errors: dict[str, float] = {}

        # 1. SAM Invariant
        p_sam, eps_sam = step_sam(self.param_sam, self.grad_at_w, self.grad_at_pert, rho=0.05, lr=0.01, weight_decay=0.001)
        err_p_sam = float(torch.max(torch.abs(p_sam - self.golden_param_sam)).item())
        err_eps_sam = float(torch.max(torch.abs(eps_sam - self.golden_epsilon_sam)).item())
        err_sam = max(err_p_sam, err_eps_sam)
        assert err_sam < atol, f"SAM invariant failed: err={err_sam:.2e} >= {atol:.2e}"
        errors["sam_err"] = err_sam

        # 2. Lookahead Invariant
        slow_out, fast_out = step_lookahead(self.slow_p, self.fast_p, alpha=0.5)
        err_slow = float(torch.max(torch.abs(slow_out - self.golden_slow_p)).item())
        err_fast = float(torch.max(torch.abs(fast_out - self.golden_fast_p)).item())
        err_lookahead = max(err_slow, err_fast)
        assert err_lookahead < atol, f"Lookahead invariant failed: err={err_lookahead:.2e} >= {atol:.2e}"
        errors["lookahead_err"] = err_lookahead

        # 3. SWA & EMA Invariant
        new_swa, new_ema = step_swa_ema(self.swa_avg, self.ema_avg, self.curr_param, n_models=4, decay=0.99)
        err_swa = float(torch.max(torch.abs(new_swa - self.golden_swa)).item())
        err_ema = float(torch.max(torch.abs(new_ema - self.golden_ema)).item())
        err_avg = max(err_swa, err_ema)
        assert err_avg < atol, f"SWA/EMA invariant failed: err={err_avg:.2e} >= {atol:.2e}"
        errors["swa_ema_err"] = err_avg

        # 4. Model Soup Invariant
        soup_out = uniform_model_soup(self.soup_models)
        err_soup = float(torch.max(torch.abs(soup_out - self.golden_soup)).item())
        assert err_soup < atol, f"Model Soup invariant failed: err={err_soup:.2e} >= {atol:.2e}"
        errors["soup_err"] = err_soup

        # 5. Linear Anchor Invariant
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics.safetensors")
    module = SharpnessAwareUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
