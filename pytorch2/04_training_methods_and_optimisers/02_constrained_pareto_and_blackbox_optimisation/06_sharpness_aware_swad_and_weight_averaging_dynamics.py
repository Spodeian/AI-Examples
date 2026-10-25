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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    from safetensors.torch import save_file


    def step_sam(
        param: torch.Tensor,
        grad_at_w: torch.Tensor,
        grad_at_pert: torch.Tensor,
        rho: float = 0.05,
        lr: float = 0.01,
        weight_decay: float = 0.001,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sharpness-Aware Minimization (SAM) update step.

        1. Perturbation: epsilon = rho * grad_at_w / (||grad_at_w||_2 + eps)
        2. Update: param_next = param - lr * (grad_at_pert + weight_decay * param)
        """
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
        """Lookahead synchronization step:
        slow_next = slow + alpha * (fast - slow)
        fast_next = slow_next
        """
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


    def main():
        torch.manual_seed(42)

        # 1. SAM Invariant
        param_sam = torch.tensor([1.2, -0.5, 0.8, -1.4], dtype=torch.float32)
        grad_at_w = torch.tensor([0.3, -0.1, 0.4, -0.2], dtype=torch.float32)
        grad_at_pert = torch.tensor([0.35, -0.08, 0.42, -0.18], dtype=torch.float32)
        golden_param_sam, golden_epsilon_sam = step_sam(
            param_sam, grad_at_w, grad_at_pert, rho=0.05, lr=0.01, weight_decay=0.001
        )

        # 2. Lookahead Invariant
        slow_p = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32)
        fast_p = torch.tensor([1.4, 1.8, -0.6, 0.9], dtype=torch.float32)
        golden_slow_p, golden_fast_p = step_lookahead(slow_p, fast_p, alpha=0.5)

        # 3. SWA & EMA Invariant
        swa_avg = torch.tensor([0.8, 1.2, -0.5, 0.4], dtype=torch.float32)
        ema_avg = torch.tensor([0.85, 1.15, -0.48, 0.42], dtype=torch.float32)
        curr_param = torch.tensor([0.9, 1.1, -0.4, 0.45], dtype=torch.float32)
        golden_swa, golden_ema = step_swa_ema(swa_avg, ema_avg, curr_param, n_models=4, decay=0.99)

        # 4. Model Soup Invariant
        soup_models = torch.tensor([
            [1.0, 2.0, -1.0, 0.5],
            [1.2, 1.9, -0.9, 0.6],
            [0.9, 2.1, -1.1, 0.4],
        ], dtype=torch.float32)
        golden_soup = uniform_model_soup(soup_models)

        # 5. Linear Anchor for cross-track parity
        anchor_x = torch.randn(2, 8, dtype=torch.float32)
        anchor_w = torch.randn(8, 4, dtype=torch.float32)
        anchor_b = torch.randn(4, dtype=torch.float32)
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        tensors = {
            "param_sam": param_sam,
            "grad_at_w": grad_at_w,
            "grad_at_pert": grad_at_pert,
            "golden_param_sam": golden_param_sam,
            "golden_epsilon_sam": golden_epsilon_sam,
            "slow_p": slow_p,
            "fast_p": fast_p,
            "golden_slow_p": golden_slow_p,
            "golden_fast_p": golden_fast_p,
            "swa_avg": swa_avg,
            "ema_avg": ema_avg,
            "curr_param": curr_param,
            "golden_swa": golden_swa,
            "golden_ema": golden_ema,
            "soup_models": soup_models,
            "golden_soup": golden_soup,
            "anchor_x": anchor_x,
            "anchor_w": anchor_w,
            "anchor_b": anchor_b,
            "golden_anchor_out": golden_anchor_out,
            # Standard Parity keys
            "weight": anchor_w.clone(),
            "bias": anchor_b.clone(),
            "test_input": anchor_x.clone(),
            "test_output": golden_anchor_out.clone(),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "06_sharpness_aware_swad_and_weight_averaging_dynamics.safetensors"
        save_file(tensors, str(out_file))
        print(f"Saved {len(tensors)} tensors to {out_file}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics.safetensors")
    module = SharpnessAwareUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
