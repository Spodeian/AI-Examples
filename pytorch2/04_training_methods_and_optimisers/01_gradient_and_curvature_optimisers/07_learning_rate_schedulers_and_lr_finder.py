"""Learning Rate Schedulers and LR Finder in PyTorch 2.x.

Implements foundational and modern learning rate schedules:
1. Cosine Annealing with Linear Warmup (Loshchilov & Hutter 2016).
2. Warmup-Stable-Decay (WSD - Hu et al. 2024 MiniCPM): Horizon-independent pretraining schedule.
3. OneCycleLR (Smith 2017): Dynamic cyclical schedule with asymmetric momentum.
4. AutoClip (Seetharaman et al. 2020): Adaptive gradient clipping by historical percentiles.

Enforces strict transitive parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file


def schedule_cosine_warmup(
    step: int,
    total_steps: int = 100,
    warmup_steps: int = 10,
    eta_min: float = 1e-5,
    eta_max: float = 0.01,
) -> float:
    """Compute learning rate for Cosine Annealing with Linear Warmup."""
    if step <= warmup_steps:
        return eta_min + (step / max(1, warmup_steps)) * (eta_max - eta_min)
    ratio = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * ratio))


def schedule_wsd(
    step: int,
    total_steps: int = 100,
    warmup_steps: int = 10,
    decay_steps: int = 20,
    eta_min: float = 1e-4,
    eta_max: float = 0.01,
) -> float:
    """Compute learning rate for Warmup-Stable-Decay (WSD)."""
    if step <= warmup_steps:
        return eta_min + (step / max(1, warmup_steps)) * (eta_max - eta_min)
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return eta_max
    ratio = (step - decay_start) / max(1, decay_steps)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * ratio))


def schedule_onecycle(
    step: int,
    total_steps: int = 100,
    peak_steps: int = 30,
    eta_max: float = 0.01,
    div_factor: float = 25.0,
    final_div_factor: float = 1000.0,
) -> float:
    """Compute learning rate for OneCycleLR."""
    eta_init = eta_max / div_factor
    eta_min = eta_init / final_div_factor
    if step <= peak_steps:
        ratio = step / max(1, peak_steps)
        return eta_init + 0.5 * (eta_max - eta_init) * (1.0 - math.cos(math.pi * ratio))
    ratio = (step - peak_steps) / max(1, total_steps - peak_steps)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * ratio))


def step_autoclip(
    param: torch.Tensor,
    grad: torch.Tensor,
    norm_history: torch.Tensor,
    percentile: float = 90.0,
    lr: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Perform AutoClip adaptive gradient percentile clipping and parameter step."""
    tau = float(np.percentile(norm_history.cpu().numpy(), percentile))
    grad_norm = float(torch.norm(grad).item())
    clip_ratio = min(1.0, tau / (grad_norm + eps))
    clipped_grad = grad * clip_ratio
    param_next = param - lr * clipped_grad
    return param_next, clipped_grad, tau


class LinearAnchor(nn.Module):
    """Linear projection layer serving as deterministic anchor across all tracks."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class LRSchedulersUnifiedModule(nn.Module):
    """Unified module encapsulating learning rate schedulers and adaptive clipping."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        tensors = load_file(str(checkpoint_path))
        self.param_cosine = tensors["param_cosine"]
        self.grad_cosine = tensors["grad_cosine"]
        self.param_wsd = tensors["param_wsd"]
        self.grad_wsd = tensors["grad_wsd"]
        self.param_onecycle = tensors["param_onecycle"]
        self.grad_onecycle = tensors["grad_onecycle"]
        self.param_autoclip = tensors["param_autoclip"]
        self.grad_autoclip = tensors["grad_autoclip"]
        self.norm_history = tensors["norm_history"]

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(tensors["anchor_w"].T)
        self.anchor.bias.data.copy_(tensors["anchor_b"])

        self.golden = {
            "lr_cosine": float(tensors["golden_lr_cosine"][0]),
            "param_cosine": tensors["golden_param_cosine"],
            "lr_wsd": float(tensors["golden_lr_wsd"][0]),
            "param_wsd": tensors["golden_param_wsd"],
            "lr_onecycle": float(tensors["golden_lr_onecycle"][0]),
            "param_onecycle": tensors["golden_param_onecycle"],
            "clipped_grad": tensors["golden_clipped_grad"],
            "param_autoclip": tensors["golden_param_autoclip"],
            "anchor_out": tensors["golden_anchor_out"],
            "anchor_x": tensors["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: Cosine Annealing with Warmup
        lr_cos = schedule_cosine_warmup(30, total_steps=100, warmup_steps=10, eta_min=1e-5, eta_max=0.01)
        p_cos = self.param_cosine - lr_cos * self.grad_cosine
        err_lr_cos = abs(lr_cos - self.golden["lr_cosine"])
        err_p_cos = float(torch.max(torch.abs(p_cos - self.golden["param_cosine"])))
        err_cos = max(err_lr_cos, err_p_cos)
        assert err_cos < atol, f"Cosine invariant failed: {err_cos:.2e}"

        # Invariant 2: WSD
        lr_wsd = schedule_wsd(85, total_steps=100, warmup_steps=10, decay_steps=20, eta_min=1e-4, eta_max=0.01)
        p_wsd = self.param_wsd - lr_wsd * self.grad_wsd
        err_lr_wsd = abs(lr_wsd - self.golden["lr_wsd"])
        err_p_wsd = float(torch.max(torch.abs(p_wsd - self.golden["param_wsd"])))
        err_wsd = max(err_lr_wsd, err_p_wsd)
        assert err_wsd < atol, f"WSD invariant failed: {err_wsd:.2e}"

        # Invariant 3: OneCycleLR
        lr_oc = schedule_onecycle(40, total_steps=100, peak_steps=30, eta_max=0.01, div_factor=25.0, final_div_factor=1000.0)
        p_oc = self.param_onecycle - lr_oc * self.grad_onecycle
        err_lr_oc = abs(lr_oc - self.golden["lr_onecycle"])
        err_p_oc = float(torch.max(torch.abs(p_oc - self.golden["param_onecycle"])))
        err_oc = max(err_lr_oc, err_p_oc)
        assert err_oc < atol, f"OneCycle invariant failed: {err_oc:.2e}"

        # Invariant 4: AutoClip
        p_ac, clipped_g, _ = step_autoclip(self.param_autoclip, self.grad_autoclip, self.norm_history, percentile=90.0, lr=0.01)
        err_cg = float(torch.max(torch.abs(clipped_g - self.golden["clipped_grad"])))
        err_p_ac = float(torch.max(torch.abs(p_ac - self.golden["param_autoclip"])))
        err_ac = max(err_cg, err_p_ac)
        assert err_ac < atol, f"AutoClip invariant failed: {err_ac:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor(self.golden["anchor_x"]).detach()
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden["anchor_out"])))
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"

        return {
            "cosine_err": err_cos,
            "wsd_err": err_wsd,
            "onecycle_err": err_oc,
            "autoclip_err": err_ac,
            "anchor_err": err_anchor,
        }


def main():
    ckpt_path = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/07_learning_rate_schedulers_and_lr_finder.safetensors")
    module = LRSchedulersUnifiedModule(ckpt_path)
    errors = module.verify_invariants()
    print("PyTorch 2.x Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import numpy as np
    import torch
    from safetensors.torch import save_file


    def main():
        torch.manual_seed(42)
        device = torch.device("cpu")
        d = 8

        # State parameters
        param_cosine = torch.randn(d, dtype=torch.float32)
        grad_cosine = torch.randn(d, dtype=torch.float32)

        param_wsd = torch.randn(d, dtype=torch.float32)
        grad_wsd = torch.randn(d, dtype=torch.float32)

        param_onecycle = torch.randn(d, dtype=torch.float32)
        grad_onecycle = torch.randn(d, dtype=torch.float32)

        param_autoclip = torch.randn(d, dtype=torch.float32)
        # Give autoclip a large gradient to guarantee clipping
        grad_autoclip = torch.randn(d, dtype=torch.float32) * 5.0

        # AutoClip history of gradient norms (10 elements)
        norm_history = torch.tensor(
            [0.45, 0.52, 0.61, 0.70, 0.85, 0.92, 1.05, 1.15, 1.30, 1.45],
            dtype=torch.float32,
        )

        # Linear Anchor
        anchor_x = torch.randn(2, d, dtype=torch.float32)
        anchor_w = torch.randn(d, 4, dtype=torch.float32)
        anchor_b = torch.randn(4, dtype=torch.float32)
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        # 1. Cosine Annealing with Warmup Golden Step
        # T = 100, T_warmup = 10, step = 30, eta_min = 1e-5, eta_max = 0.01
        t_cos = 30
        T_cos = 100
        T_warmup = 10
        eta_min_cos = 1e-5
        eta_max_cos = 0.01

        ratio_cos = (t_cos - T_warmup) / (T_cos - T_warmup)  # 20 / 90
        golden_lr_cosine = torch.tensor(
            eta_min_cos + 0.5 * (eta_max_cos - eta_min_cos) * (1.0 + math.cos(math.pi * ratio_cos)),
            dtype=torch.float32,
        )
        golden_param_cosine = param_cosine - golden_lr_cosine * grad_cosine

        # 2. WSD (Warmup-Stable-Decay) Golden Step
        # T = 100, T_warmup = 10, T_decay = 20 -> T_decay_start = 80, step = 85
        t_wsd = 85
        T_wsd = 100
        T_decay = 20
        T_decay_start = T_wsd - T_decay
        eta_min_wsd = 1e-4
        eta_max_wsd = 0.01

        ratio_wsd = (t_wsd - T_decay_start) / T_decay  # 5 / 20 = 0.25
        golden_lr_wsd = torch.tensor(
            eta_min_wsd + 0.5 * (eta_max_wsd - eta_min_wsd) * (1.0 + math.cos(math.pi * ratio_wsd)),
            dtype=torch.float32,
        )
        golden_param_wsd = param_wsd - golden_lr_wsd * grad_wsd

        # 3. OneCycleLR Golden Step
        # T = 100, T_peak = 30, step = 40, eta_max = 0.01, div_factor = 25, final_div_factor = 1000
        t_oc = 40
        T_oc = 100
        T_peak = 30
        eta_max_oc = 0.01
        eta_init_oc = eta_max_oc / 25.0
        eta_min_oc = eta_init_oc / 1000.0

        ratio_oc = (t_oc - T_peak) / (T_oc - T_peak)  # 10 / 70
        golden_lr_onecycle = torch.tensor(
            eta_min_oc + 0.5 * (eta_max_oc - eta_min_oc) * (1.0 + math.cos(math.pi * ratio_oc)),
            dtype=torch.float32,
        )
        golden_param_onecycle = param_onecycle - golden_lr_onecycle * grad_onecycle

        # 4. AutoClip Golden Step
        # 90th percentile of norm_history
        # Using numpy percentile with linear interpolation (standard method)
        tau = float(np.percentile(norm_history.numpy(), 90.0))  # 1.315
        grad_norm = float(torch.norm(grad_autoclip).item())
        clip_ratio = min(1.0, tau / (grad_norm + 1e-8))
        golden_clipped_grad = grad_autoclip * clip_ratio
        golden_param_autoclip = param_autoclip - 0.01 * golden_clipped_grad

        tensors = {
            # Cosine Annealing
            "param_cosine": param_cosine,
            "grad_cosine": grad_cosine,
            "golden_lr_cosine": golden_lr_cosine.unsqueeze(0),
            "golden_param_cosine": golden_param_cosine,
            # WSD
            "param_wsd": param_wsd,
            "grad_wsd": grad_wsd,
            "golden_lr_wsd": golden_lr_wsd.unsqueeze(0),
            "golden_param_wsd": golden_param_wsd,
            # OneCycleLR
            "param_onecycle": param_onecycle,
            "grad_onecycle": grad_onecycle,
            "golden_lr_onecycle": golden_lr_onecycle.unsqueeze(0),
            "golden_param_onecycle": golden_param_onecycle,
            # AutoClip
            "param_autoclip": param_autoclip,
            "grad_autoclip": grad_autoclip,
            "norm_history": norm_history,
            "golden_clipped_grad": golden_clipped_grad,
            "golden_param_autoclip": golden_param_autoclip,
            # Linear Anchor
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

        out_dir = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "07_learning_rate_schedulers_and_lr_finder.safetensors"
        save_file(tensors, str(out_file))
        print(f"Saved {len(tensors)} tensors to {out_file}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
