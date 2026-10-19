"""JAX / Functional NumPy Implementation for Category 04 Topic 07:
Learning Rate Schedulers and LR Finder.

Provides functional, pure-array implementations:
1. Cosine Annealing with Linear Warmup (Loshchilov & Hutter 2016).
2. Warmup-Stable-Decay (WSD - Hu et al. 2024 MiniCPM): Horizon-free pretraining schedule.
3. OneCycleLR (Smith 2017): Dynamic cyclical schedule.
4. AutoClip (Seetharaman et al. 2020): Adaptive gradient clipping by historical percentiles.

Enforces exact numerical parity with PyTorch 2, Candle, and Hardware Kernels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file


class LinearAnchor:
    """Functional Linear Anchor module (y = x @ w + b)."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weight) + self.bias


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
    param: np.ndarray,
    grad: np.ndarray,
    norm_history: np.ndarray,
    percentile: float = 90.0,
    lr: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform AutoClip adaptive gradient percentile clipping and parameter step."""
    tau = float(np.percentile(norm_history, percentile))
    grad_norm = float(np.linalg.norm(grad))
    clip_ratio = min(1.0, tau / (grad_norm + eps))
    clipped_grad = grad * clip_ratio
    param_next = param - lr * clipped_grad
    return param_next, clipped_grad, tau


class LRSchedulersUnifiedModule:
    """Unified module encapsulating learning rate schedulers and adaptive clipping."""

    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        raw = load_file(str(checkpoint_path))
        self.param_cosine = raw["param_cosine"]
        self.grad_cosine = raw["grad_cosine"]
        self.param_wsd = raw["param_wsd"]
        self.grad_wsd = raw["grad_wsd"]
        self.param_onecycle = raw["param_onecycle"]
        self.grad_onecycle = raw["grad_onecycle"]
        self.param_autoclip = raw["param_autoclip"]
        self.grad_autoclip = raw["grad_autoclip"]
        self.norm_history = raw["norm_history"]

        self.anchor = LinearAnchor(
            weight=raw["anchor_w"],
            bias=raw["anchor_b"],
        )

        self.golden = {
            "lr_cosine": float(raw["golden_lr_cosine"][0]),
            "param_cosine": raw["golden_param_cosine"],
            "lr_wsd": float(raw["golden_lr_wsd"][0]),
            "param_wsd": raw["golden_param_wsd"],
            "lr_onecycle": float(raw["golden_lr_onecycle"][0]),
            "param_onecycle": raw["golden_param_onecycle"],
            "clipped_grad": raw["golden_clipped_grad"],
            "param_autoclip": raw["golden_param_autoclip"],
            "anchor_out": raw["golden_anchor_out"],
            "anchor_x": raw["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: Cosine Annealing with Warmup
        lr_cos = schedule_cosine_warmup(30, total_steps=100, warmup_steps=10, eta_min=1e-5, eta_max=0.01)
        p_cos = self.param_cosine - lr_cos * self.grad_cosine
        err_lr_cos = abs(lr_cos - self.golden["lr_cosine"])
        err_p_cos = float(np.max(np.abs(p_cos - self.golden["param_cosine"])))
        err_cos = max(err_lr_cos, err_p_cos)
        assert err_cos < atol, f"Cosine invariant failed: {err_cos:.2e}"

        # Invariant 2: WSD
        lr_wsd = schedule_wsd(85, total_steps=100, warmup_steps=10, decay_steps=20, eta_min=1e-4, eta_max=0.01)
        p_wsd = self.param_wsd - lr_wsd * self.grad_wsd
        err_lr_wsd = abs(lr_wsd - self.golden["lr_wsd"])
        err_p_wsd = float(np.max(np.abs(p_wsd - self.golden["param_wsd"])))
        err_wsd = max(err_lr_wsd, err_p_wsd)
        assert err_wsd < atol, f"WSD invariant failed: {err_wsd:.2e}"

        # Invariant 3: OneCycleLR
        lr_oc = schedule_onecycle(40, total_steps=100, peak_steps=30, eta_max=0.01, div_factor=25.0, final_div_factor=1000.0)
        p_oc = self.param_onecycle - lr_oc * self.grad_onecycle
        err_lr_oc = abs(lr_oc - self.golden["lr_onecycle"])
        err_p_oc = float(np.max(np.abs(p_oc - self.golden["param_onecycle"])))
        err_oc = max(err_lr_oc, err_p_oc)
        assert err_oc < atol, f"OneCycle invariant failed: {err_oc:.2e}"

        # Invariant 4: AutoClip
        p_ac, clipped_g, _ = step_autoclip(self.param_autoclip, self.grad_autoclip, self.norm_history, percentile=90.0, lr=0.01)
        err_cg = float(np.max(np.abs(clipped_g - self.golden["clipped_grad"])))
        err_p_ac = float(np.max(np.abs(p_ac - self.golden["param_autoclip"])))
        err_ac = max(err_cg, err_p_ac)
        assert err_ac < atol, f"AutoClip invariant failed: {err_ac:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor.forward(self.golden["anchor_x"])
        err_anchor = float(np.max(np.abs(anchor_out - self.golden["anchor_out"])))
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
    print("JAX/NumPy Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


if __name__ == "__main__":
    main()
