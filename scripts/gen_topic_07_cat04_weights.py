"""Generate reference weights and golden outputs for Category 04, Topic 07.

Topic 07: Learning Rate Schedulers and LR Finder
(Cosine Annealing with Warmup, WSD, OneCycleLR, AutoClip Adaptive Percentile Clipping)
Enforces transitive numerical parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

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


if __name__ == "__main__":
    main()
