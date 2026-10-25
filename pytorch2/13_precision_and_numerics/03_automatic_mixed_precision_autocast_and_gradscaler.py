"""Automatic Mixed Precision Autocast and Gradscaler — PyTorch 2.x reference implementation.

Topic: 13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler
Golden weights: weights/13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler.safetensors

``python pytorch2/13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler.py`` verifies against the golden checkpoint;
``python pytorch2/13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dynamic_grad_scaler_step(
    grads: torch.Tensor,
    scale: float,
    growth_tracker: int,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 4,
) -> tuple[torch.Tensor, float, int, bool]:
    """Simulates PyTorch torch.cuda.amp.GradScaler dynamic scale update logic."""
    has_inf_nan = bool(torch.isinf(grads).any() or torch.isnan(grads).any())
    unscaled = grads * (1.0 / float(scale))

    if has_inf_nan:
        new_scale = float(scale) * float(backoff_factor)
        new_tracker = 0
        step_skipped = True
    else:
        new_tracker = int(growth_tracker) + 1
        if new_tracker >= int(growth_interval):
            new_scale = float(scale) * float(growth_factor)
            new_tracker = 0
        else:
            new_scale = float(scale)
        step_skipped = False

    return unscaled, new_scale, new_tracker, step_skipped


def amp_autocast_matmul_softmax(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Simulates torch.amp.autocast policy:
    GEMM executes in FP16, Softmax executes in FP32.
    """
    x_half = x.half()
    w_half = w.half()
    b_half = b.half()
    z_half = x_half @ w_half + b_half

    z_fp32 = z_half.float()
    probs = torch.softmax(z_fp32, dim=-1)
    return probs


def unscale_and_clip_grads(
    scaled_grads: torch.Tensor,
    scale: float,
    max_norm: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Unscales scaled gradients before computing L2 norm and applying clipping."""
    unscaled = scaled_grads * (1.0 / float(scale))
    total_norm = float(torch.norm(unscaled))
    coeff = min(1.0, float(max_norm) / (total_norm + float(eps)))
    clipped = unscaled * coeff
    return clipped, total_norm


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. AMP Autocast Matmul (FP16) + Softmax (FP32)
    amp_x = torch.randn(4, 4)
    amp_w = torch.randn(4, 5) * 0.2
    amp_b = torch.randn(5) * 0.1
    amp_probs_ref = amp_autocast_matmul_softmax(amp_x, amp_w, amp_b)

    # 2. Dynamic GradScaler: Clean step (scaling up)
    clean_grads = torch.randn(4, 5) * 65.536  # Scaled by 65536
    scale_in = 65536.0
    unscaled_clean_ref, scale_clean_ref, tracker_clean_ref, skipped_clean_ref = dynamic_grad_scaler_step(
        clean_grads, scale=scale_in, growth_tracker=3, growth_factor=2.0, backoff_factor=0.5, growth_interval=4
    )

    # 3. Dynamic GradScaler: Overflow step (scaling down)
    overflow_grads = clean_grads.clone()
    overflow_grads[1, 2] = float("inf")
    unscaled_over_ref, scale_over_ref, tracker_over_ref, skipped_over_ref = dynamic_grad_scaler_step(
        overflow_grads, scale=scale_in, growth_tracker=2, growth_factor=2.0, backoff_factor=0.5, growth_interval=4
    )

    # 4. Unscale and Clip Gradients
    clip_scaled_grads = torch.randn(4, 5) * 2000.0
    clipped_ref, norm_ref = unscale_and_clip_grads(clip_scaled_grads, scale=1000.0, max_norm=1.0)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "amp_x": amp_x,
        "amp_w": amp_w,
        "amp_b": amp_b,
        "amp_probs_ref": amp_probs_ref,
        "clean_grads": clean_grads,
        "unscaled_clean_ref": unscaled_clean_ref,
        "scale_clean_ref": torch.tensor(scale_clean_ref),
        "tracker_clean_ref": torch.tensor(tracker_clean_ref),
        "skipped_clean_ref": torch.tensor(1.0 if skipped_clean_ref else 0.0),
        "unscaled_over_ref": unscaled_over_ref,
        "scale_over_ref": torch.tensor(scale_over_ref),
        "tracker_over_ref": torch.tensor(tracker_over_ref),
        "skipped_over_ref": torch.tensor(1.0 if skipped_over_ref else 0.0),
        "clip_scaled_grads": clip_scaled_grads,
        "clipped_ref": clipped_ref,
        "norm_ref": torch.tensor(norm_ref),
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
