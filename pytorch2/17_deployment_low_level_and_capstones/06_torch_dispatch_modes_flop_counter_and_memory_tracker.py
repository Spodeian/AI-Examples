"""Torch Dispatch Modes FLOP Counter and Memory Tracker — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker
Golden weights: weights/17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


class FlopAndMemoryTrackerMode(TorchDispatchMode):
    """Intercepts PyTorch aten operations to track exact FLOPs and active allocation bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.flops: int = 0
        self.allocated_bytes: int = 0
        self.nan_detected: bool = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Disable current mode inside dispatch to prevent infinite recursion
        from torch.utils._python_dispatch import _disable_current_modes
        with _disable_current_modes():
            out = func(*args, **kwargs)

            # FLOP tracking
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "mm" in name or "matmul" in name or "addmm" in name:
                if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
                    a, b = args[0], args[1]
                    if a.ndim >= 2 and b.ndim >= 2:
                        m = a.shape[-2]
                        k = a.shape[-1]
                        n = b.shape[-1]
                        batch = a.numel() // (m * k)
                        self.flops += 2 * batch * m * n * k
            elif "add" in name or "sub" in name or "mul" in name or "silu" in name:
                if isinstance(out, torch.Tensor):
                    self.flops += out.numel()

            # Memory tracking
            if isinstance(out, torch.Tensor):
                self.allocated_bytes += out.numel() * out.element_size()
                if torch.isnan(out).any() or torch.isinf(out).any():
                    self.nan_detected = True

        return out


def tracked_mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Evaluates 2-layer MLP with residual connection while tracking FLOPs and allocations."""
    tracker = FlopAndMemoryTrackerMode()
    with tracker:
        z1 = F.silu(x @ w1 + b1)
        z2 = z1 @ w2 + b2
        y = z2 + x
    return y, z1, tracker.flops, tracker.allocated_bytes


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    m, d = 4, 4

    x = torch.randn(m, d)
    w1 = torch.randn(d, d) * 0.2
    b1 = torch.randn(d) * 0.1
    w2 = torch.randn(d, d) * 0.2
    b2 = torch.randn(d) * 0.1

    y, z1, flops, allocated_bytes = tracked_mlp_forward(x, w1, b1, w2, b2)

    linear_w = torch.randn(d, d) * 0.1
    linear_b = torch.randn(d) * 0.1
    test_out = linear_anchor(y, linear_w, linear_b)

    tensors = {
        "x": x,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "z1": z1,
        "y": y,
        "flops": torch.tensor([float(flops)]),
        "allocated_bytes": torch.tensor([float(allocated_bytes)]),
        "linear_w": linear_w,
        "linear_b": linear_b,
        "test_input": y.clone(),
        "weight": linear_w.clone(),
        "bias": linear_b.clone(),
        "test_output": test_out,
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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
