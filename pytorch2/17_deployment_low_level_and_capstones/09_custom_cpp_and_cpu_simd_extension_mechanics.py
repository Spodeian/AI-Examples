"""Custom C++ and CPU SIMD Extension Mechanics — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics
Golden weights: weights/17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def fused_simd_fma_leaky_relu(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    negative_slope: float = 0.1,
) -> torch.Tensor:
    """Simulates AVX2/AVX-512 vector FMA (y = x * w + b) followed by LeakyReLU.
    
    Guarantees contiguous memory access layout required by C++ data pointers.
    """
    x_c = x.contiguous()
    w_c = w.contiguous()
    b_c = b.contiguous()

    z = x_c * w_c + b_c
    return F.leaky_relu(z, negative_slope=negative_slope)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    m, d = 4, 4

    x = torch.randn(m, d)
    w = torch.randn(m, d) * 0.2
    b = torch.randn(m, d) * 0.1

    y = fused_simd_fma_leaky_relu(x, w, b, negative_slope=0.1)

    linear_w = torch.randn(d, d) * 0.1
    linear_b = torch.randn(d) * 0.1
    test_out = linear_anchor(y, linear_w, linear_b)

    tensors = {
        "x": x,
        "w": w,
        "b": b,
        "y": y,
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
