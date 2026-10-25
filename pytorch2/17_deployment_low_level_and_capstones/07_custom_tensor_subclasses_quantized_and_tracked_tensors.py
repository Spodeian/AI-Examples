"""Custom Tensor Subclasses Quantized and Tracked Tensors — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors
Golden weights: weights/17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


class DualTensor(torch.Tensor):
    """Wrapper subclass carrying primal value and forward-AD tangent perturbation (x + eps * x_dot)."""

    primal: torch.Tensor
    tangent: torch.Tensor

    @staticmethod
    def __new__(cls, primal: torch.Tensor, tangent: torch.Tensor) -> DualTensor:
        res = torch.Tensor._make_wrapper_subclass(
            cls,
            primal.shape,
            strides=primal.stride(),
            storage_offset=primal.storage_offset(),
            dtype=primal.dtype,
            layout=primal.layout,
            device=primal.device,
            requires_grad=False,
        )
        res.primal = primal
        res.tangent = tangent
        return res

    def __repr__(self) -> str:
        return f"DualTensor(primal={self.primal}, tangent={self.tangent})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Unpack DualTensor arguments
        primal_args = [a.primal if isinstance(a, DualTensor) else a for a in args]
        out_primal = func(*primal_args, **kwargs)

        name = func.__name__ if hasattr(func, "__name__") else str(func)
        if "mm" in name or "matmul" in name:
            a, b = args[0], args[1]
            a_p = a.primal if isinstance(a, DualTensor) else a
            a_t = a.tangent if isinstance(a, DualTensor) else torch.zeros_like(a_p)
            b_p = b.primal if isinstance(b, DualTensor) else b
            b_t = b.tangent if isinstance(b, DualTensor) else torch.zeros_like(b_p)
            out_tangent = (a_p @ b_t) + (a_t @ b_p)
        elif "add" in name:
            a, b = args[0], args[1]
            a_t = a.tangent if isinstance(a, DualTensor) else torch.zeros_like(a)
            b_t = b.tangent if isinstance(b, DualTensor) else torch.zeros_like(b)
            out_tangent = a_t + b_t
        else:
            out_tangent = torch.zeros_like(out_primal)

        return DualTensor(out_primal, out_tangent)


def forward_dual_affine(
    x_primal: torch.Tensor,
    x_tangent: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward affine step computing both primal output and JVP tangent:
    y = x @ w + b
    y_dot = x_dot @ w
    """
    x_dual = DualTensor(x_primal, x_tangent)
    y_dual = x_dual @ w + b
    return y_dual.primal, y_dual.tangent


def dequantize_affine(
    int_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Simulates LazyDequantTensor transparent dequantization: W_fp = (W_int - zp) * scale."""
    return (int_weight.to(torch.float32) - zero_point) * scale


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    m, d = 4, 4

    x_primal = torch.randn(m, d)
    x_tangent = torch.randn(m, d) * 0.1

    # Simulate INT8 quantized weights with scale and zero-point
    int_w = torch.randint(-128, 127, (d, d), dtype=torch.int8)
    scale = torch.tensor([0.005], dtype=torch.float32)
    zero_point = torch.tensor([0.0], dtype=torch.float32)
    w = dequantize_affine(int_w, scale, zero_point)
    b = torch.randn(d) * 0.1

    y_primal, y_tangent = forward_dual_affine(x_primal, x_tangent, w, b)

    linear_w = torch.randn(d, d) * 0.1
    linear_b = torch.randn(d) * 0.1
    test_out = linear_anchor(y_primal, linear_w, linear_b)

    tensors = {
        "x_primal": x_primal,
        "x_tangent": x_tangent,
        "int_w": int_w.to(torch.float32),
        "scale": scale,
        "zero_point": zero_point,
        "w": w,
        "b": b,
        "y_primal": y_primal,
        "y_tangent": y_tangent,
        "linear_w": linear_w,
        "linear_b": linear_b,
        "test_input": y_primal.clone(),
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
