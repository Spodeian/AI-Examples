"""ONNX Export Dynamo and Onnxruntime Inference — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference
Golden weights: weights/17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def onnx_dynamo_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Dynamo-compatible feedforward block with dynamic batch and sequence axes.

    Args:
        x: Input activations [B, S, D].
        w1, b1: First projection parameters [D, H] and [H].
        w2, b2: Second projection parameters [H, D] and [D].

    Returns:
        y: Projected activations [B, S, D].
    """
    h = F.gelu(x @ w1 + b1)
    y = h @ w2 + b2
    return y


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    d_in = 4
    d_hidden = 8
    d_out = 4

    w1 = torch.randn(d_in, d_hidden) * 0.25
    b1 = torch.randn(d_hidden) * 0.1
    w2 = torch.randn(d_hidden, d_out) * 0.25
    b2 = torch.randn(d_out) * 0.1

    # Dynamic Shape 1: [B=1, S=4, D=4]
    x1 = torch.randn(1, 4, d_in)
    y1 = onnx_dynamo_forward(x1, w1, b1, w2, b2)

    # Dynamic Shape 2: [B=2, S=2, D=4]
    x2 = torch.randn(2, 2, d_in)
    y2 = onnx_dynamo_forward(x2, w1, b1, w2, b2)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
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
