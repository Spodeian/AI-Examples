"""Pipeline Parallelism Gpipe 1f1b Zero Bubble and Dualpipe — PyTorch 2.x reference implementation.

Topic: 16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe
Golden weights: weights/16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe.safetensors

``python pytorch2/16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe.py`` verifies against the golden checkpoint;
``python pytorch2/16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def compute_pipeline_bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """Calculates standard GPipe/1F1B bubble fraction: (P - 1) / (M + P - 1)."""
    return (num_stages - 1) / (num_microbatches + num_stages - 1)


def pipeline_forward_step(
    x_mb: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-stage pipeline forward evaluation across microbatches.

    Args:
        x_mb: Microbatch inputs [M, B, D].
        w0, b0: Stage 0 parameters.
        w1, b1: Stage 1 parameters.

    Returns:
        h_mb: Stage 0 intermediate activations [M, B, D].
        y_mb: Stage 1 outputs [M, B, D].
    """
    m = x_mb.shape[0]
    h_list = []
    y_list = []

    for i in range(m):
        h_i = x_mb[i] @ w0 + b0
        y_i = h_i @ w1 + b1
        h_list.append(h_i)
        y_list.append(y_i)

    h_mb = torch.stack(h_list, dim=0)
    y_mb = torch.stack(y_list, dim=0)
    return h_mb, y_mb


def dualpipe_backward_step(
    x_mb: torch.Tensor,
    h_mb: torch.Tensor,
    grad_y_mb: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DualPipe / Zero-Bubble split backward computation: B_A (activation) and B_W (weight).

    Args:
        x_mb: Inputs [M, B, D].
        h_mb: Stage 0 intermediate activations [M, B, D].
        grad_y_mb: Stage 1 output gradients [M, B, D].
        w0, w1: Stage weights.

    Returns:
        grad_w0: Accumulated weight gradient for stage 0 [D, D].
        grad_w1: Accumulated weight gradient for stage 1 [D, D].
        grad_x_mb: Input gradients for stage 0 [M, B, D].
    """
    m = x_mb.shape[0]
    grad_w0 = torch.zeros_like(w0)
    grad_w1 = torch.zeros_like(w1)
    grad_x_list = []

    for i in range(m):
        # Stage 1: B_A (activation grad transmitted upstream) and B_W (weight grad accumulated)
        b_a_1 = grad_y_mb[i] @ w1.t()  # [B, D]
        b_w_1 = h_mb[i].t() @ grad_y_mb[i]  # [D, D]
        grad_w1 += b_w_1

        # Stage 0: B_A (input grad) and B_W (weight grad accumulated)
        b_a_0 = b_a_1 @ w0.t()  # [B, D]
        b_w_0 = x_mb[i].t() @ b_a_1  # [D, D]
        grad_w0 += b_w_0

        grad_x_list.append(b_a_0)

    grad_x_mb = torch.stack(grad_x_list, dim=0)
    return grad_w0, grad_w1, grad_x_mb


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    num_microbatches = 4
    batch_size = 2
    d_model = 4

    x_mb = torch.randn(num_microbatches, batch_size, d_model)
    grad_y_mb = torch.randn(num_microbatches, batch_size, d_model) * 0.2

    w0 = torch.randn(d_model, d_model) * 0.25
    b0 = torch.randn(d_model) * 0.1
    w1 = torch.randn(d_model, d_model) * 0.25
    b1 = torch.randn(d_model) * 0.1

    h_mb, y_mb = pipeline_forward_step(x_mb, w0, b0, w1, b1)
    grad_w0, grad_w1, grad_x_mb = dualpipe_backward_step(x_mb, h_mb, grad_y_mb, w0, w1)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_mb": x_mb,
        "grad_y_mb": grad_y_mb,
        "w0": w0,
        "b0": b0,
        "w1": w1,
        "b1": b1,
        "stage0_fwd_outs": h_mb,
        "stage1_fwd_outs": y_mb,
        "grad_w0": grad_w0,
        "grad_w1": grad_w1,
        "grad_x_outs": grad_x_mb,
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
