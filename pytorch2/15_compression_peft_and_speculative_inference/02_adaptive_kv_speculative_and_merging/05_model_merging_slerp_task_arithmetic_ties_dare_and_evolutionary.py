"""Model Merging Slerp Task Arithmetic TIES DARE and Evolutionary — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary
Golden weights: weights/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def slerp_merge(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Spherical Linear Interpolation (SLERP) between weight tensors.

    Preserves directional magnitude and curvature on the hyper-spherical parameter manifold.

    Args:
        v0: First parameter vector of arbitrary shape.
        v1: Second parameter vector of matching shape.
        t: Interpolation factor in [0, 1].
        eps: Threshold for small angles to fall back to linear interpolation.

    Returns:
        v_slerp: Interpolated parameter tensor matching input shape.
    """
    orig_shape = v0.shape
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()

    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)

    if norm0 < eps or norm1 < eps:
        return (1.0 - t) * v0 + t * v1

    u0 = v0_flat / norm0
    u1 = v1_flat / norm1

    dot = torch.sum(u0 * u1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)

    if torch.sin(theta) < eps or torch.abs(theta) < eps:
        return (1.0 - t) * v0 + t * v1

    sin_theta = torch.sin(theta)
    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta

    res_flat = w0 * v0_flat + w1 * v1_flat
    return res_flat.reshape(orig_shape)


def task_arithmetic_merge(
    base: torch.Tensor,
    deltas: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Task arithmetic parameter addition: theta = theta_base + sum_k w_k * delta_k.

    Args:
        base: Pretrained base weights of shape [dim1, dim2].
        deltas: Task vectors of shape [num_tasks, dim1, dim2].
        weights: Weight coefficients of shape [num_tasks].

    Returns:
        merged: Combined weight matrix of shape [dim1, dim2].
    """
    num_tasks = deltas.shape[0]
    merged = base.clone()
    for k in range(num_tasks):
        merged = merged + weights[k] * deltas[k]
    return merged


def ties_merge(
    base: torch.Tensor,
    deltas: torch.Tensor,
    trim_ratio: float = 0.5,
    scale: float = 0.5,
) -> torch.Tensor:
    """TIES-Merging (Trimming, Electing Sign, and Disjoint Merge).

    Resolves parameter interference across multiple task vectors:
    1. Trim: Keep top (1 - trim_ratio) by magnitude per task vector, zero the rest.
    2. Elect Sign: Compute majority sign consensus gamma_j = sgn(sum_m delta_{m, j}).
    3. Disjoint Merge: Average only components whose signs agree with the elected sign.

    Args:
        base: Pretrained base weights of shape [dim1, dim2].
        deltas: Task delta vectors of shape [num_tasks, dim1, dim2].
        trim_ratio: Fraction of smallest elements to zero out (e.g. 0.5 keeps top 50%).
        scale: Scaling coefficient lambda applied to consensus delta.

    Returns:
        merged: TIES merged weight tensor of shape [dim1, dim2].
    """
    num_tasks, dim1, dim2 = deltas.shape
    total_elements = dim1 * dim2
    keep_k = max(1, int(round(total_elements * (1.0 - trim_ratio))))

    # 1. Trimming step
    trimmed_deltas = torch.zeros_like(deltas)
    for m in range(num_tasks):
        d_flat = deltas[m].flatten()
        abs_d = torch.abs(d_flat)
        _, top_idx = torch.topk(abs_d, k=keep_k, largest=True)
        trimmed_flat = torch.zeros_like(d_flat)
        trimmed_flat[top_idx] = d_flat[top_idx]
        trimmed_deltas[m] = trimmed_flat.reshape(dim1, dim2)

    # 2. Elect sign
    sum_deltas = torch.sum(trimmed_deltas, dim=0)
    elected_sign = torch.sign(sum_deltas)

    # 3. Disjoint merge: filter and average
    merged_delta = torch.zeros((dim1, dim2), dtype=deltas.dtype, device=deltas.device)
    counts = torch.zeros((dim1, dim2), dtype=deltas.dtype, device=deltas.device)

    for m in range(num_tasks):
        agree_mask = (torch.sign(trimmed_deltas[m]) == elected_sign) & (trimmed_deltas[m] != 0.0)
        merged_delta += torch.where(agree_mask, trimmed_deltas[m], torch.zeros_like(trimmed_deltas[m]))
        counts += agree_mask.to(deltas.dtype)

    avg_delta = torch.where(counts > 0, merged_delta / counts, torch.zeros_like(merged_delta))
    return base + scale * avg_delta


def dare_rescale(
    delta: torch.Tensor,
    mask: torch.Tensor,
    drop_prob: float = 0.5,
) -> torch.Tensor:
    """DARE (Drop And REscale) delta transformation.

    Applies dropout mask and rescales surviving elements by 1 / (1 - p).

    Args:
        delta: Task vector of shape [dim1, dim2].
        mask: Deterministic binary retention mask of shape [dim1, dim2] (1 = retain, 0 = drop).
        drop_prob: Dropout probability p in [0, 1).

    Returns:
        rescaled_delta: Sparsified and rescaled task vector of shape [dim1, dim2].
    """
    rescale_factor = 1.0 / (1.0 - drop_prob)
    return delta * mask * rescale_factor


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. SLERP
    v0 = torch.randn(16) * 0.5
    v1 = torch.randn(16) * 0.5
    slerp_out = slerp_merge(v0, v1, t=0.3)

    # 2. Task Arithmetic
    dim1, dim2 = 8, 8
    num_tasks = 3
    base_w = torch.randn(dim1, dim2) * 0.2
    deltas = torch.randn(num_tasks, dim1, dim2) * 0.1
    task_weights = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)
    arithmetic_out = task_arithmetic_merge(base_w, deltas, task_weights)

    # 3. TIES Merging
    ties_out = ties_merge(base_w, deltas, trim_ratio=0.5, scale=0.6)

    # 4. DARE
    dare_mask = (torch.rand(dim1, dim2) > 0.5).to(torch.float32)
    dare_out = dare_rescale(deltas[0], dare_mask, drop_prob=0.5)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "v0": v0,
        "v1": v1,
        "slerp_out": slerp_out,
        "base_w": base_w,
        "deltas": deltas,
        "task_weights": task_weights,
        "arithmetic_out": arithmetic_out,
        "ties_out": ties_out,
        "dare_mask": dare_mask,
        "dare_out": dare_out,
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
        assert err < 1e-5, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
