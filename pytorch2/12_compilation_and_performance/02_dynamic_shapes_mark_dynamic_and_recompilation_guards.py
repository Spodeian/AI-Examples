"""Dynamic Shapes Mark Dynamic and Recompilation Guards — PyTorch 2.x reference implementation.

Topic: 12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards
Golden weights: weights/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.safetensors

``python pytorch2/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.py`` verifies against the golden checkpoint;
``python pytorch2/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dynamic_masked_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling across variable sequence lengths."""
    mask_expanded = mask.unsqueeze(-1)
    sum_x = torch.sum(x * mask_expanded, dim=1)
    lengths = torch.sum(mask, dim=-1, keepdim=True)
    return sum_x / torch.clamp(lengths, min=1.0)


def bucket_allocation(seq_lens: torch.Tensor, bucket_boundaries: torch.Tensor) -> torch.Tensor:
    """Assign sequence lengths to the smallest enclosing bucket boundary."""
    diffs = bucket_boundaries.unsqueeze(0) - seq_lens.unsqueeze(1)
    valid = diffs >= 0.0
    cost = torch.where(valid, diffs, torch.tensor(float("inf")))
    idx = torch.argmin(cost, dim=-1)
    return bucket_boundaries[idx]


def dynamic_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Root Mean Square Normalization (Zhang & Sennrich 2019) across dynamic features."""
    variance = torch.mean(x ** 2, dim=-1, keepdim=True)
    return (x / torch.sqrt(variance + float(eps))) * weight


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Dynamic masked pool: B=3, L=5, D=4
    pool_x = torch.randn(3, 5, 4)
    pool_mask = torch.tensor([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    masked_pool_ref = dynamic_masked_pool(pool_x, pool_mask)

    # 2. Bucket allocation: N=6 lengths, buckets = [8, 16, 32, 64]
    seq_lens = torch.tensor([5.0, 12.0, 16.0, 17.0, 31.0, 48.0])
    buckets = torch.tensor([8.0, 16.0, 32.0, 64.0])
    allocated_buckets_ref = bucket_allocation(seq_lens, buckets)

    # 3. Dynamic RMSNorm: B=4, D=6
    rms_x = torch.randn(4, 6)
    rms_w = torch.randn(6) * 0.2 + 1.0
    dynamic_rms_ref = dynamic_rms_norm(rms_x, rms_w, eps=1e-5)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "pool_x": pool_x,
        "pool_mask": pool_mask,
        "masked_pool_ref": masked_pool_ref,
        "seq_lens": seq_lens,
        "buckets": buckets,
        "allocated_buckets_ref": allocated_buckets_ref,
        "rms_x": rms_x,
        "rms_w": rms_w,
        "dynamic_rms_ref": dynamic_rms_ref,
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
