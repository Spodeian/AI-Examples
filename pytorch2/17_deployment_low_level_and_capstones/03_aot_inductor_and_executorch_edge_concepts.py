"""AOT Inductor and Executorch Edge Concepts — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts
Golden weights: weights/17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def aot_edge_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edge operator composition: Linear -> LayerNorm -> SiLU.

    Args:
        x: Input activations [B, D].
        w, b: Linear weights and bias [D, D] and [D].
        gamma, beta: LayerNorm scale and shift parameters [D] and [D].

    Returns:
        y: Final activated output [B, D].
        z: Pre-norm linear activation [B, D].
        hat_z: Normalized activation [B, D].
    """
    z = x @ w + b
    # LayerNorm over feature dimension
    mu = torch.mean(z, dim=-1, keepdim=True)
    var = torch.var(z, dim=-1, unbiased=False, keepdim=True)
    hat_z = ((z - mu) / torch.sqrt(var + 1e-5)) * gamma + beta
    y = F.silu(hat_z)
    return y, z, hat_z


def plan_static_memory_arena(
    tensor_bytes: list[int],
    reuse_pairs: list[tuple[int, int]],
) -> tuple[list[int], int]:
    """Simulates AOTInductor / ExecuTorch static memory arena planning with buffer reuse.

    Args:
        tensor_bytes: List of required byte sizes for each buffer.
        reuse_pairs: Pairs (a, b) where buffer b reuses the memory allocated for buffer a.

    Returns:
        offsets: Byte offset for each buffer inside the static arena.
        peak_bytes: Total allocated arena size in bytes.
    """
    offsets = [0] * len(tensor_bytes)
    current_offset = 0
    reused = {}

    for src, dst in reuse_pairs:
        reused[dst] = src

    for i, size in enumerate(tensor_bytes):
        if i in reused:
            offsets[i] = offsets[reused[i]]
        else:
            offsets[i] = current_offset
            current_offset += size

    return offsets, current_offset


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    batch_size = 4
    d_model = 4

    x = torch.randn(batch_size, d_model)
    w = torch.randn(d_model, d_model) * 0.25
    b = torch.randn(d_model) * 0.1
    gamma = torch.ones(d_model)
    beta = torch.zeros(d_model)

    y, z, hat_z = aot_edge_forward(x, w, b, gamma, beta)

    # 4 buffers of size 4*4*4 = 64 bytes each; buffer 3 reuses buffer 1
    buffer_sizes = [64, 64, 64, 64]
    offsets, peak_bytes = plan_static_memory_arena(buffer_sizes, reuse_pairs=[(1, 3)])
    arena_offsets = torch.tensor(offsets, dtype=torch.int64)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "w": w,
        "b": b,
        "gamma": gamma,
        "beta": beta,
        "y_out": y,
        "linear_z": z,
        "norm_z": hat_z,
        "arena_offsets": arena_offsets,
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
