"""Torch Compile Modes Backends and Graph Breaks — PyTorch 2.x reference implementation.

Topic: 12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks
Golden weights: weights/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.safetensors

``python pytorch2/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.py`` verifies against the golden checkpoint;
``python pytorch2/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def fused_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation (Shazeer 2020): y = (x * sigmoid(x)) * gate."""
    return (x * torch.sigmoid(x)) * gate


def fused_add_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused residual addition and Layer Normalization (Ba et al. 2016)."""
    h = x + residual
    mean = torch.mean(h, dim=-1, keepdim=True)
    var = torch.mean((h - mean) ** 2, dim=-1, keepdim=True)
    norm = (h - mean) / torch.sqrt(var + float(eps))
    return norm * gamma + beta


def tensor_cond_select(
    x: torch.Tensor,
    threshold: float,
    scale_high: float,
    scale_low: float,
) -> torch.Tensor:
    """Branchless conditional selection (torch.cond / jax.lax.cond equivalent)."""
    return torch.where(x > float(threshold), x * float(scale_high), x * float(scale_low))


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. SwiGLU inputs: B=4, D=6
    swiglu_x = torch.randn(4, 6)
    swiglu_gate = torch.randn(4, 6)
    swiglu_ref = fused_swiglu(swiglu_x, swiglu_gate)

    # 2. Add + LayerNorm inputs: B=4, D=6
    ln_x = torch.randn(4, 6)
    ln_residual = torch.randn(4, 6) * 0.5
    ln_gamma = torch.randn(6) * 0.2 + 1.0
    ln_beta = torch.randn(6) * 0.1
    add_ln_ref = fused_add_layernorm(ln_x, ln_residual, ln_gamma, ln_beta, eps=1e-5)

    # 3. Conditional select inputs: B=4, D=4
    cond_x = torch.randn(4, 4)
    cond_ref = tensor_cond_select(cond_x, threshold=0.0, scale_high=2.0, scale_low=0.5)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "swiglu_x": swiglu_x,
        "swiglu_gate": swiglu_gate,
        "swiglu_ref": swiglu_ref,
        "ln_x": ln_x,
        "ln_residual": ln_residual,
        "ln_gamma": ln_gamma,
        "ln_beta": ln_beta,
        "add_ln_ref": add_ln_ref,
        "cond_x": cond_x,
        "cond_ref": cond_ref,
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
