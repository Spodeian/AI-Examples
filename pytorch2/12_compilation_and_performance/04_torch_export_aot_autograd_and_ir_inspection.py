"""Torch Export AOT Autograd and IR Inspection — PyTorch 2.x reference implementation.

Topic: 12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection
Golden weights: weights/12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection.safetensors

``python pytorch2/12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection.py`` verifies against the golden checkpoint;
``python pytorch2/12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def aten_core_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Decomposed Core ATen operator representation of Softmax."""
    m = torch.amax(x, dim=dim, keepdim=True)
    e = torch.exp(x - m)
    s = torch.sum(e, dim=dim, keepdim=True)
    return e / s


def aten_core_layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Decomposed Core ATen operator representation of LayerNorm."""
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
    rsqrt_var = torch.rsqrt(var + float(eps))
    norm = (x - mean) * rsqrt_var
    return norm * gamma + beta


def aot_forward_backward_affine_gelu(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """AOTAutograd captured forward and reverse-mode adjoint VJP graph for Affine + GeLU:
    z = x @ w + b
    a = 0.5 * z * (1 + erf(z / sqrt(2)))
    """
    z = x @ w + b
    sqrt_2 = 1.4142135623730951
    sqrt_2_pi = 2.5066282746310002
    
    # Forward activation
    a = 0.5 * z * (1.0 + torch.erf(z / sqrt_2))

    # Backward adjoints (VJP graph)
    d_gelu_dz = 0.5 * (1.0 + torch.erf(z / sqrt_2)) + (z / sqrt_2_pi) * torch.exp(-0.5 * z * z)
    gz = grad_out * d_gelu_dz
    grad_w = x.t() @ gz
    grad_b = torch.sum(gz, dim=0)
    grad_x = gz @ w.t()
    return a, grad_w, grad_b, grad_x


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Core ATen Softmax
    sm_x = torch.randn(4, 6)
    sm_ref = aten_core_softmax(sm_x, dim=-1)

    # 2. Core ATen LayerNorm
    ln_x = torch.randn(4, 5)
    ln_gamma = torch.randn(5) * 0.2 + 1.0
    ln_beta = torch.randn(5) * 0.1
    ln_ref = aten_core_layernorm(ln_x, ln_gamma, ln_beta, eps=1e-5)

    # 3. AOT Forward-Backward Affine + GeLU (Batch=3, D_in=4, D_out=5)
    mlp_x = torch.randn(3, 4)
    mlp_w = torch.randn(4, 5) * 0.2
    mlp_b = torch.randn(5) * 0.1
    mlp_grad_out = torch.randn(3, 5)

    fwd_a_ref, grad_w_ref, grad_b_ref, grad_x_ref = aot_forward_backward_affine_gelu(
        mlp_x, mlp_w, mlp_b, mlp_grad_out
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "sm_x": sm_x,
        "sm_ref": sm_ref,
        "ln_x": ln_x,
        "ln_gamma": ln_gamma,
        "ln_beta": ln_beta,
        "ln_ref": ln_ref,
        "mlp_x": mlp_x,
        "mlp_w": mlp_w,
        "mlp_b": mlp_b,
        "mlp_grad_out": mlp_grad_out,
        "fwd_a_ref": fwd_a_ref,
        "grad_w_ref": grad_w_ref,
        "grad_b_ref": grad_b_ref,
        "grad_x_ref": grad_x_ref,
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
