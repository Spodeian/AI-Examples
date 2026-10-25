"""Torch Fx Graph Capture and Custom Compiler Passes — PyTorch 2.x reference implementation.

Topic: 12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes
Golden weights: weights/12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes.safetensors

``python pytorch2/12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes.py`` verifies against the golden checkpoint;
``python pytorch2/12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def algebraic_linear_bn_folding(
    w: torch.Tensor,
    b: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algebraic folding of BatchNorm into Linear/Conv parameters:
    
    W_fold = W * (gamma / sqrt(var + eps))
    b_fold = (b - mean) * (gamma / sqrt(var + eps)) + beta
    """
    scale = gamma / torch.sqrt(running_var + float(eps))
    w_fold = w * scale.unsqueeze(0)
    b_fold = (b - running_mean) * scale + beta
    return w_fold, b_fold


def sequential_linear_bn(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Unfolded sequential Linear -> BatchNorm forward pass."""
    z = x @ w + b
    return ((z - running_mean) / torch.sqrt(running_var + float(eps))) * gamma + beta


def parallel_gemm_fusion(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal GEMM fusion pass (e.g. Q, K, V projection concatenation)."""
    w_fused = torch.cat([w1, w2, w3], dim=-1)
    b_fused = torch.cat([b1, b2, b3], dim=-1)
    return w_fused, b_fused


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Linear-BatchNorm folding: D_in=4, D_out=5
    lin_x = torch.randn(4, 4)
    lin_w = torch.randn(4, 5) * 0.3
    lin_b = torch.randn(5) * 0.1
    bn_mean = torch.randn(5) * 0.2
    bn_var = torch.rand(5) + 0.5
    bn_gamma = torch.randn(5) * 0.2 + 1.0
    bn_beta = torch.randn(5) * 0.1

    w_fold_ref, b_fold_ref = algebraic_linear_bn_folding(
        lin_w, lin_b, bn_mean, bn_var, bn_gamma, bn_beta, eps=1e-5
    )
    folded_out_ref = lin_x @ w_fold_ref + b_fold_ref

    # 2. Horizontal parallel GEMM fusion: D_in=4, D_q=3, D_k=3, D_v=3
    w1 = torch.randn(4, 3) * 0.2
    b1 = torch.randn(3) * 0.1
    w2 = torch.randn(4, 3) * 0.2
    b2 = torch.randn(3) * 0.1
    w3 = torch.randn(4, 3) * 0.2
    b3 = torch.randn(3) * 0.1

    w_fused_ref, b_fused_ref = parallel_gemm_fusion(w1, b1, w2, b2, w3, b3)
    fused_out_ref = lin_x @ w_fused_ref + b_fused_ref

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "lin_x": lin_x,
        "lin_w": lin_w,
        "lin_b": lin_b,
        "bn_mean": bn_mean,
        "bn_var": bn_var,
        "bn_gamma": bn_gamma,
        "bn_beta": bn_beta,
        "w_fold_ref": w_fold_ref,
        "b_fold_ref": b_fold_ref,
        "folded_out_ref": folded_out_ref,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3,
        "w_fused_ref": w_fused_ref,
        "b_fused_ref": b_fused_ref,
        "fused_out_ref": fused_out_ref,
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
