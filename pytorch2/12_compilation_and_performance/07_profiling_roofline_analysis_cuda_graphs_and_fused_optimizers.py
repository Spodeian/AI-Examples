"""Profiling Roofline Analysis CUDA Graphs and Fused Optimizers — PyTorch 2.x reference implementation.

Topic: 12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers
Golden weights: weights/12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers.safetensors

``python pytorch2/12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers.py`` verifies against the golden checkpoint;
``python pytorch2/12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def roofline_metrics(
    flops: float,
    bytes_transferred: float,
    peak_tflops: float = 312.0,  # e.g. A100 Tensor Core TF32 peak
    peak_bandwidth_tbs: float = 2.039,  # A100 HBM2e peak TB/s
) -> tuple[float, float, float]:
    """Evaluates Williams Roofline Model:
    Arithmetic Intensity: I = FLOPs / Bytes
    Ridge Point: I* = Peak_TFLOPs / Peak_Bandwidth_TB_s
    Attainable Performance: P = min(Peak_TFLOPs, I * Peak_Bandwidth_TB_s)
    """
    intensity = float(flops) / max(float(bytes_transferred), 1.0)
    ridge_point = float(peak_tflops) / float(peak_bandwidth_tbs)
    attainable_tflops = min(float(peak_tflops), intensity * float(peak_bandwidth_tbs))
    return intensity, ridge_point, attainable_tflops


def fused_adamw_step(
    w: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    step: int = 1,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused AdamW step combining moment updates, bias correction, weight decay, and step:
    m_t = beta1 * m + (1 - beta1) * g
    v_t = beta2 * v + (1 - beta2) * g^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    w_t = w * (1 - lr * weight_decay) - lr * (m_hat / (sqrt(v_hat) + eps))
    """
    b1 = float(beta1)
    b2 = float(beta2)
    t = float(step)

    m_t = b1 * m + (1.0 - b1) * g
    v_t = b2 * v + (1.0 - b2) * (g * g)

    bias_corr1 = 1.0 - (b1 ** t)
    bias_corr2 = 1.0 - (b2 ** t)

    m_hat = m_t / bias_corr1
    v_hat = v_t / bias_corr2

    # Decoupled weight decay update
    w_decayed = w * (1.0 - float(lr) * float(weight_decay))
    w_t = w_decayed - float(lr) * (m_hat / (torch.sqrt(v_hat) + float(eps)))
    return w_t, m_t, v_t


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Fused AdamW Step: Parameter D_in=4, D_out=5
    adam_w = torch.randn(4, 5) * 0.5
    adam_g = torch.randn(4, 5) * 0.1
    adam_m = torch.randn(4, 5) * 0.02
    adam_v = torch.rand(4, 5) * 0.05 + 0.01

    w_ref, m_ref, v_ref = fused_adamw_step(
        adam_w, adam_g, adam_m, adam_v,
        step=5, lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=1e-2, eps=1e-8
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "adam_w": adam_w,
        "adam_g": adam_g,
        "adam_m": adam_m,
        "adam_v": adam_v,
        "adam_w_ref": w_ref,
        "adam_m_ref": m_ref,
        "adam_v_ref": v_ref,
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
