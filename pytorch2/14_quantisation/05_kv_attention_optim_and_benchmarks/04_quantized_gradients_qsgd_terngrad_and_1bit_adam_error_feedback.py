"""Quantized Gradients QSGD Terngrad and 1bit Adam Error Feedback — PyTorch 2.x reference implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback
Golden weights: weights/14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback.safetensors

``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def qsgd_quantize(grad: torch.Tensor, num_levels: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantized SGD (QSGD) with L2 norm scaling and uniform coordinate quantization.

    Args:
        grad: Input gradient tensor of shape [rows, cols].
        num_levels: Number of positive quantization levels (e.g. 7 for 3-bit / 8 levels).

    Returns:
        quant_grad: Dequantized reconstructed gradient tensor [rows, cols].
        norm: L2 norm scaling factor per row [rows, 1].
    """
    norm = torch.norm(grad, p=2, dim=-1, keepdim=True)
    xi = torch.abs(grad) / (norm + 1e-8)
    tilde_xi = torch.round(num_levels * xi) / float(num_levels)
    quant_grad = torch.sign(grad) * norm * tilde_xi
    return quant_grad, norm


def terngrad_quantize(grad: torch.Tensor, clip_factor: float = 2.5) -> tuple[torch.Tensor, torch.Tensor]:
    """TernGrad 3-level {-scale, 0, +scale} gradient quantization with outlier clipping.

    Args:
        grad: Input gradient tensor of shape [rows, cols].
        clip_factor: Standard deviation clipping threshold multiplier (default 2.5).

    Returns:
        quant_grad: Reconstructed ternary gradient tensor [rows, cols].
        scale: Max absolute magnitude per row after clipping [rows, 1].
    """
    mean = torch.mean(grad, dim=-1, keepdim=True)
    var = torch.mean((grad - mean) ** 2, dim=-1, keepdim=True)
    std = torch.sqrt(var + 1e-8)
    c = clip_factor * std
    g_clip = torch.clamp(grad, -c, c)
    scale = torch.amax(torch.abs(g_clip), dim=-1, keepdim=True)
    norm_g = torch.abs(g_clip) / (scale + 1e-8)
    quant_grad = torch.sign(g_clip) * (norm_g >= 0.5).to(grad.dtype) * scale
    return quant_grad, scale


def onebit_adam_compress(
    momentum: torch.Tensor, error_buffer: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1-bit Adam momentum compression with Error Feedback (EF).

    Args:
        momentum: Current gradient / momentum tensor [rows, cols].
        error_buffer: Accumulated quantization residual from previous step [rows, cols].

    Returns:
        m_hat: 1-bit compressed and scaled momentum tensor [rows, cols].
        new_error: Updated residual error buffer to accumulate into next step [rows, cols].
        scale: Mean absolute scaling factor per row [rows, 1].
    """
    m_tilde = momentum + error_buffer
    scale = torch.mean(torch.abs(m_tilde), dim=-1, keepdim=True)
    signs = torch.where(m_tilde >= 0, torch.ones_like(m_tilde), -torch.ones_like(m_tilde))
    m_hat = scale * signs
    new_error = m_tilde - m_hat
    return m_hat, new_error, scale


def onebit_adam_update_step(
    param: torch.Tensor,
    m_hat: torch.Tensor,
    v: torch.Tensor,
    lr: float = 1e-3,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> torch.Tensor:
    """Applies a parameter update using 1-bit compressed momentum and local variance."""
    denom = torch.sqrt(v) + eps
    step = m_hat / denom + weight_decay * param
    return param - lr * step


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    grad = torch.randn(16, 32) * 0.05
    param = torch.randn(16, 32) * 0.1
    v = torch.abs(torch.randn(16, 32)) * 0.01 + 1e-5
    error_0 = torch.zeros(16, 32)

    # QSGD
    qsgd_grad, qsgd_norm = qsgd_quantize(grad, num_levels=7)

    # TernGrad
    terngrad_grad, terngrad_scale = terngrad_quantize(grad, clip_factor=2.5)

    # 1-bit Adam step 1
    m_hat_1, error_1, scale_1 = onebit_adam_compress(grad, error_0)

    # 1-bit Adam step 2 (with accumulated error)
    grad_2 = torch.randn(16, 32) * 0.05
    m_hat_2, error_2, scale_2 = onebit_adam_compress(grad_2, error_1)

    # Update param using m_hat_1
    new_param = onebit_adam_update_step(param, m_hat_1, v, lr=1e-3, eps=1e-8, weight_decay=0.01)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "grad": grad,
        "param": param,
        "v": v,
        "error_0": error_0,
        "qsgd_grad": qsgd_grad,
        "qsgd_norm": qsgd_norm,
        "terngrad_grad": terngrad_grad,
        "terngrad_scale": terngrad_scale,
        "adam_m_hat_1": m_hat_1,
        "adam_error_1": error_1,
        "adam_scale_1": scale_1,
        "grad_2": grad_2,
        "adam_m_hat_2": m_hat_2,
        "adam_error_2": error_2,
        "adam_scale_2": scale_2,
        "new_param": new_param,
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
