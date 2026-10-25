"""Quantized Optimizer States: 8-bit and 4-bit Adam — PyTorch 2.x reference implementation.

Topic: 14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam
Golden weights: weights/14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam.safetensors

``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def quantize_8bit_first_moment(
    m: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-wise symmetric dynamic INT8 quantization for first momentum m_t."""
    orig_shape = m.shape
    m_flat = m.reshape(-1, block_size)
    scale = torch.clamp(torch.max(torch.abs(m_flat), dim=-1, keepdim=True).values / 127.0, min=1e-7)
    m_int = torch.clamp(torch.round(m_flat / scale), -127.0, 127.0)
    m_deq = m_int * scale
    return m_deq.reshape(orig_shape).clone(), m_int.reshape(orig_shape).clone(), scale.squeeze(-1).clone()


def quantize_8bit_second_moment(
    v: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-wise non-linear log2 quantization for non-negative second momentum v_t."""
    orig_shape = v.shape
    v_flat = v.reshape(-1, block_size)
    v_max = torch.clamp(torch.max(v_flat, dim=-1, keepdim=True).values, min=1e-8)
    v_norm = torch.clamp(v_flat / v_max, min=1e-8)
    v_log = torch.log2(v_norm)

    # Map dynamic range [-24.0, 0.0] into [0, 255]
    v_int = torch.clamp(torch.round((v_log + 24.0) * (255.0 / 24.0)), 0.0, 255.0)
    v_deq_norm = 2.0 ** ((v_int * (24.0 / 255.0)) - 24.0)
    v_deq = v_deq_norm * v_max
    return v_deq.reshape(orig_shape).clone(), v_int.reshape(orig_shape).clone(), v_max.squeeze(-1).clone()


def quantize_4bit_moments(
    m: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """4-bit block-wise optimizer state quantization: signed 4-bit m_t and log 4-bit v_t."""
    orig_shape = m.shape
    # 1. 4-bit signed m_t [-7, 7]
    m_flat = m.reshape(-1, block_size)
    scale_m = torch.clamp(torch.max(torch.abs(m_flat), dim=-1, keepdim=True).values / 7.0, min=1e-7)
    m_int4 = torch.clamp(torch.round(m_flat / scale_m), -7.0, 7.0)
    m_deq4 = m_int4 * scale_m

    # 2. 4-bit log v_t [0, 15] across range [-15.0, 0.0]
    v_flat = v.reshape(-1, block_size)
    v_max = torch.clamp(torch.max(v_flat, dim=-1, keepdim=True).values, min=1e-8)
    v_norm = torch.clamp(v_flat / v_max, min=1e-6)
    v_log = torch.log2(v_norm)
    v_int4 = torch.clamp(torch.round((v_log + 15.0) * (15.0 / 15.0)), 0.0, 15.0)
    v_deq_norm = 2.0 ** ((v_int4 * (15.0 / 15.0)) - 15.0)
    v_deq4 = v_deq_norm * v_max

    return (
        m_deq4.reshape(orig_shape).clone(),
        m_int4.reshape(orig_shape).clone(),
        v_deq4.reshape(orig_shape).clone(),
        v_int4.reshape(orig_shape).clone(),
    )


def adamw_update_step(
    theta: torch.Tensor,
    grad: torch.Tensor,
    m_prev_deq: torch.Tensor,
    v_prev_deq: torch.Tensor,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
    step: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full AdamW parameter update with explicit momentum integration."""
    m_t = beta1 * m_prev_deq + (1.0 - beta1) * grad
    v_t = beta2 * v_prev_deq + (1.0 - beta2) * (grad ** 2)

    # Bias corrections
    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    m_hat = m_t / bc1
    v_hat = v_t / bc2

    # Parameter update
    update = (m_hat / (torch.sqrt(v_hat) + eps)) + weight_decay * theta
    theta_next = theta - lr * update
    return theta_next.clone(), m_t.clone(), v_t.clone()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    d_in, d_out = 16, 16
    block_size = 16

    theta = torch.randn(d_in, d_out) * 0.5
    grad = torch.randn(d_in, d_out) * 0.05
    # Historical moments
    m_prev = torch.randn(d_in, d_out) * 0.02
    v_prev = (torch.randn(d_in, d_out) * 0.01) ** 2 + 1e-6

    # 1. Quantize initial moments to 8-bit
    m_deq8_init, m_int8_init, scale_m8 = quantize_8bit_first_moment(m_prev, block_size=block_size)
    v_deq8_init, v_int8_init, max_v8 = quantize_8bit_second_moment(v_prev, block_size=block_size)

    # 2. Execute 8-bit AdamW step
    theta_next_8bit, m_next8, v_next8 = adamw_update_step(
        theta, grad, m_deq8_init, v_deq8_init, beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=1e-2, step=1
    )
    m_deq8_next, m_int8_next, _ = quantize_8bit_first_moment(m_next8, block_size=block_size)
    v_deq8_next, v_int8_next, _ = quantize_8bit_second_moment(v_next8, block_size=block_size)

    # 3. Quantize initial moments to 4-bit
    m_deq4_init, m_int4_init, v_deq4_init, v_int4_init = quantize_4bit_moments(m_prev, v_prev, block_size=block_size)

    # 4. Execute 4-bit AdamW step
    theta_next_4bit, m_next4, v_next4 = adamw_update_step(
        theta, grad, m_deq4_init, v_deq4_init, beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=1e-2, step=1
    )
    m_deq4_next, m_int4_next, v_deq4_next, v_int4_next = quantize_4bit_moments(m_next4, v_next4, block_size=block_size)

    # 5. FP32 exact unquantized reference step
    theta_next_fp32, m_next_fp32, v_next_fp32 = adamw_update_step(
        theta, grad, m_prev, v_prev, beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=1e-2, step=1
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "theta": theta,
        "grad": grad,
        "m_prev": m_prev,
        "v_prev": v_prev,
        "m_deq8_init": m_deq8_init,
        "m_int8_init": m_int8_init,
        "scale_m8": scale_m8,
        "v_deq8_init": v_deq8_init,
        "v_int8_init": v_int8_init,
        "max_v8": max_v8,
        "theta_next_8bit": theta_next_8bit,
        "m_deq8_next": m_deq8_next,
        "v_deq8_next": v_deq8_next,
        "m_deq4_init": m_deq4_init,
        "m_int4_init": m_int4_init,
        "v_deq4_init": v_deq4_init,
        "v_int4_init": v_int4_init,
        "theta_next_4bit": theta_next_4bit,
        "m_deq4_next": m_deq4_next,
        "v_deq4_next": v_deq4_next,
        "theta_next_fp32": theta_next_fp32,
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
