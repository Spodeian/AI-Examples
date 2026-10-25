"""QLoRA, LoftQ, and QA-LoRA Quantized PEFT — functional JAX implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft.
Run ``python jax/14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft.py``
to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft.safetensors"

# Canonical QLoRA NF4 16-level standard codebook (Dettmers et al. 2023)
NF4_LEVELS = jnp.array([
    -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0,
])


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def nf4_quantize(
    x: jnp.ndarray,
    block_size: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Quantize tensor x to 4-bit NormalFloat (NF4) with per-block scale normalization."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, block_size)

    abs_max = jnp.max(jnp.abs(x_flat), axis=-1, keepdims=True)
    scale = jnp.clip(abs_max, a_min=1e-7)

    norm_x = x_flat / scale
    diff = jnp.abs(jnp.expand_dims(norm_x, -1) - NF4_LEVELS)
    idx = jnp.argmin(diff, axis=-1)

    deq = NF4_LEVELS[idx] * scale
    return deq.reshape(orig_shape), idx.reshape(orig_shape), jnp.squeeze(scale, -1)


def double_quantize(
    c1: jnp.ndarray,
    dq_block_size: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """QLoRA Double Quantization: quantize FP32 scales c1 to 8-bit integers."""
    c1_flat = c1.reshape(-1, dq_block_size)
    c2 = jnp.clip(jnp.max(c1_flat, axis=-1, keepdims=True) / 127.0, a_min=1e-7)
    q2 = jnp.clip(jnp.round(c1_flat / c2), -127.0, 127.0)
    c1_deq = q2 * c2
    return c1_deq.reshape(c1.shape), q2.reshape(c1.shape), jnp.squeeze(c2, -1)


def qlora_forward(
    x: jnp.ndarray,
    w_deq: jnp.ndarray,
    lora_a: jnp.ndarray,
    lora_b: jnp.ndarray,
    scaling: float,
    bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """QLoRA forward pass: Base NF4 frozen GEMM + low-rank LoRA adapter."""
    base_out = x @ w_deq
    lora_out = (x @ lora_a @ lora_b) * scaling
    y = base_out + lora_out
    if bias is not None:
        y = y + bias
    return y


def qa_lora_forward(
    x: jnp.ndarray,
    w_deq: jnp.ndarray,
    lora_a: jnp.ndarray,
    lora_b: jnp.ndarray,
    group_scales: jnp.ndarray,
    scaling: float,
    group_size: int = 4,
    bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """QA-LoRA: group-wise quantization-aware low-rank adaptation."""
    g_expanded = jnp.repeat(group_scales, group_size, axis=0)
    a_qa = lora_a * g_expanded
    base_out = x @ w_deq
    lora_out = (x @ a_qa @ lora_b) * scaling
    y = base_out + lora_out
    if bias is not None:
        y = y + bias
    return y, a_qa


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor parity
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. NF4 quantization parity
    w_nf4_deq, nf4_indices, c1 = nf4_quantize(g["w_orig"], block_size=8)
    err = float(jnp.max(jnp.abs(w_nf4_deq - g["w_nf4_deq"])))
    assert err < 1e-5, f"w_nf4_deq error {err:.2e}"
    err = float(jnp.max(jnp.abs(nf4_indices - g["nf4_indices"])))
    assert err < 1e-5, f"nf4_indices error {err:.2e}"
    err = float(jnp.max(jnp.abs(c1 - g["c1"])))
    assert err < 1e-5, f"c1 error {err:.2e}"

    # 3. Double Quantization parity
    c1_deq, q2, c2 = double_quantize(g["c1"], dq_block_size=8)
    err = float(jnp.max(jnp.abs(c1_deq - g["c1_deq"])))
    assert err < 1e-5, f"c1_deq error {err:.2e}"
    err = float(jnp.max(jnp.abs(q2 - g["q2"])))
    assert err < 1e-5, f"q2 error {err:.2e}"
    err = float(jnp.max(jnp.abs(c2 - g["c2"])))
    assert err < 1e-5, f"c2 error {err:.2e}"

    # 4. QLoRA forward pass
    scaling = 2.0 / 4.0
    out_qlora = qlora_forward(g["x_raw"], g["w_dq_deq"], g["lora_a"], g["lora_b"], scaling, g["b_orig"])
    err = float(jnp.max(jnp.abs(out_qlora - g["out_qlora"])))
    assert err < 1e-5, f"out_qlora error {err:.2e}"

    # 5. LoftQ forward pass
    out_loftq = qlora_forward(g["x_raw"], g["loftq_q_deq"], g["loftq_a"], g["loftq_b"], scaling=1.0, bias=g["b_orig"])
    err = float(jnp.max(jnp.abs(out_loftq - g["out_loftq"])))
    assert err < 1e-5, f"out_loftq error {err:.2e}"

    # 6. QA-LoRA forward pass
    out_qa_lora, a_qa = qa_lora_forward(
        g["x_raw"], g["w_nf4_deq"], g["lora_a"], g["lora_b"], g["group_scales"], scaling, group_size=4, bias=g["b_orig"]
    )
    err = float(jnp.max(jnp.abs(out_qa_lora - g["out_qa_lora"])))
    assert err < 1e-5, f"out_qa_lora error {err:.2e}"
    err = float(jnp.max(jnp.abs(a_qa - g["a_qa"])))
    assert err < 1e-5, f"a_qa error {err:.2e}"

    print("14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
