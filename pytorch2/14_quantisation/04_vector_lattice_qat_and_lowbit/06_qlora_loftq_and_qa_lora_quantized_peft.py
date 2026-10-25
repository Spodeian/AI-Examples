"""QLoRA, LoftQ, and QA-LoRA Quantized PEFT — PyTorch 2.x reference implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft
Golden weights: weights/14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft.safetensors

``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"

# Canonical QLoRA NF4 16-level standard codebook (Dettmers et al. 2023)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0,
])


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def nf4_quantize(
    x: torch.Tensor,
    block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize tensor x to 4-bit NormalFloat (NF4) with per-block scale normalization."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, block_size)
    levels = NF4_LEVELS.to(x.device)

    # Scale per block: max absolute value
    abs_max = torch.max(torch.abs(x_flat), dim=-1, keepdim=True).values
    scale = torch.clamp(abs_max, min=1e-7)

    # Normalize into [-1, 1]
    norm_x = x_flat / scale
    # Nearest neighbor matching
    diff = torch.abs(norm_x.unsqueeze(-1) - levels)
    idx = torch.argmin(diff, dim=-1)

    # Dequantized values
    deq = levels[idx] * scale
    return deq.reshape(orig_shape).clone(), idx.reshape(orig_shape).clone(), scale.squeeze(-1).clone()


def double_quantize(
    c1: torch.Tensor,
    dq_block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QLoRA Double Quantization: quantize FP32 scales c1 to 8-bit integers with second-level scales."""
    c1_flat = c1.reshape(-1, dq_block_size)
    c2 = torch.max(c1_flat, dim=-1, keepdim=True).values / 127.0
    c2 = torch.clamp(c2, min=1e-7)
    q2 = torch.clamp(torch.round(c1_flat / c2), -127.0, 127.0)
    c1_deq = q2 * c2
    return c1_deq.reshape(c1.shape).clone(), q2.reshape(c1.shape).clone(), c2.squeeze(-1).clone()


def qlora_forward(
    x: torch.Tensor,
    w_deq: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """QLoRA forward pass: Base NF4 frozen GEMM + low-rank LoRA adapter."""
    base_out = x @ w_deq
    lora_out = (x @ lora_a @ lora_b) * scaling
    y = base_out + lora_out
    if bias is not None:
        y = y + bias
    return y


def loftq_init(
    w: torch.Tensor,
    rank: int = 4,
    num_iters: int = 2,
    block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LoftQ: alternating SVD quantization residual initialization."""
    # Alternating optimization:
    # 1. Q = quantize(W - A @ B)
    # 2. U, S, V = svd(W - Q) -> A = U sqrt(S), B = sqrt(S) V^T
    a = torch.zeros(w.shape[0], rank, device=w.device)
    b = torch.zeros(rank, w.shape[1], device=w.device)
    q_deq = w.clone()

    for _ in range(num_iters):
        residual = w - (a @ b)
        q_deq, _, _ = nf4_quantize(residual, block_size=block_size)
        u, s, vh = torch.linalg.svd(w - q_deq, full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]
        s_sqrt = torch.diag(torch.sqrt(torch.clamp(s_r, min=1e-7)))
        a = u_r @ s_sqrt
        b = s_sqrt @ vh_r

    return q_deq.clone(), a.clone(), b.clone()


def qa_lora_forward(
    x: torch.Tensor,
    w_deq: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    group_scales: torch.Tensor,
    scaling: float,
    group_size: int = 4,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """QA-LoRA: group-wise quantization-aware low-rank adaptation."""
    # group_scales: [num_groups, 1] where num_groups = d_in // group_size
    # Repeat group_scales to match d_in: shape [d_in, 1]
    g_expanded = group_scales.repeat_interleave(group_size, dim=0)
    a_qa = lora_a * g_expanded
    base_out = x @ w_deq
    lora_out = (x @ a_qa @ lora_b) * scaling
    y = base_out + lora_out
    if bias is not None:
        y = y + bias
    return y, a_qa.clone()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, d_in, d_out, rank = 8, 16, 16, 4
    block_size = 8
    group_size = 4
    num_groups = d_in // group_size
    scaling = 2.0 / float(rank)

    x_raw = torch.randn(b, d_in) * 1.2
    w_orig = torch.randn(d_in, d_out) * 0.35
    b_orig = torch.randn(d_out) * 0.05

    # 1. Primary NF4 quantization
    w_nf4_deq, nf4_indices, c1 = nf4_quantize(w_orig, block_size=block_size)

    # 2. QLoRA Double Quantization
    c1_deq, q2, c2 = double_quantize(c1, dq_block_size=8)
    # Reconstruct NF4 with double quantized scales:
    w_flat = w_orig.reshape(-1, block_size)
    levels = NF4_LEVELS.to(w_orig.device)
    diff = torch.abs((w_flat / torch.clamp(c1.unsqueeze(-1), min=1e-7)).unsqueeze(-1) - levels)
    idx_flat = torch.argmin(diff, dim=-1)
    w_dq_deq = (levels[idx_flat] * c1_deq.unsqueeze(-1)).reshape(d_in, d_out).clone()

    # 3. Standard LoRA adapters (random initialization)
    lora_a = torch.randn(d_in, rank) * 0.1
    lora_b = torch.randn(rank, d_out) * 0.1

    # 4. QLoRA forward pass
    out_qlora = qlora_forward(x_raw, w_dq_deq, lora_a, lora_b, scaling, b_orig)

    # 5. LoftQ alternating initialization
    loftq_q_deq, loftq_a, loftq_b = loftq_init(w_orig, rank=rank, num_iters=2, block_size=block_size)
    out_loftq = qlora_forward(x_raw, loftq_q_deq, loftq_a, loftq_b, scaling=1.0, bias=b_orig)

    # 6. QA-LoRA group-wise adaptation
    group_scales = torch.rand(num_groups, 1) * 0.5 + 0.75
    out_qa_lora, a_qa = qa_lora_forward(
        x_raw, w_nf4_deq, lora_a, lora_b, group_scales, scaling, group_size=group_size, bias=b_orig
    )

    # 7. FP32 unquantized ground truth
    out_fp32 = x_raw @ (w_orig + (lora_a @ lora_b) * scaling) + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_raw": x_raw,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "w_nf4_deq": w_nf4_deq,
        "nf4_indices": nf4_indices.float(),
        "c1": c1,
        "c1_deq": c1_deq,
        "q2": q2,
        "c2": c2,
        "w_dq_deq": w_dq_deq,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "out_qlora": out_qlora,
        "loftq_q_deq": loftq_q_deq,
        "loftq_a": loftq_a,
        "loftq_b": loftq_b,
        "out_loftq": out_loftq,
        "group_scales": group_scales,
        "a_qa": a_qa,
        "out_qa_lora": out_qa_lora,
        "out_fp32": out_fp32,
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
