"""Zero Stages 1 2 3 and FSDP2 Fully Shard — PyTorch 2.x reference implementation.

Topic: 16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard
Golden weights: weights/16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard.safetensors

``python pytorch2/16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard.py`` verifies against the golden checkpoint;
``python pytorch2/16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def fsdp2_allgather_param(param_shards: torch.Tensor) -> torch.Tensor:
    """FSDP2 AllGather: reconstructs full 2D weight matrix from 1D/2D local parameter shards."""
    world_size = param_shards.shape[0]
    return torch.cat([param_shards[r] for r in range(world_size)], dim=0)


def fsdp2_forward_matmul(
    x: torch.Tensor,
    param_shards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FSDP2 forward: AllGather local parameter shards on-demand, compute matmul, and return output."""
    w_full = fsdp2_allgather_param(param_shards)
    h = x @ w_full
    return h, w_full


def fsdp2_backward_reducescatter(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FSDP2 backward: compute full weight gradient and ReduceScatter across ranks."""
    grad_w_full = x.t() @ grad_out
    chunks = torch.chunk(grad_w_full, world_size, dim=0)
    grad_w_shards = torch.stack(chunks, dim=0)
    return grad_w_full, grad_w_shards


def zero1_adam_step(
    p: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    step: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ZeRO-1 Adam optimizer step on partitioned parameter slice."""
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g ** 2)

    bias_corr1 = 1.0 - beta1 ** step
    bias_corr2 = 1.0 - beta2 ** step

    m_hat = m_new / bias_corr1
    v_hat = v_new / bias_corr2

    p_new = p - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return p_new, m_new, v_new


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    world_size = 2
    x = torch.randn(4, 8)
    param_shards = torch.randn(world_size, 4, 6) * 0.2
    grad_out = torch.randn(4, 6)

    # 1. FSDP2 Forward AllGather + Matmul
    fsdp_fwd_out, w_full = fsdp2_forward_matmul(x, param_shards)

    # 2. FSDP2 Backward ReduceScatter
    grad_w_full, grad_w_shards = fsdp2_backward_reducescatter(x, grad_out, world_size)

    # 3. ZeRO-1 Adam Step
    p_init = torch.randn(8)
    g_init = torch.randn(8) * 0.5
    m_init = torch.randn(8) * 0.1
    v_init = torch.abs(torch.randn(8)) * 0.1 + 0.01
    p_new, m_new, v_new = zero1_adam_step(p_init, g_init, m_init, v_init, lr=0.01, step=1)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "param_shards": param_shards,
        "w_full": w_full,
        "fsdp_fwd_out": fsdp_fwd_out,
        "grad_out": grad_out,
        "grad_w_full": grad_w_full,
        "grad_w_shards": grad_w_shards,
        "p_init": p_init,
        "g_init": g_init,
        "m_init": m_init,
        "v_init": v_init,
        "p_new": p_new,
        "m_new": m_new,
        "v_new": v_new,
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
        assert err < 1e-5, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
