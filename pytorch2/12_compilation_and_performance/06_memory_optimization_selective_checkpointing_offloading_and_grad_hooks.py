"""Memory Optimization Selective Checkpointing Offloading and Grad Hooks — PyTorch 2.x reference implementation.

Topic: 12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks
Golden weights: weights/12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks.safetensors

``python pytorch2/12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks.py`` verifies against the golden checkpoint;
``python pytorch2/12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def selective_checkpoint_silu_gate(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selective activation checkpointing: saves only pre-activation z = x @ w + b,
    recomputes SiLU(z) during backward pass, and computes exact parameter gradients.
    
    Forward:
    z = x @ w + b
    a = z * sigmoid(z)
    y = a * gate
    
    Backward:
    Recomputes a = z * sigmoid(z) from saved z.
    da/dz = sigmoid(z) * (1.0 + z * (1.0 - sigmoid(z)))
    """
    z = x @ w + b
    sig_z = torch.sigmoid(z)
    a = z * sig_z
    y = a * gate

    # Backward pass with recomputed a and sig_z
    g_a = grad_out * gate
    da_dz = sig_z * (1.0 + z * (1.0 - sig_z))
    g_z = g_a * da_dz

    grad_w = x.t() @ g_z
    grad_b = torch.sum(g_z, dim=0)
    grad_x = g_z @ w.t()
    grad_gate = grad_out * a
    return y, grad_w, grad_b, grad_x, grad_gate


def post_accumulate_grad_hook_step(
    w: torch.Tensor,
    grad_w: torch.Tensor,
    running_m: torch.Tensor,
    beta: float = 0.9,
    lr: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulates post_accumulate_grad_hook executing asynchronous gradient update."""
    m_new = float(beta) * running_m + (1.0 - float(beta)) * grad_w
    w_new = w - float(lr) * m_new
    return w_new, m_new


def memory_footprint_analysis(
    batch: int,
    seq_len: int,
    d_model: int,
    d_ffn: int,
) -> dict[str, float]:
    """Computes theoretical activation memory bytes for full vs selective vs no checkpointing."""
    element_bytes = 4.0  # float32
    tokens = float(batch * seq_len)

    # Standard: saves all inputs, intermediate projections, and non-linearities
    mem_full = tokens * (d_model + d_ffn + d_ffn + d_model) * element_bytes
    # Selective: saves only boundary projections (d_model), recomputes d_ffn activations
    mem_selective = tokens * (d_model + d_model) * element_bytes
    savings_ratio = 1.0 - (mem_selective / mem_full)
    return {
        "mem_full_bytes": mem_full,
        "mem_selective_bytes": mem_selective,
        "savings_ratio": savings_ratio,
    }


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Selective Checkpoint SiLU Gate: Batch=3, D_in=4, D_out=5
    x = torch.randn(3, 4)
    w = torch.randn(4, 5) * 0.2
    b = torch.randn(5) * 0.1
    gate = torch.randn(3, 5) * 0.5 + 1.0
    grad_out = torch.randn(3, 5)

    y_ref, grad_w_ref, grad_b_ref, grad_x_ref, grad_gate_ref = selective_checkpoint_silu_gate(
        x, w, b, gate, grad_out
    )

    # 2. Post-Accumulate Grad Hook step
    running_m = torch.randn(4, 5) * 0.05
    w_new_ref, m_new_ref = post_accumulate_grad_hook_step(
        w, grad_w_ref, running_m, beta=0.9, lr=0.01
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "chk_x": x,
        "chk_w": w,
        "chk_b": b,
        "chk_gate": gate,
        "chk_grad_out": grad_out,
        "chk_y_ref": y_ref,
        "chk_grad_w_ref": grad_w_ref,
        "chk_grad_b_ref": grad_b_ref,
        "chk_grad_x_ref": grad_x_ref,
        "chk_grad_gate_ref": grad_gate_ref,
        "hook_running_m": running_m,
        "hook_w_new_ref": w_new_ref,
        "hook_m_new_ref": m_new_ref,
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
