"""Mamba Selective SSM and Mamba2 SSD — PyTorch 2.x reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd.safetensors

``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def mamba_selective_forward(
    x_seq: torch.Tensor,
    a_log: torch.Tensor,
    w_b: torch.Tensor,
    w_c: torch.Tensor,
    w_delta: torch.Tensor,
    b_delta: torch.Tensor,
    d_skip: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selective State Space Model (Mamba-1) recurrent unrolling.
    
    Equations:
    B_t = x_t @ W_b^T [B, D_state]
    C_t = x_t @ W_c^T [B, D_state]
    Delta_t = softplus(x_t @ W_delta^T + b_delta) [B, D_model]
    A_bar_t = exp(-exp(a_log).unsqueeze(0) * Delta_t.unsqueeze(-1)) [B, D_model, D_state]
    B_bar_t = Delta_t.unsqueeze(-1) * B_t.unsqueeze(1) [B, D_model, D_state]
    h_t = A_bar_t * h_{t-1} + B_bar_t * x_t.unsqueeze(-1)
    y_t = (h_t * C_t.unsqueeze(1)).sum(dim=-1) + x_t * d_skip
    """
    B, T, D_model = x_seq.shape
    D_state = a_log.shape[1]

    a_real = -torch.exp(a_log)  # [D_model, D_state] < 0 for contractive stability
    h_t = torch.zeros(B, D_model, D_state, device=x_seq.device, dtype=x_seq.dtype)
    outs = []

    for t in range(T):
        xt = x_seq[:, t, :]  # [B, D_model]
        b_t = xt @ w_b.t()   # [B, D_state]
        c_t = xt @ w_c.t()   # [B, D_state]
        delta_t = F.softplus(xt @ w_delta.t() + b_delta)  # [B, D_model]

        # Discretize
        # Delta_t [B, D_model, 1] * a_real [1, D_model, D_state]
        dt_a = delta_t.unsqueeze(-1) * a_real.unsqueeze(0)
        a_bar = torch.exp(dt_a)  # [B, D_model, D_state]

        # B_bar = delta_t.unsqueeze(-1) * b_t.unsqueeze(1) -> [B, D_model, D_state]
        b_bar = delta_t.unsqueeze(-1) * b_t.unsqueeze(1)

        # Recurrent state update: h_t = A_bar * h_{t-1} + B_bar * x_t
        h_t = a_bar * h_t + b_bar * xt.unsqueeze(-1)

        # Output readout: y_t = sum_d (h_t * c_t) + xt * d_skip
        y_t = (h_t * c_t.unsqueeze(1)).sum(dim=-1) + xt * d_skip
        outs.append(y_t.unsqueeze(1))

    seq_out = torch.cat(outs, dim=1)
    return seq_out, h_t


def mamba2_ssd_mask(
    delta_seq: torch.Tensor,
    a_scalar: torch.Tensor,
) -> torch.Tensor:
    """Computes the 1-semiseparable causal decay matrix L for Mamba-2 SSD.
    
    L[i, j] = exp(sum_{k=j+1}^i a_k) for i >= j, else 0.
    """
    B, T = delta_seq.shape
    # a_k = -softplus(...) * a_scalar
    decay = -F.softplus(delta_seq) * torch.abs(a_scalar)  # [B, T]
    cumsum_decay = torch.cumsum(decay, dim=-1)  # [B, T]

    # L[b, i, j] = exp(cumsum[b, i] - cumsum[b, j]) for i >= j
    diff = cumsum_decay.unsqueeze(2) - cumsum_decay.unsqueeze(1)  # [B, T, T]
    mask = torch.tril(torch.ones(T, T, device=delta_seq.device, dtype=torch.bool))
    l_matrix = torch.where(mask.unsqueeze(0), torch.exp(diff), torch.zeros_like(diff))
    return l_matrix


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    batch = 8
    in_dim = 4
    out_dim = 4
    test_in = torch.randn(batch, in_dim)
    w_linear = torch.randn(in_dim, out_dim) * 0.1
    b_linear = torch.randn(out_dim) * 0.1
    test_out = linear_anchor(test_in, w_linear, b_linear)

    # 2. Sequence data for Mamba
    B = 2
    T = 4
    D_model = 4
    D_state = 4

    x_seq = torch.randn(B, T, D_model)
    a_log = torch.randn(D_model, D_state) * 0.2
    w_b = torch.randn(D_state, D_model) * 0.2
    w_c = torch.randn(D_state, D_model) * 0.2
    w_delta = torch.randn(D_model, D_model) * 0.2
    b_delta = torch.zeros(D_model)
    d_skip = torch.randn(D_model) * 0.1

    seq_out, final_h = mamba_selective_forward(
        x_seq, a_log, w_b, w_c, w_delta, b_delta, d_skip
    )

    delta_scalar = torch.randn(B, T) * 0.2
    a_scalar = torch.tensor([0.5])
    l_matrix = mamba2_ssd_mask(delta_scalar, a_scalar)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "x_seq": x_seq,
        "a_log": a_log,
        "w_b": w_b,
        "w_c": w_c,
        "w_delta": w_delta,
        "b_delta": b_delta,
        "d_skip": d_skip,
        "seq_out": seq_out,
        "final_h": final_h,
        "delta_scalar": delta_scalar,
        "a_scalar": a_scalar,
        "l_matrix": l_matrix,
    }


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
