"""Linear Attention, RWKV, RetNet, GLA, DeltaNet, and TTT — PyTorch 2.x reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt.safetensors

``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def retnet_recurrent_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent Retention (RetNet) inference mode.
    
    S_t = gamma * S_{t-1} + K_t^T V_t  [B, D_k, D_v]
    O_t = Q_t S_t                      [B, D_v]
    """
    B, T, D_k = q.shape
    D_v = v.shape[-1]
    s_t = torch.zeros(B, D_k, D_v, device=q.device, dtype=q.dtype)
    outs = []
    for t in range(T):
        qt = q[:, t, :].unsqueeze(1)  # [B, 1, D_k]
        kt = k[:, t, :].unsqueeze(1)  # [B, 1, D_k]
        vt = v[:, t, :].unsqueeze(1)  # [B, 1, D_v]

        # s_t = gamma * s_{t-1} + kt.T @ vt
        s_t = gamma * s_t + torch.bmm(kt.transpose(1, 2), vt)  # [B, D_k, D_v]
        ot = torch.bmm(qt, s_t).squeeze(1)  # [B, D_v]
        outs.append(ot.unsqueeze(1))

    seq_out = torch.cat(outs, dim=1)
    return seq_out, s_t


def retnet_parallel_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Parallel Retention (RetNet) training mode.
    
    O = (Q K^T * D) V
    where D_{i, j} = gamma^{i - j} for i >= j, else 0.
    """
    B, T, D_k = q.shape
    scores = torch.bmm(q, k.transpose(1, 2))  # [B, T, T]

    # Causal decay matrix D
    indices = torch.arange(T, device=q.device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)  # [T, T], diff[i, j] = i - j
    mask = diff >= 0
    decay = torch.where(mask, gamma ** diff.float(), torch.zeros_like(diff, dtype=torch.float32))

    retention_matrix = scores * decay.unsqueeze(0)  # [B, T, T]
    out = torch.bmm(retention_matrix, v)  # [B, T, D_v]
    return out


def deltanet_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeltaNet online associative memory update.
    
    S_t = S_{t-1} + beta_t * (V_t - S_{t-1} K_t) K_t^T
    O_t = Q_t S_t
    """
    B, T, D_k = q.shape
    D_v = v.shape[-1]
    s_t = torch.zeros(B, D_k, D_v, device=q.device, dtype=q.dtype)
    outs = []
    for t in range(T):
        qt = q[:, t, :].unsqueeze(1)       # [B, 1, D_k]
        kt = k[:, t, :].unsqueeze(2)       # [B, D_k, 1]
        vt = v[:, t, :].unsqueeze(1)       # [B, 1, D_v]
        b_val = beta[:, t].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

        # pred_v = kt^T @ s_t -> [B, 1, D_v]
        pred_v = torch.bmm(kt.transpose(1, 2), s_t)
        err_v = vt - pred_v  # [B, 1, D_v]

        # delta update: s_t = s_{t-1} + b_val * (kt @ err_v)
        s_t = s_t + b_val * torch.bmm(kt, err_v)  # [B, D_k, D_v]

        ot = torch.bmm(qt, s_t).squeeze(1)  # [B, D_v]
        outs.append(ot.unsqueeze(1))

    seq_out = torch.cat(outs, dim=1)
    return seq_out, s_t


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

    # 2. Sequence data for RetNet & DeltaNet
    B = 2
    T = 4
    D_k = 4
    D_v = 4
    gamma = 0.9

    q = torch.randn(B, T, D_k) * 0.2
    k = torch.randn(B, T, D_k) * 0.2
    v = torch.randn(B, T, D_v) * 0.2
    gamma_tensor = torch.tensor([gamma])

    ret_recurrent_out, ret_final_s = retnet_recurrent_forward(q, k, v, gamma)
    ret_parallel_out = retnet_parallel_forward(q, k, v, gamma)

    beta = torch.sigmoid(torch.randn(B, T) * 0.5)
    delta_out, delta_final_s = deltanet_forward(q, k, v, beta)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "q": q,
        "k": k,
        "v": v,
        "gamma": gamma_tensor,
        "ret_recurrent_out": ret_recurrent_out,
        "ret_parallel_out": ret_parallel_out,
        "ret_final_s": ret_final_s,
        "beta": beta,
        "delta_out": delta_out,
        "delta_final_s": delta_final_s,
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
