"""xLSTM sLSTM and mLSTM — PyTorch 2.x reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm.safetensors

``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def slstm_forward(
    x_seq: torch.Tensor,
    w_i: torch.Tensor,
    w_f: torch.Tensor,
    w_z: torch.Tensor,
    w_o: torch.Tensor,
    u_i: torch.Tensor,
    u_f: torch.Tensor,
    u_z: torch.Tensor,
    u_o: torch.Tensor,
    b_i: torch.Tensor,
    b_f: torch.Tensor,
    b_z: torch.Tensor,
    b_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sLSTM forward unrolling with exponential gating and stabilization state."""
    B, T, _ = x_seq.shape
    D_hid = w_i.shape[0]

    c_s = torch.zeros(B, D_hid, device=x_seq.device, dtype=x_seq.dtype)
    n_s = torch.zeros(B, D_hid, device=x_seq.device, dtype=x_seq.dtype)
    m_s = torch.zeros(B, D_hid, device=x_seq.device, dtype=x_seq.dtype)
    h_s = torch.zeros(B, D_hid, device=x_seq.device, dtype=x_seq.dtype)
    outs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        raw_i = xt @ w_i.t() + h_s @ u_i.t() + b_i
        raw_f = xt @ w_f.t() + h_s @ u_f.t() + b_f
        raw_z = xt @ w_z.t() + h_s @ u_z.t() + b_z
        raw_o = xt @ w_o.t() + h_s @ u_o.t() + b_o

        log_sig_f = torch.log(torch.sigmoid(raw_f) + 1e-12)
        m_next = torch.maximum(log_sig_f + m_s, raw_i)

        i_gate = torch.exp(raw_i - m_next)
        f_gate = torch.sigmoid(raw_f) * torch.exp(m_s - m_next)
        z_cand = torch.tanh(raw_z)
        o_gate = torch.sigmoid(raw_o)

        c_s = f_gate * c_s + i_gate * z_cand
        n_s = f_gate * n_s + i_gate
        m_s = m_next

        h_tilde = c_s / (n_s + 1e-6)
        h_s = o_gate * h_tilde
        outs.append(h_s.unsqueeze(1))

    seq_out = torch.cat(outs, dim=1)
    return seq_out, c_s, n_s


def mlstm_forward(
    x_seq: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_i: torch.Tensor,
    w_f: torch.Tensor,
    w_o: torch.Tensor,
    b_q: torch.Tensor,
    b_k: torch.Tensor,
    b_v: torch.Tensor,
    b_i: torch.Tensor,
    b_f: torch.Tensor,
    b_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mLSTM forward unrolling with matrix memory outer-product updates."""
    B, T, _ = x_seq.shape
    D_hid = w_q.shape[0]

    C_m = torch.zeros(B, D_hid, D_hid, device=x_seq.device, dtype=x_seq.dtype)
    n_m = torch.zeros(B, D_hid, device=x_seq.device, dtype=x_seq.dtype)
    m_m = torch.zeros(B, 1, device=x_seq.device, dtype=x_seq.dtype)
    outs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        q_t = xt @ w_q.t() + b_q
        k_t = xt @ w_k.t() + b_k
        v_t = xt @ w_v.t() + b_v

        raw_i = (xt @ w_i.t() + b_i).mean(dim=-1, keepdim=True)
        raw_f = (xt @ w_f.t() + b_f).mean(dim=-1, keepdim=True)
        raw_o = xt @ w_o.t() + b_o

        log_sig_f = torch.log(torch.sigmoid(raw_f) + 1e-12)
        m_next = torch.maximum(log_sig_f + m_m, raw_i)

        i_gate = torch.exp(raw_i - m_next)
        f_gate = torch.sigmoid(raw_f) * torch.exp(m_m - m_next)
        o_gate = torch.sigmoid(raw_o)

        vk = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))
        C_m = f_gate.unsqueeze(-1) * C_m + i_gate.unsqueeze(-1) * vk
        n_m = f_gate * n_m + i_gate * k_t
        m_m = m_next

        C_q = torch.bmm(C_m, q_t.unsqueeze(-1)).squeeze(-1)
        denom = torch.abs(torch.bmm(n_m.unsqueeze(1), q_t.unsqueeze(-1)).squeeze(-1))
        denom = torch.clamp(denom, min=1.0)
        h_tilde = C_q / denom
        h_m = o_gate * h_tilde
        outs.append(h_m.unsqueeze(1))

    seq_out = torch.cat(outs, dim=1)
    return seq_out, C_m, n_m


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor for cross-track parity verification
    batch = 8
    in_dim = 4
    out_dim = 4

    test_in = torch.randn(batch, in_dim)
    w_linear = torch.randn(in_dim, out_dim) * 0.1
    b_linear = torch.randn(out_dim) * 0.1
    test_out = linear_anchor(test_in, w_linear, b_linear)

    # 2. Sequence data for sLSTM & mLSTM
    B = 2
    T = 4
    D_in = 4
    D_hid = 4

    x_seq = torch.randn(B, T, D_in)

    # --- sLSTM Parameters ---
    slstm_w_i = torch.randn(D_hid, D_in) * 0.2
    slstm_w_f = torch.randn(D_hid, D_in) * 0.2
    slstm_w_z = torch.randn(D_hid, D_in) * 0.2
    slstm_w_o = torch.randn(D_hid, D_in) * 0.2

    slstm_u_i = torch.randn(D_hid, D_hid) * 0.2
    slstm_u_f = torch.randn(D_hid, D_hid) * 0.2
    slstm_u_z = torch.randn(D_hid, D_hid) * 0.2
    slstm_u_o = torch.randn(D_hid, D_hid) * 0.2

    slstm_b_i = torch.zeros(D_hid)
    slstm_b_f = torch.ones(D_hid)
    slstm_b_z = torch.zeros(D_hid)
    slstm_b_o = torch.zeros(D_hid)

    slstm_seq_out, slstm_final_c, slstm_final_n = slstm_forward(
        x_seq,
        slstm_w_i,
        slstm_w_f,
        slstm_w_z,
        slstm_w_o,
        slstm_u_i,
        slstm_u_f,
        slstm_u_z,
        slstm_u_o,
        slstm_b_i,
        slstm_b_f,
        slstm_b_z,
        slstm_b_o,
    )

    # --- mLSTM Parameters ---
    mlstm_w_q = torch.randn(D_hid, D_in) * 0.2
    mlstm_w_k = torch.randn(D_hid, D_in) * 0.2
    mlstm_w_v = torch.randn(D_hid, D_in) * 0.2
    mlstm_w_i = torch.randn(D_hid, D_in) * 0.2
    mlstm_w_f = torch.randn(D_hid, D_in) * 0.2
    mlstm_w_o = torch.randn(D_hid, D_in) * 0.2

    mlstm_b_q = torch.zeros(D_hid)
    mlstm_b_k = torch.zeros(D_hid)
    mlstm_b_v = torch.zeros(D_hid)
    mlstm_b_i = torch.zeros(D_hid)
    mlstm_b_f = torch.ones(D_hid)
    mlstm_b_o = torch.zeros(D_hid)

    mlstm_seq_out, mlstm_final_c, mlstm_final_n = mlstm_forward(
        x_seq,
        mlstm_w_q,
        mlstm_w_k,
        mlstm_w_v,
        mlstm_w_i,
        mlstm_w_f,
        mlstm_w_o,
        mlstm_b_q,
        mlstm_b_k,
        mlstm_b_v,
        mlstm_b_i,
        mlstm_b_f,
        mlstm_b_o,
    )

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "x_seq": x_seq,
        # sLSTM weights & targets
        "slstm_w_i": slstm_w_i,
        "slstm_w_f": slstm_w_f,
        "slstm_w_z": slstm_w_z,
        "slstm_w_o": slstm_w_o,
        "slstm_u_i": slstm_u_i,
        "slstm_u_f": slstm_u_f,
        "slstm_u_z": slstm_u_z,
        "slstm_u_o": slstm_u_o,
        "slstm_b_i": slstm_b_i,
        "slstm_b_f": slstm_b_f,
        "slstm_b_z": slstm_b_z,
        "slstm_b_o": slstm_b_o,
        "slstm_seq_out": slstm_seq_out,
        "slstm_final_c": slstm_final_c,
        "slstm_final_n": slstm_final_n,
        # mLSTM weights & targets
        "mlstm_w_q": mlstm_w_q,
        "mlstm_w_k": mlstm_w_k,
        "mlstm_w_v": mlstm_w_v,
        "mlstm_w_i": mlstm_w_i,
        "mlstm_w_f": mlstm_w_f,
        "mlstm_w_o": mlstm_w_o,
        "mlstm_b_q": mlstm_b_q,
        "mlstm_b_k": mlstm_b_k,
        "mlstm_b_v": mlstm_b_v,
        "mlstm_b_i": mlstm_b_i,
        "mlstm_b_f": mlstm_b_f,
        "mlstm_b_o": mlstm_b_o,
        "mlstm_seq_out": mlstm_seq_out,
        "mlstm_final_c": mlstm_final_c,
        "mlstm_final_n": mlstm_final_n,
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
