"""PyTorch 2.x implementation for Category 05 Subfolder 03 Topic 01:
RNN, LSTM, GRU & Attention Seq2Seq
in 05_architectures/03_recurrent_ssm_and_memory.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def lstm_forward(
    x_seq: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor,
    b_hh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recurrent LSTM unrolling forward pass.
    
    Equations:
    (i_t, f_t, g_t, o_t)^T = (sigmoid, sigmoid, tanh, sigmoid)^T (W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    B, T, _ = x_seq.shape
    h_t = h0.clone()
    c_t = c0.clone()
    outputs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        gates = xt @ w_ih.t() + b_ih + h_t @ w_hh.t() + b_hh
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        outputs.append(h_t.unsqueeze(1))

    seq_out = torch.cat(outputs, dim=1)
    return seq_out, h_t, c_t


def gru_forward(
    x_seq: torch.Tensor,
    h0: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor,
    b_hh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent GRU unrolling forward pass.
    
    Equations:
    r_t = sigmoid(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
    z_t = sigmoid(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
    n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """
    B, T, D_in = x_seq.shape
    D_hid = h0.shape[-1]
    h_t = h0.clone()
    outputs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        gi = xt @ w_ih.t() + b_ih
        gh = h_t @ w_hh.t() + b_hh

        r_t = torch.sigmoid(gi[:, :D_hid] + gh[:, :D_hid])
        z_t = torch.sigmoid(gi[:, D_hid : 2 * D_hid] + gh[:, D_hid : 2 * D_hid])
        n_t = torch.tanh(gi[:, 2 * D_hid :] + r_t * gh[:, 2 * D_hid :])

        h_t = (1.0 - z_t) * n_t + z_t * h_t
        outputs.append(h_t.unsqueeze(1))

    seq_out = torch.cat(outputs, dim=1)
    return seq_out, h_t


def luong_attention_forward(
    decoder_query: torch.Tensor,
    enc_states: torch.Tensor,
    w_attn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Luong Multiplicative Attention forward pass.
    
    score(s, h_i) = s^T W h_i
    alpha = softmax(scores)
    context = sum_i alpha_i h_i
    """
    s_proj = decoder_query @ w_attn  # [B, D_hid]
    scores = torch.bmm(enc_states, s_proj.unsqueeze(-1)).squeeze(-1)  # [B, T]
    attn_weights = torch.softmax(scores, dim=-1)  # [B, T]
    context = torch.bmm(attn_weights.unsqueeze(1), enc_states).squeeze(1)  # [B, D_hid]
    return context, attn_weights


def verify_against_golden() -> None:
    """Verifies PyTorch implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = torch.max(torch.abs(test_out - weights["test_output"])).item()
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    # 2. LSTM Verification
    lstm_seq, lstm_h, lstm_c = lstm_forward(
        weights["x_seq"],
        weights["h0"],
        weights["c0"],
        weights["lstm_w_ih"],
        weights["lstm_w_hh"],
        weights["lstm_b_ih"],
        weights["lstm_b_hh"],
    )
    err_lstm_seq = torch.max(torch.abs(lstm_seq - weights["lstm_seq_out"])).item()
    err_lstm_h = torch.max(torch.abs(lstm_h - weights["lstm_final_h"])).item()
    err_lstm_c = torch.max(torch.abs(lstm_c - weights["lstm_final_c"])).item()
    assert err_lstm_seq < 1e-6, f"LSTM sequence error: {err_lstm_seq}"
    assert err_lstm_h < 1e-6, f"LSTM final h error: {err_lstm_h}"
    assert err_lstm_c < 1e-6, f"LSTM final c error: {err_lstm_c}"

    # 3. GRU Verification
    gru_seq, gru_h = gru_forward(
        weights["x_seq"],
        weights["h0"],
        weights["gru_w_ih"],
        weights["gru_w_hh"],
        weights["gru_b_ih"],
        weights["gru_b_hh"],
    )
    err_gru_seq = torch.max(torch.abs(gru_seq - weights["gru_seq_out"])).item()
    err_gru_h = torch.max(torch.abs(gru_h - weights["gru_final_h"])).item()
    assert err_gru_seq < 1e-6, f"GRU sequence error: {err_gru_seq}"
    assert err_gru_h < 1e-6, f"GRU final h error: {err_gru_h}"

    # 4. Luong Attention Verification
    context, attn_w = luong_attention_forward(
        weights["decoder_query"],
        weights["lstm_seq_out"],
        weights["w_attn"],
    )
    err_ctx = torch.max(torch.abs(context - weights["context"])).item()
    err_attn = torch.max(torch.abs(attn_w - weights["attn_weights"])).item()
    assert err_ctx < 1e-6, f"Attention context error: {err_ctx}"
    assert err_attn < 1e-6, f"Attention weights error: {err_attn}"

    print("PyTorch 2.x LSTM, GRU & Attention Seq2Seq successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
