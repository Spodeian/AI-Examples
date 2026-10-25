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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file


    def generate_weights() -> None:
        torch.manual_seed(42)

        # 1. Standard Linear Anchor for test_cross_track_parity.py
        batch = 8
        in_dim = 4
        out_dim = 4

        test_in = torch.randn(batch, in_dim)
        w_linear = torch.randn(in_dim, out_dim) * 0.1
        b_linear = torch.randn(out_dim) * 0.1
        test_out = test_in @ w_linear + b_linear

        # 2. Sequence data for LSTM & GRU
        # B = 2, T = 4, D_in = 4, D_hid = 4
        B = 2
        T = 4
        D_in = 4
        D_hid = 4

        x_seq = torch.randn(B, T, D_in)
        h0 = torch.zeros(B, D_hid)
        c0 = torch.zeros(B, D_hid)

        # LSTM Parameters: 4 * D_hid = 16
        lstm_w_ih = torch.randn(4 * D_hid, D_in) * 0.2
        lstm_w_hh = torch.randn(4 * D_hid, D_hid) * 0.2
        lstm_b_ih = torch.zeros(4 * D_hid)
        lstm_b_hh = torch.zeros(4 * D_hid)
        # Set forget gate bias to 1.0 (indices D_hid to 2*D_hid)
        lstm_b_ih[D_hid : 2 * D_hid] = 1.0

        # Execute LSTM forward pass
        h_t = h0.clone()
        c_t = c0.clone()
        lstm_outputs = []

        for t in range(T):
            xt = x_seq[:, t, :]
            gates = xt @ lstm_w_ih.t() + lstm_b_ih + h_t @ lstm_w_hh.t() + lstm_b_hh
            i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            lstm_outputs.append(h_t.unsqueeze(1))

        lstm_seq_out = torch.cat(lstm_outputs, dim=1)  # [B, T, D_hid]

        # GRU Parameters: 3 * D_hid = 12
        gru_w_ih = torch.randn(3 * D_hid, D_in) * 0.2
        gru_w_hh = torch.randn(3 * D_hid, D_hid) * 0.2
        gru_b_ih = torch.zeros(3 * D_hid)
        gru_b_hh = torch.zeros(3 * D_hid)

        # Execute GRU forward pass
        h_gru = h0.clone()
        gru_outputs = []

        for t in range(T):
            xt = x_seq[:, t, :]
            gi = xt @ gru_w_ih.t() + gru_b_ih
            gh = h_gru @ gru_w_hh.t() + gru_b_hh

            r_t = torch.sigmoid(gi[:, :D_hid] + gh[:, :D_hid])
            z_t = torch.sigmoid(gi[:, D_hid : 2 * D_hid] + gh[:, D_hid : 2 * D_hid])
            n_t = torch.tanh(gi[:, 2 * D_hid :] + r_t * gh[:, 2 * D_hid :])

            h_gru = (1.0 - z_t) * n_t + z_t * h_gru
            gru_outputs.append(h_gru.unsqueeze(1))

        gru_seq_out = torch.cat(gru_outputs, dim=1)

        # 3. Luong Dot-Product Attention Verification
        # Decoder query s: [B, D_hid], Encoder keys H_enc: [B, T, D_hid]
        w_attn = torch.randn(D_hid, D_hid) * 0.2
        decoder_query = torch.randn(B, D_hid)

        # Luong score: s @ W @ h_i.T
        s_proj = decoder_query @ w_attn  # [B, D_hid]
        scores = torch.bmm(lstm_seq_out, s_proj.unsqueeze(-1)).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(scores, dim=-1)  # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_seq_out).squeeze(1)  # [B, D_hid]

        tensors = {
            "test_input": test_in,
            "weight": w_linear,
            "bias": b_linear,
            "test_output": test_out,
            "x_seq": x_seq,
            "h0": h0,
            "c0": c0,
            "lstm_w_ih": lstm_w_ih,
            "lstm_w_hh": lstm_w_hh,
            "lstm_b_ih": lstm_b_ih,
            "lstm_b_hh": lstm_b_hh,
            "lstm_seq_out": lstm_seq_out,
            "lstm_final_h": h_t,
            "lstm_final_c": c_t,
            "gru_w_ih": gru_w_ih,
            "gru_w_hh": gru_w_hh,
            "gru_b_ih": gru_b_ih,
            "gru_b_hh": gru_b_hh,
            "gru_seq_out": gru_seq_out,
            "gru_final_h": h_gru,
            "w_attn": w_attn,
            "decoder_query": decoder_query,
            "attn_weights": attn_weights,
            "context": context,
        }

        out_dir = Path("weights/05_architectures/03_recurrent_ssm_and_memory")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "01_rnn_lstm_gru_and_attention_seq2seq.safetensors"
        save_file(tensors, str(out_path))
        print(f"Saved {len(tensors)} tensors to {out_path}")


    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_golden()
