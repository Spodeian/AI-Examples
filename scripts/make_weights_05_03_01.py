"""Generate deterministic seed-42 SafeTensors weights for Category 05 Subfolder 03 Topic 01:
01_rnn_lstm_gru_and_attention_seq2seq.safetensors
"""

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


if __name__ == "__main__":
    generate_weights()
