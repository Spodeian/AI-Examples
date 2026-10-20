"""Functional JAX implementation for Category 05 Subfolder 03 Topic 01:
RNN, LSTM, GRU & Attention Seq2Seq
in 05_architectures/03_recurrent_ssm_and_memory.
"""

from __future__ import annotations

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def lstm_forward(
    x_seq: jnp.ndarray,
    h0: jnp.ndarray,
    c0: jnp.ndarray,
    w_ih: jnp.ndarray,
    w_hh: jnp.ndarray,
    b_ih: jnp.ndarray,
    b_hh: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Recurrent LSTM unrolling forward pass in JAX."""
    B, T, _ = x_seq.shape
    h_t = h0
    c_t = c0
    outputs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        gates = xt @ w_ih.T + b_ih + h_t @ w_hh.T + b_hh
        i_t, f_t, g_t, o_t = jnp.split(gates, 4, axis=-1)
        i_t = jax.nn.sigmoid(i_t)
        f_t = jax.nn.sigmoid(f_t)
        g_t = jnp.tanh(g_t)
        o_t = jax.nn.sigmoid(o_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * jnp.tanh(c_t)
        outputs.append(jnp.expand_dims(h_t, axis=1))

    seq_out = jnp.concatenate(outputs, axis=1)
    return seq_out, h_t, c_t


def gru_forward(
    x_seq: jnp.ndarray,
    h0: jnp.ndarray,
    w_ih: jnp.ndarray,
    w_hh: jnp.ndarray,
    b_ih: jnp.ndarray,
    b_hh: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recurrent GRU unrolling forward pass in JAX."""
    B, T, D_in = x_seq.shape
    D_hid = h0.shape[-1]
    h_t = h0
    outputs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        gi = xt @ w_ih.T + b_ih
        gh = h_t @ w_hh.T + b_hh

        r_t = jax.nn.sigmoid(gi[:, :D_hid] + gh[:, :D_hid])
        z_t = jax.nn.sigmoid(gi[:, D_hid : 2 * D_hid] + gh[:, D_hid : 2 * D_hid])
        n_t = jnp.tanh(gi[:, 2 * D_hid :] + r_t * gh[:, 2 * D_hid :])

        h_t = (1.0 - z_t) * n_t + z_t * h_t
        outputs.append(jnp.expand_dims(h_t, axis=1))

    seq_out = jnp.concatenate(outputs, axis=1)
    return seq_out, h_t


def luong_attention_forward(
    decoder_query: jnp.ndarray,
    enc_states: jnp.ndarray,
    w_attn: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Luong Multiplicative Attention forward pass in JAX."""
    s_proj = decoder_query @ w_attn  # [B, D_hid]
    scores = jnp.einsum("btd,bd->bt", enc_states, s_proj)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("bt,btd->bd", attn_weights, enc_states)
    return context, attn_weights


def verify_against_golden() -> None:
    """Verifies JAX implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = float(jnp.max(jnp.abs(test_out - weights["test_output"])))
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
    err_lstm_seq = float(jnp.max(jnp.abs(lstm_seq - weights["lstm_seq_out"])))
    err_lstm_h = float(jnp.max(jnp.abs(lstm_h - weights["lstm_final_h"])))
    err_lstm_c = float(jnp.max(jnp.abs(lstm_c - weights["lstm_final_c"])))
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
    err_gru_seq = float(jnp.max(jnp.abs(gru_seq - weights["gru_seq_out"])))
    err_gru_h = float(jnp.max(jnp.abs(gru_h - weights["gru_final_h"])))
    assert err_gru_seq < 1e-6, f"GRU sequence error: {err_gru_seq}"
    assert err_gru_h < 1e-6, f"GRU final h error: {err_gru_h}"

    # 4. Luong Attention Verification
    context, attn_w = luong_attention_forward(
        weights["decoder_query"],
        weights["lstm_seq_out"],
        weights["w_attn"],
    )
    err_ctx = float(jnp.max(jnp.abs(context - weights["context"])))
    err_attn = float(jnp.max(jnp.abs(attn_w - weights["attn_weights"])))
    assert err_ctx < 1e-6, f"Attention context error: {err_ctx}"
    assert err_attn < 1e-6, f"Attention weights error: {err_attn}"

    print("Functional JAX LSTM, GRU & Attention Seq2Seq successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
