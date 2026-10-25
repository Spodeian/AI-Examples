"""xLSTM sLSTM and mLSTM — functional JAX implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm.
Run ``python jax/05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm.safetensors"


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def slstm_forward(
    x_seq: jnp.ndarray,
    w_i: jnp.ndarray,
    w_f: jnp.ndarray,
    w_z: jnp.ndarray,
    w_o: jnp.ndarray,
    u_i: jnp.ndarray,
    u_f: jnp.ndarray,
    u_z: jnp.ndarray,
    u_o: jnp.ndarray,
    b_i: jnp.ndarray,
    b_f: jnp.ndarray,
    b_z: jnp.ndarray,
    b_o: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """sLSTM recurrent forward pass in JAX."""
    B, T, _ = x_seq.shape
    D_hid = w_i.shape[0]

    c_s = jnp.zeros((B, D_hid), dtype=x_seq.dtype)
    n_s = jnp.zeros((B, D_hid), dtype=x_seq.dtype)
    m_s = jnp.zeros((B, D_hid), dtype=x_seq.dtype)
    h_s = jnp.zeros((B, D_hid), dtype=x_seq.dtype)
    outs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        raw_i = xt @ w_i.T + h_s @ u_i.T + b_i
        raw_f = xt @ w_f.T + h_s @ u_f.T + b_f
        raw_z = xt @ w_z.T + h_s @ u_z.T + b_z
        raw_o = xt @ w_o.T + h_s @ u_o.T + b_o

        log_sig_f = jnp.log(_sigmoid(raw_f) + 1e-12)
        m_next = jnp.maximum(log_sig_f + m_s, raw_i)

        i_gate = jnp.exp(raw_i - m_next)
        f_gate = _sigmoid(raw_f) * jnp.exp(m_s - m_next)
        z_cand = jnp.tanh(raw_z)
        o_gate = _sigmoid(raw_o)

        c_s = f_gate * c_s + i_gate * z_cand
        n_s = f_gate * n_s + i_gate
        m_s = m_next

        h_tilde = c_s / (n_s + 1e-6)
        h_s = o_gate * h_tilde
        outs.append(jnp.expand_dims(h_s, axis=1))

    seq_out = jnp.concatenate(outs, axis=1)
    return seq_out, c_s, n_s


def mlstm_forward(
    x_seq: jnp.ndarray,
    w_q: jnp.ndarray,
    w_k: jnp.ndarray,
    w_v: jnp.ndarray,
    w_i: jnp.ndarray,
    w_f: jnp.ndarray,
    w_o: jnp.ndarray,
    b_q: jnp.ndarray,
    b_k: jnp.ndarray,
    b_v: jnp.ndarray,
    b_i: jnp.ndarray,
    b_f: jnp.ndarray,
    b_o: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """mLSTM recurrent forward pass in JAX."""
    B, T, _ = x_seq.shape
    D_hid = w_q.shape[0]

    C_m = jnp.zeros((B, D_hid, D_hid), dtype=x_seq.dtype)
    n_m = jnp.zeros((B, D_hid), dtype=x_seq.dtype)
    m_m = jnp.zeros((B, 1), dtype=x_seq.dtype)
    outs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        q_t = xt @ w_q.T + b_q
        k_t = xt @ w_k.T + b_k
        v_t = xt @ w_v.T + b_v

        raw_i = jnp.mean(xt @ w_i.T + b_i, axis=-1, keepdims=True)
        raw_f = jnp.mean(xt @ w_f.T + b_f, axis=-1, keepdims=True)
        raw_o = xt @ w_o.T + b_o

        log_sig_f = jnp.log(_sigmoid(raw_f) + 1e-12)
        m_next = jnp.maximum(log_sig_f + m_m, raw_i)

        i_gate = jnp.exp(raw_i - m_next)
        f_gate = _sigmoid(raw_f) * jnp.exp(m_m - m_next)
        o_gate = _sigmoid(raw_o)

        vk = jnp.expand_dims(v_t, axis=-1) @ jnp.expand_dims(k_t, axis=1)
        C_m = jnp.expand_dims(f_gate, axis=-1) * C_m + jnp.expand_dims(i_gate, axis=-1) * vk
        n_m = f_gate * n_m + i_gate * k_t
        m_m = m_next

        C_q = jnp.squeeze(C_m @ jnp.expand_dims(q_t, axis=-1), axis=-1)
        denom = jnp.squeeze(jnp.abs(jnp.expand_dims(n_m, axis=1) @ jnp.expand_dims(q_t, axis=-1)), axis=-1)
        denom = jnp.clip(denom, a_min=1.0)
        h_tilde = C_q / denom
        h_m = o_gate * h_tilde
        outs.append(jnp.expand_dims(h_m, axis=1))

    seq_out = jnp.concatenate(outs, axis=1)
    return seq_out, C_m, n_m


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    s_seq, s_c, s_n = slstm_forward(
        g["x_seq"],
        g["slstm_w_i"],
        g["slstm_w_f"],
        g["slstm_w_z"],
        g["slstm_w_o"],
        g["slstm_u_i"],
        g["slstm_u_f"],
        g["slstm_u_z"],
        g["slstm_u_o"],
        g["slstm_b_i"],
        g["slstm_b_f"],
        g["slstm_b_z"],
        g["slstm_b_o"],
    )
    err_slstm = float(jnp.max(jnp.abs(s_seq - g["slstm_seq_out"])))
    assert err_slstm < 1e-5, f"sLSTM error {err_slstm:.2e}"

    m_seq, m_c, m_n = mlstm_forward(
        g["x_seq"],
        g["mlstm_w_q"],
        g["mlstm_w_k"],
        g["mlstm_w_v"],
        g["mlstm_w_i"],
        g["mlstm_w_f"],
        g["mlstm_w_o"],
        g["mlstm_b_q"],
        g["mlstm_b_k"],
        g["mlstm_b_v"],
        g["mlstm_b_i"],
        g["mlstm_b_f"],
        g["mlstm_b_o"],
    )
    err_mlstm = float(jnp.max(jnp.abs(m_seq - g["mlstm_seq_out"])))
    assert err_mlstm < 1e-5, f"mLSTM error {err_mlstm:.2e}"

    print("05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
