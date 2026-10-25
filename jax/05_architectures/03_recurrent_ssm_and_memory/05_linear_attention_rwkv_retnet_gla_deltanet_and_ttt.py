"""Linear Attention, RWKV, RetNet, GLA, DeltaNet, and TTT — JAX reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def retnet_recurrent_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recurrent Retention (RetNet) inference mode."""
    B, T, D_k = q.shape
    D_v = v.shape[-1]
    s_t = jnp.zeros((B, D_k, D_v), dtype=q.dtype)
    outs = []

    for t in range(T):
        qt = q[:, t, :]  # [B, D_k]
        kt = k[:, t, :]  # [B, D_k]
        vt = v[:, t, :]  # [B, D_v]

        # s_t = gamma * s_{t-1} + kt^T @ vt
        s_t = gamma * s_t + jnp.einsum("bi,bj->bij", kt, vt)
        ot = jnp.einsum("bi,bij->bj", qt, s_t)
        outs.append(jnp.expand_dims(ot, 1))

    seq_out = jnp.concatenate(outs, axis=1)
    return seq_out, s_t


def retnet_parallel_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """Parallel Retention (RetNet) training mode."""
    B, T, D_k = q.shape
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2))  # [B, T, T]

    indices = jnp.arange(T)
    diff = jnp.expand_dims(indices, 1) - jnp.expand_dims(indices, 0)
    mask = diff >= 0
    decay = jnp.where(mask, gamma ** diff.astype(jnp.float32), 0.0)

    retention_matrix = scores * jnp.expand_dims(decay, 0)
    return jnp.matmul(retention_matrix, v)


def deltanet_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """DeltaNet online associative memory update."""
    B, T, D_k = q.shape
    D_v = v.shape[-1]
    s_t = jnp.zeros((B, D_k, D_v), dtype=q.dtype)
    outs = []

    for t in range(T):
        qt = q[:, t, :]  # [B, D_k]
        kt = k[:, t, :]  # [B, D_k]
        vt = v[:, t, :]  # [B, D_v]
        b_val = beta[:, t:t+1]  # [B, 1]

        # pred_v = kt^T @ s_t
        pred_v = jnp.einsum("bi,bij->bj", kt, s_t)
        err_v = vt - pred_v  # [B, D_v]

        # delta update: s_t = s_{t-1} + beta * (kt @ err_v)
        s_t = s_t + jnp.expand_dims(b_val, -1) * jnp.einsum("bi,bj->bij", kt, err_v)
        ot = jnp.einsum("bi,bij->bj", qt, s_t)
        outs.append(jnp.expand_dims(ot, 1))

    seq_out = jnp.concatenate(outs, axis=1)
    return seq_out, s_t


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. RetNet recurrent & parallel
    gamma = float(golden["gamma"][0])
    ret_out, ret_s = retnet_recurrent_forward(golden["q"], golden["k"], golden["v"], gamma)
    err_ret = float(jnp.max(jnp.abs(ret_out - golden["ret_recurrent_out"])))
    err_rets = float(jnp.max(jnp.abs(ret_s - golden["ret_final_s"])))
    assert err_ret < 1e-5, f"retnet_recurrent_out max abs error {err_ret:.2e}"
    assert err_rets < 1e-5, f"ret_final_s max abs error {err_rets:.2e}"

    ret_par = retnet_parallel_forward(golden["q"], golden["k"], golden["v"], gamma)
    err_par = float(jnp.max(jnp.abs(ret_par - golden["ret_parallel_out"])))
    assert err_par < 1e-5, f"retnet_parallel_out max abs error {err_par:.2e}"

    # 3. DeltaNet
    del_out, del_s = deltanet_forward(golden["q"], golden["k"], golden["v"], golden["beta"])
    err_del = float(jnp.max(jnp.abs(del_out - golden["delta_out"])))
    err_dels = float(jnp.max(jnp.abs(del_s - golden["delta_final_s"])))
    assert err_del < 1e-5, f"delta_out max abs error {err_del:.2e}"
    assert err_dels < 1e-5, f"delta_final_s max abs error {err_dels:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
