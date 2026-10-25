"""Mamba Selective SSM and Mamba2 SSD — functional JAX implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd.
Run ``python jax/05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd.safetensors"


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def mamba_selective_forward(
    x_seq: jnp.ndarray,
    a_log: jnp.ndarray,
    w_b: jnp.ndarray,
    w_c: jnp.ndarray,
    w_delta: jnp.ndarray,
    b_delta: jnp.ndarray,
    d_skip: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Selective State Space Model (Mamba-1) recurrent unrolling in JAX."""
    B, T, D_model = x_seq.shape
    D_state = a_log.shape[1]

    a_real = -jnp.exp(a_log)  # [D_model, D_state]
    h_t = jnp.zeros((B, D_model, D_state), dtype=x_seq.dtype)
    outs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        b_t = xt @ w_b.T
        c_t = xt @ w_c.T
        delta_t = _softplus(xt @ w_delta.T + b_delta)

        dt_a = jnp.expand_dims(delta_t, axis=-1) * jnp.expand_dims(a_real, axis=0)
        a_bar = jnp.exp(dt_a)

        b_bar = jnp.expand_dims(delta_t, axis=-1) * jnp.expand_dims(b_t, axis=1)

        h_t = a_bar * h_t + b_bar * jnp.expand_dims(xt, axis=-1)
        y_t = jnp.sum(h_t * jnp.expand_dims(c_t, axis=1), axis=-1) + xt * d_skip
        outs.append(jnp.expand_dims(y_t, axis=1))

    seq_out = jnp.concatenate(outs, axis=1)
    return seq_out, h_t


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    seq_out, final_h = mamba_selective_forward(
        g["x_seq"],
        g["a_log"],
        g["w_b"],
        g["w_c"],
        g["w_delta"],
        g["b_delta"],
        g["d_skip"],
    )
    err_seq = float(jnp.max(jnp.abs(seq_out - g["seq_out"])))
    assert err_seq < 1e-5, f"Mamba sequence error {err_seq:.2e}"

    print("05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
