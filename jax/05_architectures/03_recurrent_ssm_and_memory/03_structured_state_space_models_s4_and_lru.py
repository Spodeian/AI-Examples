"""Structured State Space Models S4 and LRU — functional JAX implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru.
Run ``python jax/05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def lru_recurrent_forward(
    x_seq: jnp.ndarray,
    nu: jnp.ndarray,
    theta: jnp.ndarray,
    b_re: jnp.ndarray,
    b_im: jnp.ndarray,
    c_re: jnp.ndarray,
    c_im: jnp.ndarray,
    d_skip: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linear Recurrent Unit (LRU) recurrent forward pass in JAX."""
    B, T, D_in = x_seq.shape
    D_state = nu.shape[0]

    diag_mag = jnp.exp(-jnp.clip(nu, a_min=1e-4))
    diag_re = diag_mag * jnp.cos(theta)
    diag_im = diag_mag * jnp.sin(theta)

    h_re = jnp.zeros((B, D_state), dtype=x_seq.dtype)
    h_im = jnp.zeros((B, D_state), dtype=x_seq.dtype)

    outs = []
    for t in range(T):
        xt = x_seq[:, t, :]
        bu_re = xt @ b_re.T
        bu_im = xt @ b_im.T

        h_re_next = diag_re * h_re - diag_im * h_im + bu_re
        h_im_next = diag_re * h_im + diag_im * h_re + bu_im

        h_re = h_re_next
        h_im = h_im_next

        y_t = h_re @ c_re.T - h_im @ c_im.T + xt * d_skip
        outs.append(jnp.expand_dims(y_t, axis=1))

    seq_out = jnp.concatenate(outs, axis=1)
    return seq_out, h_re, h_im


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    seq_out, h_re, h_im = lru_recurrent_forward(
        g["x_seq"],
        g["nu"],
        g["theta"],
        g["b_re"],
        g["b_im"],
        g["c_re"],
        g["c_im"],
        g["d_skip"],
    )
    err_seq = float(jnp.max(jnp.abs(seq_out - g["seq_out"])))
    assert err_seq < 1e-5, f"LRU sequence error {err_seq:.2e}"

    print("05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
