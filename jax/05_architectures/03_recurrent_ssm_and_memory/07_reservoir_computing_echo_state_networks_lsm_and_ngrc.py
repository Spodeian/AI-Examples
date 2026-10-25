"""Reservoir Computing, Echo State Networks, LSM, and NG-RC — JAX reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def esn_forward(
    x_seq: jnp.ndarray,
    w_in: jnp.ndarray,
    w_res: jnp.ndarray,
    b_res: jnp.ndarray,
    alpha: float,
    w_out: jnp.ndarray,
    b_out: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Leaky Echo State Network recurrent forward pass in JAX."""
    B, T, D_in = x_seq.shape
    D_res = w_in.shape[0]

    h_t = jnp.zeros((B, D_res), dtype=x_seq.dtype)
    outs = []

    for t in range(T):
        xt = x_seq[:, t, :]
        pre_act = xt @ w_in.T + h_t @ w_res.T + b_res
        tilde_h = jnp.tanh(pre_act)
        h_t = (1.0 - alpha) * h_t + alpha * tilde_h
        yt = h_t @ w_out.T + b_out
        outs.append(jnp.expand_dims(yt, 1))

    seq_out = jnp.concatenate(outs, axis=1)
    return seq_out, h_t


def ngrc_features(x_curr: jnp.ndarray, x_prev: jnp.ndarray) -> jnp.ndarray:
    """Next-Generation Reservoir Computing (NG-RC) nonlinear feature expansion in JAX."""
    lin = jnp.concatenate([x_curr, x_prev], axis=-1)
    quad = jnp.einsum("bi,bj->bij", x_curr, x_prev).reshape(x_curr.shape[0], -1)
    return jnp.concatenate([lin, quad], axis=-1)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. ESN
    alpha = float(golden["alpha"][0])
    seq_out, final_h = esn_forward(
        golden["x_seq"], golden["w_in"], golden["w_res"], golden["b_res"],
        alpha, golden["w_out"], golden["b_out"]
    )
    err_s = float(jnp.max(jnp.abs(seq_out - golden["seq_out"])))
    err_h = float(jnp.max(jnp.abs(final_h - golden["final_h"])))
    assert err_s < 1e-5, f"esn seq_out max abs error {err_s:.2e}"
    assert err_h < 1e-5, f"esn final_h max abs error {err_h:.2e}"

    # 3. NG-RC
    feat = ngrc_features(golden["x_curr"], golden["x_prev"])
    err_f = float(jnp.max(jnp.abs(feat - golden["ngrc_feat"])))
    assert err_f < 1e-5, f"ngrc_feat max abs error {err_f:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
