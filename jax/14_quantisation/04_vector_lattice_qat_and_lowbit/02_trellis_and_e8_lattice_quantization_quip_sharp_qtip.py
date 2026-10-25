"""Trellis and E8 Lattice Quantization QuIP# QTIP — functional JAX implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip. Run ``python jax/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def project_d8(x: jnp.ndarray) -> jnp.ndarray:
    """Conway & Sloane fast projection onto the D8 checkerboard lattice."""
    f = jnp.round(x)
    s = jnp.sum(f, axis=-1, keepdims=True).astype(jnp.int32)
    is_odd = (s % 2 != 0)

    diff = x - f
    abs_diff = jnp.abs(diff)
    max_idx = jnp.argmax(abs_diff, axis=-1, keepdims=True)

    delta = jnp.where(diff >= 0.0, 1.0, -1.0)
    flip_mask = jnp.zeros_like(x)
    # Put 1.0 at max_idx
    flip_mask = jnp.take_along_axis(flip_mask, max_idx, axis=-1)
    # Using one_hot for flip_mask
    one_hot = (jnp.arange(8) == max_idx).astype(jnp.float32)

    f_adjusted = jnp.where(is_odd & (one_hot > 0.0), f + delta, f)
    return f_adjusted


def project_e8(x: jnp.ndarray) -> jnp.ndarray:
    """Fast nearest-point decoding for the 8D Gosset lattice E8."""
    p0 = project_d8(x)
    dist0 = jnp.sum((x - p0) ** 2, axis=-1, keepdims=True)

    p1 = project_d8(x - 0.5) + 0.5
    dist1 = jnp.sum((x - p1) ** 2, axis=-1, keepdims=True)

    return jnp.where(dist0 <= dist1, p0, p1)


def quantize_e8_lattice(vectors: jnp.ndarray, scale: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Quantize vectors in R^8 to E8 Gosset lattice scaled by step size `scale`."""
    x_scaled = vectors / scale
    lattice_points = project_e8(x_scaled)
    w_deq = lattice_points * scale
    return w_deq, lattice_points


def trellis_coded_quantize_1d(
    x: jnp.ndarray,
    codebook_subsets: jnp.ndarray,
    transitions: jnp.ndarray,
) -> jnp.ndarray:
    """Trellis-Coded Quantization (TCQ) forward Viterbi decoding over 1D sequence."""
    l = x.shape[0]
    n_states = transitions.shape[0]
    inf = 1e9

    dp = [[inf] * n_states for _ in range(l)]
    back_state = [[0] * n_states for _ in range(l)]
    back_level = [[0.0] * n_states for _ in range(l)]

    x_list = [float(v) for v in x]
    subsets_list = [[float(v) for v in codebook_subsets[s]] for s in range(n_states)]
    trans_list = [[int(transitions[s, 0]), int(transitions[s, 1])] for s in range(n_states)]

    for s in range(n_states):
        best_cost = inf
        best_lvl = 0.0
        for lvl in subsets_list[s]:
            cost = (x_list[0] - lvl) ** 2
            if cost < best_cost:
                best_cost = cost
                best_lvl = lvl
        dp[0][s] = best_cost
        back_level[0][s] = best_lvl

    for t in range(1, l):
        for s in range(n_states):
            for bit in (0, 1):
                next_s = trans_list[s][bit]
                for lvl in subsets_list[next_s]:
                    cost = dp[t - 1][s] + (x_list[t] - lvl) ** 2
                    if cost < dp[t][next_s]:
                        dp[t][next_s] = cost
                        back_state[t][next_s] = s
                        back_level[t][next_s] = lvl

    recon = [0.0] * l
    best_last = 0
    min_final = inf
    for s in range(n_states):
        if dp[-1][s] < min_final:
            min_final = dp[-1][s]
            best_last = s

    curr_s = best_last
    for t in range(l - 1, -1, -1):
        recon[t] = back_level[t][curr_s]
        curr_s = back_state[t][curr_s]

    return jnp.array(recon, dtype=jnp.float32)


def e8_forward(
    x: jnp.ndarray,
    w_orig: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inference forward pass using E8 lattice quantized weight matrix."""
    d_in, d_out = w_orig.shape
    w_vecs = w_orig.reshape((-1, 8))
    w_deq_vecs, _ = quantize_e8_lattice(w_vecs, scale)
    w_deq = w_deq_vecs.reshape((d_in, d_out))
    y = x @ w_deq
    if bias is not None:
        y = y + bias
    return y, w_deq


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor parity
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. E8 Lattice quantization parity
    w_e8_deq, lattice_points = quantize_e8_lattice(g["w_vecs"], g["e8_scale"])
    err_deq = float(jnp.max(jnp.abs(w_e8_deq - g["w_e8_deq"])))
    err_pts = float(jnp.max(jnp.abs(lattice_points - g["lattice_points"])))
    assert err_deq < 1e-5, f"E8 dequant error {err_deq:.2e}"
    assert err_pts < 1e-5, f"E8 lattice points error {err_pts:.2e}"

    # 3. Trellis coded quantization parity
    tcq_recon = trellis_coded_quantize_1d(g["tcq_x"], g["tcq_subsets"], g["tcq_transitions"])
    err_tcq = float(jnp.max(jnp.abs(tcq_recon - g["tcq_recon"])))
    assert err_tcq < 1e-5, f"TCQ recon error {err_tcq:.2e}"

    # 4. E8 Forward parity
    out_e8, w_full = e8_forward(g["x_calib"], g["w_orig"], g["e8_scale"], g["b_orig"])
    err_full = float(jnp.max(jnp.abs(w_full - g["w_deq_full"])))
    err_out = float(jnp.max(jnp.abs(out_e8 - g["out_e8"])))
    assert err_full < 1e-5, f"E8 full weight error {err_full:.2e}"
    assert err_out < 1e-5, f"E8 output error {err_out:.2e}"

    print("14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
