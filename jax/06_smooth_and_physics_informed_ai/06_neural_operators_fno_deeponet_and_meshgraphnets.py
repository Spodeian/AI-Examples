"""Neural Operators FNO DeepONet and Meshgraphnets — functional JAX implementation.

Topic: 06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.
Run ``python jax/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def fno_spectral_conv_1d(
    v: jnp.ndarray,
    r_real: jnp.ndarray,
    r_imag: jnp.ndarray,
    s_len: int,
    k_modes: int,
) -> jnp.ndarray:
    k = jnp.arange(k_modes)[:, None]
    n = jnp.arange(s_len)[None, :]
    angle = 2.0 * jnp.pi * k * n / s_len

    c_mat = jnp.cos(angle)
    s_mat = -jnp.sin(angle)

    v_r = jnp.matmul(v, c_mat.T)
    v_i = jnp.matmul(v, s_mat.T)

    u_r = jnp.einsum("bck, koc -> bok", v_r, r_real) - jnp.einsum("bck, koc -> bok", v_i, r_imag)
    u_i = jnp.einsum("bck, koc -> bok", v_r, r_imag) + jnp.einsum("bck, koc -> bok", v_i, r_real)

    weights_k = jnp.array([1.0] + [2.0] * (k_modes - 1))[:, None]
    m_cos = jnp.cos(angle) * weights_k
    m_sin = jnp.sin(angle) * weights_k

    k_v = (1.0 / s_len) * (jnp.matmul(u_r, m_cos) - jnp.matmul(u_i, m_sin))
    return k_v


def compute_neural_operators_all(g: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    # 1. FNO
    s_len, k_modes = 8, 4
    fno_input = g["fno_input"]
    w_p = g["w_p"]
    b_p = g["b_p"]
    v0 = jax.nn.gelu(w_p @ fno_input + b_p, approximate=False)

    r_real = g["r_real"]
    r_imag = g["r_imag"]
    k_v0 = fno_spectral_conv_1d(v0, r_real, r_imag, s_len, k_modes)

    w_bypass = g["w_bypass"]
    bypass = jnp.einsum("oc, bcs -> bos", w_bypass, v0)
    v1 = jax.nn.gelu(k_v0 + bypass, approximate=False)

    w_q = g["w_q"]
    b_q = g["b_q"]
    fno_output = w_q @ v1 + b_q

    # 2. DeepONet
    u_sensor = g["u_sensor"]
    y_query = g["y_query"]
    w_b = g["w_b"]
    b_b = g["b_b"]
    w_t = g["w_t"]
    b_t = g["b_t"]
    b_out0 = g["b_out0"]

    branch_feat = jax.nn.gelu(u_sensor @ w_b + b_b, approximate=False)
    trunk_feat = jax.nn.gelu(y_query @ w_t + b_t, approximate=False)
    deeponet_output = branch_feat @ trunk_feat.T + b_out0

    # 3. MeshGraphNets
    v_nodes = g["v_nodes"]
    src = g["src_indices"]
    dst = g["dst_indices"]
    e_feat = g["e_feat"]
    w_e = g["w_e"]
    b_e = g["b_e"]
    w_v = g["w_v"]
    b_v = g["b_v"]

    e_in = jnp.concatenate([v_nodes[src], v_nodes[dst], e_feat], axis=-1)
    e_up = jax.nn.gelu(e_in @ w_e + b_e, approximate=False)

    e_agg = jnp.zeros((4, 4)).at[dst].add(e_up)
    v_in = jnp.concatenate([v_nodes, e_agg], axis=-1)
    mgn_output = jax.nn.gelu(v_in @ w_v + b_v, approximate=False)

    return {
        "v0": v0,
        "k_v0": k_v0,
        "v1": v1,
        "fno_output": fno_output,
        "branch_feat": branch_feat,
        "trunk_feat": trunk_feat,
        "deeponet_output": deeponet_output,
        "mgn_output": mgn_output,
    }


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    res = compute_neural_operators_all(g)
    for name, computed in res.items():
        diff = float(jnp.max(jnp.abs(computed - g[name])))
        rel = diff / (float(jnp.max(jnp.abs(g[name]))) + 1e-8)
        assert diff < 1e-5 or rel < 1e-5, f"{name}: max abs diff {diff:.2e}, rel diff {rel:.2e}"

    print("06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
