"""AWQ Activation Aware Weight Quantization — functional JAX implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization. Run ``python jax/14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def rtn_quantize_per_column(
    weight: jnp.ndarray,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Standard Round-To-Nearest (RTN) per-column symmetric quantization."""
    scale = jnp.max(jnp.abs(weight), axis=0, keepdims=True) / qmax
    scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    q = jnp.clip(jnp.round(weight / scale), qmin, qmax)
    dequant = q * scale
    return dequant, scale


def awq_compute_channel_scales(
    s_act: jnp.ndarray,
    s_w: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """Computes AWQ per-channel protection scales: s = (s_act^alpha) / (s_w^(1 - alpha))."""
    eps = 1e-6
    s_act_safe = jnp.clip(s_act, a_min=eps)
    s_w_safe = jnp.clip(s_w, a_min=eps)
    s = (s_act_safe ** alpha) / (s_w_safe ** (1.0 - alpha))
    norm = jnp.sqrt(jnp.max(s) * jnp.min(s) + eps)
    s_normalized = s / norm
    return s_normalized


def awq_search_optimal_scale(
    x_calib: jnp.ndarray,
    weight: jnp.ndarray,
    alpha_grid: list[float] | None = None,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, float, jnp.ndarray]:
    """Searches optimal AWQ alpha over candidate grid, minimizing output MSE."""
    if alpha_grid is None:
        alpha_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    s_act = jnp.mean(jnp.abs(x_calib), axis=0)
    s_w = jnp.max(jnp.abs(weight), axis=1)

    y_ref = x_calib @ weight
    best_loss = float("inf")
    best_alpha = 0.5
    best_scale = s_act
    best_w_eff = weight

    for alpha in alpha_grid:
        s = awq_compute_channel_scales(s_act, s_w, alpha)
        w_scaled = weight * jnp.expand_dims(s, 1)
        w_scaled_q, _ = rtn_quantize_per_column(w_scaled, qmin=qmin, qmax=qmax)
        w_eff = w_scaled_q / jnp.expand_dims(s, 1)

        y_pred = x_calib @ w_eff
        loss = float(jnp.mean((y_ref - y_pred) ** 2))

        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_scale = s
            best_w_eff = w_eff

    return best_scale, best_alpha, best_w_eff


def awq_quantize_with_scale(
    weight: jnp.ndarray,
    scale: jnp.ndarray,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    w_scaled = weight * jnp.expand_dims(scale, 1)
    w_scaled_q, col_scales = rtn_quantize_per_column(w_scaled, qmin=qmin, qmax=qmax)
    w_eff = w_scaled_q / jnp.expand_dims(scale, 1)
    return w_eff, col_scales


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. RTN baseline
    w_rtn, col_scales = rtn_quantize_per_column(g["w_orig"], qmin=-8.0, qmax=7.0)
    err_rtn = float(jnp.max(jnp.abs(w_rtn - g["w_rtn"])))
    assert err_rtn < 1e-5, f"RTN weight error {err_rtn:.2e}"

    # 3. Channel norms
    s_act = jnp.mean(jnp.abs(g["x_calib"]), axis=0)
    s_w = jnp.max(jnp.abs(g["w_orig"]), axis=1)
    assert float(jnp.max(jnp.abs(s_act - g["s_act"]))) < 1e-5
    assert float(jnp.max(jnp.abs(s_w - g["s_w"]))) < 1e-5

    # 4. AWQ search
    alpha_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    best_scale, best_alpha, w_awq = awq_search_optimal_scale(
        g["x_calib"], g["w_orig"], alpha_grid=alpha_grid, qmin=-8.0, qmax=7.0
    )
    assert abs(best_alpha - float(g["best_alpha"][0])) < 1e-5
    assert float(jnp.max(jnp.abs(best_scale - g["best_scale"]))) < 1e-5
    assert float(jnp.max(jnp.abs(w_awq - g["w_awq"]))) < 1e-5

    # 5. Output activation parity
    out_awq = g["x_calib"][:8] @ w_awq + g["b_orig"]
    assert float(jnp.max(jnp.abs(out_awq - g["out_awq"]))) < 1e-5

    print("14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
