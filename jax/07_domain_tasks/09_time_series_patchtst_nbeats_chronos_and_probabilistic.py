"""Time Series Patchtst Nbeats Chronos and Probabilistic — JAX reference implementation.

Topic: 07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic
Golden weights: weights/07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def revin_normalize(
    x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reversible Instance Normalization (RevIN) forward normalization in JAX."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    stdev = jnp.sqrt(var + eps)
    x_norm = ((x - mean) / stdev) * gamma + beta
    return x_norm, mean, stdev


def revin_denormalize(
    y: jnp.ndarray, mean: jnp.ndarray, stdev: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray
) -> jnp.ndarray:
    """RevIN reverse denormalization on forecasted horizon."""
    return ((y - beta) / gamma) * stdev + mean


def patchtst_patching(x: jnp.ndarray, patch_len: int, stride: int) -> jnp.ndarray:
    """Extracts overlapping 1D patches from lookback time series."""
    b, m, l = x.shape
    n_patches = (l - patch_len) // stride + 1
    patches_list = [x[:, :, i * stride : i * stride + patch_len] for i in range(n_patches)]
    # Stack along dimension 2 -> [B, M, N_patches, patch_len]
    stacked = jnp.stack(patches_list, axis=2)
    return stacked.reshape(b * m, n_patches, patch_len)


def nbeats_block(
    x: jnp.ndarray,
    fc_weights: list[jnp.ndarray],
    fc_biases: list[jnp.ndarray],
    theta_b_weight: jnp.ndarray,
    theta_f_weight: jnp.ndarray,
    basis_b: jnp.ndarray,
    basis_f: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """N-BEATS generic doubly-residual block in JAX."""
    h = x
    for w, b in zip(fc_weights, fc_biases):
        h = jax.nn.relu(h @ w + b)
    theta_b = h @ theta_b_weight
    theta_f = h @ theta_f_weight
    backcast = theta_b @ basis_b
    forecast = theta_f @ basis_f
    return backcast, forecast


def pinball_quantile_loss(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, quantiles: jnp.ndarray
) -> jnp.ndarray:
    """Computes pinball (quantile) loss across multiple quantiles in JAX."""
    err = jnp.expand_dims(y_true, axis=1) - y_pred
    q = quantiles.reshape(1, -1, 1)
    loss = jnp.maximum(q * err, (q - 1.0) * err)
    return jnp.mean(loss)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    raw = load_file(str(path))
    tensors = {k: jnp.array(v) for k, v in raw.items()}

    # Anchor check
    out = linear_anchor(tensors["test_input"], tensors["weight"], tensors["bias"])
    err = float(jnp.max(jnp.abs(out - tensors["test_output"])))
    assert err < 1e-5, f"test_output parity error: {err}"

    # 1. RevIN
    norm_x, mean, stdev = revin_normalize(
        tensors["revin_x"], tensors["revin_gamma"], tensors["revin_beta"]
    )
    err_norm = float(jnp.max(jnp.abs(norm_x - tensors["revin_norm_x"])))
    assert err_norm < 1e-5, f"revin_norm_x parity error: {err_norm}"

    denorm_y = revin_denormalize(
        tensors["norm_y"], mean, stdev, tensors["revin_gamma"], tensors["revin_beta"]
    )
    err_denorm = float(jnp.max(jnp.abs(denorm_y - tensors["denorm_y_ref"])))
    assert err_denorm < 1e-5, f"denorm_y parity error: {err_denorm}"

    # 2. PatchTST patching
    patches = patchtst_patching(tensors["revin_x"], patch_len=8, stride=4)
    err_patches = float(jnp.max(jnp.abs(patches - tensors["patches_ref"])))
    assert err_patches < 1e-5, f"patches parity error: {err_patches}"

    # 3. N-BEATS block
    backcast, forecast = nbeats_block(
        tensors["nbeats_x"],
        [tensors["w_fc1"], tensors["w_fc2"]],
        [tensors["b_fc1"], tensors["b_fc2"]],
        tensors["w_theta_b"],
        tensors["w_theta_f"],
        tensors["basis_b"],
        tensors["basis_f"],
    )
    err_backcast = float(jnp.max(jnp.abs(backcast - tensors["backcast_ref"])))
    err_forecast = float(jnp.max(jnp.abs(forecast - tensors["forecast_ref"])))
    assert err_backcast < 1e-5, f"backcast parity error: {err_backcast}"
    assert err_forecast < 1e-5, f"forecast parity error: {err_forecast}"

    # 4. Pinball loss
    loss = pinball_quantile_loss(
        tensors["y_true"], tensors["y_pred_quantiles"], tensors["quantiles"]
    )
    err_loss = float(jnp.abs(loss - tensors["pinball_loss_ref"]))
    assert err_loss < 1e-5, f"pinball loss parity error: {err_loss}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
