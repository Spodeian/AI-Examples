"""Calibration Observers Minmax Percentile Mse KL and Aciq — JAX implementation.

Topic: 14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq
Golden weights: weights/14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq.safetensors
"""

from __future__ import annotations

from pathlib import Path
import jax.numpy as jnp
from safetensors.torch import load_file

TOPIC = "14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(x, weight) + bias


def quantize_dequantize_symmetric(x: jnp.ndarray, alpha: float, qmax: float = 7.0) -> jnp.ndarray:
    if alpha == 0.0:
        return jnp.zeros_like(x)
    scale = alpha / qmax
    q = jnp.clip(jnp.round(x / scale), -qmax, qmax)
    return q * scale


def minmax_observer(x: jnp.ndarray, qmax: float = 7.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    min_val = float(jnp.min(x))
    max_val = float(jnp.max(x))
    amax = float(jnp.max(jnp.abs(x)))
    dequant = quantize_dequantize_symmetric(x, amax, qmax)
    stats = jnp.array([min_val, max_val, amax], dtype=jnp.float32)
    return dequant, stats


def moving_average_minmax_observer(
    batches: jnp.ndarray,
    momentum: float = 0.1,
) -> jnp.ndarray:
    running_min = float(jnp.min(batches[0]))
    running_max = float(jnp.max(batches[0]))
    for b in range(1, batches.shape[0]):
        batch_min = float(jnp.min(batches[b]))
        batch_max = float(jnp.max(batches[b]))
        running_min = (1.0 - momentum) * running_min + momentum * batch_min
        running_max = (1.0 - momentum) * running_max + momentum * batch_max
    return jnp.array([running_min, running_max], dtype=jnp.float32)


def percentile_observer(
    x: jnp.ndarray,
    percentile: float = 0.95,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    sorted_abs = jnp.sort(jnp.abs(x.flatten()))
    k = int(percentile * float(len(sorted_abs) - 1))
    alpha = float(sorted_abs[k])
    dequant = quantize_dequantize_symmetric(x, alpha, qmax)
    return dequant, jnp.array([alpha], dtype=jnp.float32)


def mse_observer(
    x: jnp.ndarray,
    num_candidates: int = 10,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    amax = float(jnp.max(jnp.abs(x)))
    best_mse = float("inf")
    best_alpha = amax
    for k in range(num_candidates):
        candidate = amax * (0.55 + 0.05 * float(k))
        cand_dequant = quantize_dequantize_symmetric(x, candidate, qmax)
        mse = float(jnp.mean((x - cand_dequant) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = candidate
    best_dequant = quantize_dequantize_symmetric(x, best_alpha, qmax)
    return best_dequant, jnp.array([best_alpha], dtype=jnp.float32)


def kl_divergence_observer(
    x: jnp.ndarray,
    num_bins: int = 64,
    num_quant_bins: int = 8,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_abs = jnp.abs(x.flatten())
    amax = float(jnp.max(x_abs))
    bin_width = amax / float(num_bins)
    bin_idx = jnp.clip((x_abs / bin_width).astype(jnp.int32), 0, num_bins - 1)
    hist = jnp.bincount(bin_idx, length=num_bins).astype(jnp.float32)

    best_kl = float("inf")
    best_t = num_bins

    candidates = list(range(num_quant_bins, num_bins + 1, 8))
    for T in candidates:
        P = hist[:T]
        P = P.at[-1].add(jnp.sum(hist[T:]))

        bin_size = T // num_quant_bins
        Q_chunks = []
        for k in range(num_quant_bins):
            start = k * bin_size
            end = (k + 1) * bin_size
            chunk = P[start:end]
            nonzero = float(jnp.sum((chunk > 0).astype(jnp.float32)))
            if nonzero > 0.0:
                mean_val = jnp.sum(chunk) / nonzero
                q_sub = jnp.where(chunk > 0, mean_val, 0.0)
            else:
                q_sub = jnp.zeros_like(chunk)
            Q_chunks.append(q_sub)
        Q = jnp.concatenate(Q_chunks)

        P_smooth = P + 1e-7
        Q_smooth = Q + 1e-7
        p_norm = P_smooth / jnp.sum(P_smooth)
        q_norm = Q_smooth / jnp.sum(Q_smooth)
        kl = float(jnp.sum(p_norm * jnp.log(p_norm / q_norm)))
        if kl < best_kl:
            best_kl = kl
            best_t = T

    best_alpha = (float(best_t) / float(num_bins)) * amax
    dequant = quantize_dequantize_symmetric(x, best_alpha, qmax)
    return dequant, jnp.array([best_alpha], dtype=jnp.float32)


def aciq_gaussian_observer(
    x: jnp.ndarray,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    beta = 2.8279
    mu = float(jnp.mean(x))
    sigma = float(jnp.sqrt(jnp.mean((x - mu) ** 2)))
    alpha = beta * sigma
    dequant = quantize_dequantize_symmetric(x, alpha, qmax)
    return dequant, jnp.array([alpha], dtype=jnp.float32)


def aciq_laplace_observer(
    x: jnp.ndarray,
    qmax: float = 7.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    beta = 3.89
    mu = float(jnp.mean(x))
    b = float(jnp.mean(jnp.abs(x - mu)))
    alpha = beta * b
    dequant = quantize_dequantize_symmetric(x, alpha, qmax)
    return dequant, jnp.array([alpha], dtype=jnp.float32)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. MinMax Observer
    minmax_deq, minmax_s = minmax_observer(g["raw_act"])
    assert float(jnp.max(jnp.abs(minmax_deq - g["minmax_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(minmax_s - g["minmax_stats"]))) < 1e-5

    # 3. Moving Average MinMax
    mov_s = moving_average_minmax_observer(g["raw_batches"], momentum=0.1)
    assert float(jnp.max(jnp.abs(mov_s - g["moving_avg_stats"]))) < 1e-5

    # 4. Percentile Observer
    perc_deq, perc_a = percentile_observer(g["raw_act"], percentile=0.95)
    assert float(jnp.max(jnp.abs(perc_deq - g["percentile_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(perc_a - g["percentile_alpha"]))) < 1e-5

    # 5. MSE Observer
    mse_deq, mse_a = mse_observer(g["raw_act"], num_candidates=10)
    assert float(jnp.max(jnp.abs(mse_deq - g["mse_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(mse_a - g["mse_alpha"]))) < 1e-5

    # 6. KL Divergence Observer
    kl_deq, kl_a = kl_divergence_observer(g["raw_act"], num_bins=64, num_quant_bins=8)
    assert float(jnp.max(jnp.abs(kl_deq - g["kl_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(kl_a - g["kl_alpha"]))) < 1e-5

    # 7. ACIQ Gaussian
    ag_deq, ag_a = aciq_gaussian_observer(g["raw_act"])
    assert float(jnp.max(jnp.abs(ag_deq - g["aciq_gauss_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(ag_a - g["aciq_gauss_alpha"]))) < 1e-5

    # 8. ACIQ Laplace
    al_deq, al_a = aciq_laplace_observer(g["raw_laplace"])
    assert float(jnp.max(jnp.abs(al_deq - g["aciq_laplace_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(al_a - g["aciq_laplace_alpha"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
