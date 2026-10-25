"""Calibration Observers Minmax Percentile Mse KL and Aciq — PyTorch 2.x reference implementation.

Topic: 14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq
Golden weights: weights/14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq.safetensors

``python pytorch2/14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def quantize_dequantize_symmetric(x: torch.Tensor, alpha: float, qmax: float = 7.0) -> torch.Tensor:
    """Symmetric uniform quantize and dequantize with scale = alpha / qmax."""
    if alpha == 0.0:
        return torch.zeros_like(x)
    scale = alpha / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale


def minmax_observer(x: torch.Tensor, qmax: float = 7.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes exact running min, max, absolute max, and dequantized tensor."""
    min_val = float(torch.min(x))
    max_val = float(torch.max(x))
    amax = float(torch.max(torch.abs(x)))
    dequant = quantize_dequantize_symmetric(x, amax, qmax)
    stats = torch.tensor([min_val, max_val, amax], dtype=torch.float32)
    return dequant, stats


def moving_average_minmax_observer(
    batches: torch.Tensor,
    momentum: float = 0.1,
) -> torch.Tensor:
    """Computes exponential moving average of min and max across batches."""
    running_min = float(torch.min(batches[0]))
    running_max = float(torch.max(batches[0]))
    for b in range(1, batches.shape[0]):
        batch_min = float(torch.min(batches[b]))
        batch_max = float(torch.max(batches[b]))
        running_min = (1.0 - momentum) * running_min + momentum * batch_min
        running_max = (1.0 - momentum) * running_max + momentum * batch_max
    return torch.tensor([running_min, running_max], dtype=torch.float32)


def percentile_observer(
    x: torch.Tensor,
    percentile: float = 0.95,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes clipping threshold from given percentile of absolute values."""
    sorted_abs, _ = torch.sort(torch.abs(x.flatten()))
    k = int(percentile * float(len(sorted_abs) - 1))
    alpha = float(sorted_abs[k])
    dequant = quantize_dequantize_symmetric(x, alpha, qmax)
    return dequant, torch.tensor([alpha], dtype=torch.float32)


def mse_observer(
    x: torch.Tensor,
    num_candidates: int = 10,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid search over candidate thresholds to minimize reconstruction L2 error."""
    amax = float(torch.max(torch.abs(x)))
    best_mse = float("inf")
    best_alpha = amax
    for k in range(num_candidates):
        candidate = amax * (0.55 + 0.05 * float(k))
        cand_dequant = quantize_dequantize_symmetric(x, candidate, qmax)
        mse = float(torch.mean((x - cand_dequant) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = candidate
    best_dequant = quantize_dequantize_symmetric(x, best_alpha, qmax)
    return best_dequant, torch.tensor([best_alpha], dtype=torch.float32)


def kl_divergence_observer(
    x: torch.Tensor,
    num_bins: int = 64,
    num_quant_bins: int = 8,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TensorRT-style entropy calibration via discrete histogram KL divergence."""
    x_abs = torch.abs(x.flatten())
    amax = float(torch.max(x_abs))
    bin_width = amax / float(num_bins)
    bin_idx = torch.clamp((x_abs / bin_width).long(), 0, num_bins - 1)
    hist = torch.bincount(bin_idx, minlength=num_bins).float()

    best_kl = float("inf")
    best_t = num_bins

    # Step through candidate thresholds from num_quant_bins to num_bins in steps of 8
    candidates = list(range(num_quant_bins, num_bins + 1, 8))
    for T in candidates:
        P = hist[:T].clone()
        P[-1] += torch.sum(hist[T:])

        bin_size = T // num_quant_bins
        Q = torch.zeros_like(P)
        for k in range(num_quant_bins):
            start = k * bin_size
            end = (k + 1) * bin_size
            chunk = P[start:end]
            nonzero = float(torch.sum((chunk > 0).float()))
            if nonzero > 0.0:
                mean_val = torch.sum(chunk) / nonzero
                Q[start:end] = torch.where(chunk > 0, mean_val, 0.0)

        P_smooth = P + 1e-7
        Q_smooth = Q + 1e-7
        p_norm = P_smooth / torch.sum(P_smooth)
        q_norm = Q_smooth / torch.sum(Q_smooth)
        kl = float(torch.sum(p_norm * torch.log(p_norm / q_norm)))
        if kl < best_kl:
            best_kl = kl
            best_t = T

    best_alpha = (float(best_t) / float(num_bins)) * amax
    dequant = quantize_dequantize_symmetric(x, best_alpha, qmax)
    return dequant, torch.tensor([best_alpha], dtype=torch.float32)


def aciq_gaussian_observer(
    x: torch.Tensor,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical Clipping for Integer Quantization (ACIQ) on Gaussian distributions."""
    # Optimal constant for 4-bit (16 levels, 7 positive levels) Gaussian: beta = 2.8279
    beta = 2.8279
    mu = float(torch.mean(x))
    sigma = float(torch.sqrt(torch.mean((x - mu) ** 2)))
    alpha = beta * sigma
    dequant = quantize_dequantize_symmetric(x, alpha, qmax)
    return dequant, torch.tensor([alpha], dtype=torch.float32)


def aciq_laplace_observer(
    x: torch.Tensor,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical Clipping for Integer Quantization (ACIQ) on Laplace distributions."""
    # Optimal constant for 4-bit Laplace: beta = 3.89
    beta = 3.89
    mu = float(torch.mean(x))
    b = float(torch.mean(torch.abs(x - mu)))
    alpha = beta * b
    dequant = quantize_dequantize_symmetric(x, alpha, qmax)
    return dequant, torch.tensor([alpha], dtype=torch.float32)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Standard Gaussian activations
    raw_act = torch.randn(256) * 1.5

    # 2. MinMax Observer
    minmax_dequant, minmax_stats = minmax_observer(raw_act)

    # 3. Moving Average MinMax Observer
    raw_batches = torch.randn(3, 64) * 1.2
    moving_avg_stats = moving_average_minmax_observer(raw_batches, momentum=0.1)

    # 4. Percentile Observer (95%)
    percentile_dequant, percentile_alpha = percentile_observer(raw_act, percentile=0.95)

    # 5. MSE Observer (Grid search)
    mse_dequant, mse_alpha = mse_observer(raw_act, num_candidates=10)

    # 6. KL Divergence Observer (TensorRT entropy)
    kl_dequant, kl_alpha = kl_divergence_observer(raw_act, num_bins=64, num_quant_bins=8)

    # 7. ACIQ Gaussian Observer
    aciq_gauss_dequant, aciq_gauss_alpha = aciq_gaussian_observer(raw_act)

    # 8. ACIQ Laplace Observer
    u = torch.rand(256) - 0.5
    raw_laplace = -torch.sign(u) * torch.log(1.0 - 2.0 * torch.abs(u)) * 0.8
    aciq_laplace_dequant, aciq_laplace_alpha = aciq_laplace_observer(raw_laplace)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "raw_act": raw_act,
        "minmax_dequant": minmax_dequant,
        "minmax_stats": minmax_stats,
        "raw_batches": raw_batches,
        "moving_avg_stats": moving_avg_stats,
        "percentile_dequant": percentile_dequant,
        "percentile_alpha": percentile_alpha,
        "mse_dequant": mse_dequant,
        "mse_alpha": mse_alpha,
        "kl_dequant": kl_dequant,
        "kl_alpha": kl_alpha,
        "aciq_gauss_dequant": aciq_gauss_dequant,
        "aciq_gauss_alpha": aciq_gauss_alpha,
        "raw_laplace": raw_laplace,
        "aciq_laplace_dequant": aciq_laplace_dequant,
        "aciq_laplace_alpha": aciq_laplace_alpha,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
