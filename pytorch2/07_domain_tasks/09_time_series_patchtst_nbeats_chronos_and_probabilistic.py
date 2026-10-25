"""Time Series Patchtst Nbeats Chronos and Probabilistic — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic
Golden weights: weights/07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic.safetensors

``python pytorch2/07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def revin_normalize(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reversible Instance Normalization (RevIN) forward normalization.
    x: [B, M, L] (Batch, Channels/Variates, Lookback)
    gamma, beta: [1, M, 1] learnable affine parameters
    Returns: (x_norm, mean, stdev)
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    stdev = torch.sqrt(var + eps)
    x_norm = ((x - mean) / stdev) * gamma + beta
    return x_norm, mean, stdev


def revin_denormalize(y: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """RevIN reverse denormalization on forecasted horizon.
    y: [B, M, H] (Batch, Channels/Variates, Horizon)
    """
    return ((y - beta) / gamma) * stdev + mean


def patchtst_patching(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """Extracts overlapping 1D patches from lookback time series.
    x: [B, M, L]
    Returns: [B * M, N_patches, patch_len]
    """
    b, m, l = x.shape
    # Unfold along time dimension
    # shape: [B, M, N_patches, patch_len]
    patches = x.unfold(dimension=-1, size=patch_len, step=stride)
    n_patches = patches.shape[2]
    # Merge batch and variate dimensions for channel independence: [B * M, N_patches, patch_len]
    return patches.contiguous().view(b * m, n_patches, patch_len)


def nbeats_block(
    x: torch.Tensor,
    fc_weights: list[torch.Tensor],
    fc_biases: list[torch.Tensor],
    theta_b_weight: torch.Tensor,
    theta_f_weight: torch.Tensor,
    basis_b: torch.Tensor,
    basis_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """N-BEATS generic doubly-residual block.
    x: [B, L]
    fc layers: 2 hidden layers with ReLU
    theta_b: [B, Theta_b] -> backcast: [B, L]
    theta_f: [B, Theta_f] -> forecast: [B, H]
    Returns: (backcast, forecast)
    """
    h = x
    for w, b in zip(fc_weights, fc_biases):
        h = F.relu(h @ w + b)
    theta_b = h @ theta_b_weight
    theta_f = h @ theta_f_weight
    backcast = theta_b @ basis_b
    forecast = theta_f @ basis_f
    return backcast, forecast


def pinball_quantile_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Computes pinball (quantile) loss across multiple quantiles.
    y_true: [B, H]
    y_pred: [B, Q, H]
    quantiles: [Q]
    Returns: scalar mean loss
    """
    # error: [B, Q, H]
    err = y_true.unsqueeze(1) - y_pred
    q = quantiles.view(1, -1, 1)
    loss = torch.maximum(q * err, (q - 1.0) * err)
    return loss.mean()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }

    # 1. RevIN normalization & denormalization
    b, m, l, h = 2, 3, 24, 8
    revin_x = torch.randn(b, m, l) * 2.5 + 10.0
    gamma = torch.ones(1, m, 1) + torch.randn(1, m, 1) * 0.1
    beta = torch.randn(1, m, 1) * 0.1
    norm_x, mean, stdev = revin_normalize(revin_x, gamma, beta)
    
    # Fake forecasted horizon
    norm_y = torch.randn(b, m, h)
    denorm_y = revin_denormalize(norm_y, mean, stdev, gamma, beta)

    tensors["revin_x"] = revin_x
    tensors["revin_gamma"] = gamma
    tensors["revin_beta"] = beta
    tensors["revin_norm_x"] = norm_x
    tensors["revin_mean"] = mean
    tensors["revin_stdev"] = stdev
    tensors["norm_y"] = norm_y
    tensors["denorm_y_ref"] = denorm_y

    # 2. PatchTST patching
    patch_len = 8
    stride = 4
    patches_ref = patchtst_patching(revin_x, patch_len, stride)
    tensors["patches_ref"] = patches_ref

    # 3. N-BEATS Block
    d_l, d_h, d_hid, d_theta = 24, 8, 32, 16
    nbeats_x = torch.randn(4, d_l)
    w_fc1 = torch.randn(d_l, d_hid) * 0.1
    b_fc1 = torch.randn(d_hid) * 0.1
    w_fc2 = torch.randn(d_hid, d_hid) * 0.1
    b_fc2 = torch.randn(d_hid) * 0.1
    w_theta_b = torch.randn(d_hid, d_theta) * 0.1
    w_theta_f = torch.randn(d_hid, d_theta) * 0.1
    basis_b = torch.randn(d_theta, d_l) * 0.1
    basis_f = torch.randn(d_theta, d_h) * 0.1

    backcast_ref, forecast_ref = nbeats_block(
        nbeats_x,
        [w_fc1, w_fc2],
        [b_fc1, b_fc2],
        w_theta_b,
        w_theta_f,
        basis_b,
        basis_f,
    )
    tensors["nbeats_x"] = nbeats_x
    tensors["w_fc1"] = w_fc1
    tensors["b_fc1"] = b_fc1
    tensors["w_fc2"] = w_fc2
    tensors["b_fc2"] = b_fc2
    tensors["w_theta_b"] = w_theta_b
    tensors["w_theta_f"] = w_theta_f
    tensors["basis_b"] = basis_b
    tensors["basis_f"] = basis_f
    tensors["backcast_ref"] = backcast_ref
    tensors["forecast_ref"] = forecast_ref

    # 4. Pinball Quantile Loss
    y_true = torch.randn(4, d_h)
    # y_pred for 3 quantiles: 0.1, 0.5, 0.9
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    y_pred_quantiles = torch.randn(4, 3, d_h)
    pinball_loss_ref = pinball_quantile_loss(y_true, y_pred_quantiles, quantiles)

    tensors["y_true"] = y_true
    tensors["quantiles"] = quantiles
    tensors["y_pred_quantiles"] = y_pred_quantiles
    tensors["pinball_loss_ref"] = pinball_loss_ref

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
