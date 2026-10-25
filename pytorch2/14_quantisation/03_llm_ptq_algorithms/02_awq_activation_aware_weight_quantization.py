"""AWQ Activation Aware Weight Quantization — PyTorch 2.x reference implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization
Golden weights: weights/14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization.safetensors

``python pytorch2/14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def rtn_quantize_per_column(
    weight: torch.Tensor,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard Round-To-Nearest (RTN) per-column symmetric quantization."""
    scale = torch.max(torch.abs(weight), dim=0, keepdim=True).values / qmax
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(weight / scale), qmin, qmax)
    dequant = q * scale
    return dequant, scale


def awq_compute_channel_scales(
    s_act: torch.Tensor,
    s_w: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Computes AWQ per-channel protection scales: s = (s_act^alpha) / (s_w^(1 - alpha))."""
    eps = 1e-6
    s_act_safe = torch.clamp(s_act, min=eps)
    s_w_safe = torch.clamp(s_w, min=eps)
    s = (s_act_safe ** alpha) / (s_w_safe ** (1.0 - alpha))
    norm = torch.sqrt(torch.max(s) * torch.min(s) + eps)
    s_normalized = s / norm
    return s_normalized


def awq_search_optimal_scale(
    x_calib: torch.Tensor,
    weight: torch.Tensor,
    alpha_grid: list[float] | None = None,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Searches optimal AWQ alpha over candidate grid, minimizing output MSE."""
    if alpha_grid is None:
        alpha_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    s_act = torch.mean(torch.abs(x_calib), dim=0)  # [d_in]
    s_w = torch.max(torch.abs(weight), dim=1).values  # [d_in]

    y_ref = x_calib @ weight
    best_loss = float("inf")
    best_alpha = 0.5
    best_scale = torch.ones_like(s_act)
    best_w_eff = weight.clone()

    for alpha in alpha_grid:
        s = awq_compute_channel_scales(s_act, s_w, alpha)
        # Scale weights along input channels (rows)
        w_scaled = weight * s.unsqueeze(1)
        # Quantize scaled weights per column
        w_scaled_q, _ = rtn_quantize_per_column(w_scaled, qmin=qmin, qmax=qmax)
        # Dequantize and undo channel protection scaling
        w_eff = w_scaled_q / s.unsqueeze(1)

        y_pred = x_calib @ w_eff
        loss = float(torch.mean((y_ref - y_pred) ** 2))

        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_scale = s
            best_w_eff = w_eff

    return best_scale, best_alpha, best_w_eff


def awq_quantize_with_scale(
    weight: torch.Tensor,
    scale: torch.Tensor,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies known AWQ channel scales, quantizes, and produces effective weights."""
    w_scaled = weight * scale.unsqueeze(1)
    w_scaled_q, col_scales = rtn_quantize_per_column(w_scaled, qmin=qmin, qmax=qmax)
    w_eff = w_scaled_q / scale.unsqueeze(1)
    return w_eff, col_scales


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Calibration activations and weight with prominent outlier channels
    n_samples, d_in, d_out = 64, 16, 8
    x_calib = torch.randn(n_samples, d_in)
    # Channel 0 and 1 are emergent outlier channels (12x and 6x larger)
    x_calib[:, 0] *= 12.0
    x_calib[:, 1] *= 6.0

    w_orig = torch.randn(d_in, d_out) * 0.25
    b_orig = torch.randn(d_out) * 0.1

    # 1. Unscaled RTN baseline
    w_rtn, rtn_col_scales = rtn_quantize_per_column(w_orig, qmin=-8.0, qmax=7.0)
    out_rtn = x_calib[:8] @ w_rtn + b_orig

    # 2. AWQ search and optimal scaling
    alpha_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    best_scale, best_alpha, w_awq = awq_search_optimal_scale(
        x_calib, w_orig, alpha_grid=alpha_grid, qmin=-8.0, qmax=7.0
    )
    out_awq = x_calib[:8] @ w_awq + b_orig

    # 3. Channel activation norms
    s_act = torch.mean(torch.abs(x_calib), dim=0)
    s_w = torch.max(torch.abs(w_orig), dim=1).values

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "s_act": s_act,
        "s_w": s_w,
        "w_rtn": w_rtn,
        "rtn_col_scales": rtn_col_scales,
        "out_rtn": out_rtn,
        "best_scale": best_scale,
        "best_alpha": torch.tensor([best_alpha], dtype=torch.float32),
        "w_awq": w_awq,
        "out_awq": out_awq,
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
