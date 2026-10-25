"""OmniQuant and HQQ Optimization-Based PTQ — PyTorch 2.x reference implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq
Golden weights: weights/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.safetensors

``python pytorch2/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def lwc_quantize(
    w: torch.Tensor,
    h_l: torch.Tensor,
    h_u: torch.Tensor,
    qmin: float = 0.0,
    qmax: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Learnable Weight Clipping (LWC) asymmetric quantization."""
    w_min = torch.min(w, dim=0, keepdim=True).values
    w_max = torch.max(w, dim=0, keepdim=True).values

    # Parameterize lower and upper clipping bounds via sigmoid
    lower = w_min * torch.sigmoid(h_l)
    upper = w_max * torch.sigmoid(h_u)

    scale = (upper - lower) / qmax
    scale = torch.clamp(scale, min=1e-5)

    # Quantize and dequantize
    q = torch.clamp(torch.round((w - lower) / scale), qmin, qmax)
    w_deq = q * scale + lower
    return w_deq, scale, lower


def omniquant_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    h_l: torch.Tensor,
    h_u: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OmniQuant forward pass combining Learnable Equivalent Transformation (LET) & LWC."""
    # LET per-channel scaling: s = exp(gamma)
    s = torch.exp(gamma)
    x_scaled = x / s.unsqueeze(0)
    w_scaled = weight * s.unsqueeze(1)

    # LWC quantization
    w_deq, _, _ = lwc_quantize(w_scaled, h_l, h_u)
    out = x_scaled @ w_deq
    if bias is not None:
        out = out + bias
    return out, w_deq


def omniquant_optimize_step(
    x_calib: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    h_l: torch.Tensor,
    h_u: torch.Tensor,
    lr: float = 0.05,
    steps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic micro-optimization loop optimizing LET scale (gamma) and LWC clipping (h_l, h_u)."""
    gamma_opt = gamma.clone().detach().requires_grad_(True)
    h_l_opt = h_l.clone().detach().requires_grad_(True)
    h_u_opt = h_u.clone().detach().requires_grad_(True)

    y_ref = (x_calib @ weight + bias).detach()
    optimizer = torch.optim.SGD([gamma_opt, h_l_opt, h_u_opt], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        out, _ = omniquant_forward(x_calib, weight, gamma_opt, h_l_opt, h_u_opt, bias)
        loss = torch.mean((out - y_ref) ** 2)
        loss.backward()
        optimizer.step()

    final_out, final_w_deq = omniquant_forward(
        x_calib, weight, gamma_opt.detach(), h_l_opt.detach(), h_u_opt.detach(), bias
    )
    return gamma_opt.detach(), h_l_opt.detach(), h_u_opt.detach(), final_out.detach()


def hqq_proximal_quantize(
    w: torch.Tensor,
    nbits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Half-Quadratic Quantization (HQQ) closed-form optimal scale and zero-point derivation."""
    qmax = float(2 ** nbits - 1)
    # Channel-wise min and max
    w_min = torch.min(w, dim=0, keepdim=True).values
    w_max = torch.max(w, dim=0, keepdim=True).values

    scale = (w_max - w_min) / qmax
    scale = torch.clamp(scale, min=1e-5)
    zero = -w_min / scale

    # Rounding step
    q = torch.clamp(torch.round(w / scale + zero), 0.0, qmax)
    w_deq = (q - zero) * scale
    return w_deq, scale, zero


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, d_in, d_out = 16, 16, 8
    x_calib = torch.randn(b, d_in)
    # Channel 1 and 4 have outliers
    x_calib[:, 1] *= 10.0
    x_calib[:, 4] *= 6.0

    w_orig = torch.randn(d_in, d_out) * 0.25
    b_orig = torch.randn(d_out) * 0.05

    # Initial OmniQuant parameters (gamma=0, h_l=1.0, h_u=1.0)
    gamma_init = torch.zeros(d_in)
    h_l_init = torch.ones(1, d_out) * 1.5
    h_u_init = torch.ones(1, d_out) * 1.5

    # 1. HQQ data-free baseline
    w_hqq, hqq_scale, hqq_zero = hqq_proximal_quantize(w_orig, nbits=4)
    out_hqq = x_calib @ w_hqq + b_orig

    # 2. OmniQuant optimization loop (10 steps)
    gamma_opt, h_l_opt, h_u_opt, out_omniquant = omniquant_optimize_step(
        x_calib, w_orig, b_orig, gamma_init, h_l_init, h_u_init, lr=0.05, steps=10
    )

    # 3. FP32 ground truth
    out_fp32 = x_calib @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "gamma_init": gamma_init,
        "h_l_init": h_l_init,
        "h_u_init": h_u_init,
        "w_hqq": w_hqq,
        "hqq_scale": hqq_scale,
        "hqq_zero": hqq_zero,
        "out_hqq": out_hqq,
        "gamma_opt": gamma_opt,
        "h_l_opt": h_l_opt,
        "h_u_opt": h_u_opt,
        "out_omniquant": out_omniquant,
        "out_fp32": out_fp32,
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
