"""QAT STE LSQ LSQ+ PACT and DoReFa — PyTorch 2.x reference implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa
Golden weights: weights/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.safetensors

``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ste_quantize(
    w: torch.Tensor,
    scale: torch.Tensor,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight-Through Estimator (STE) uniform quantization with gradient pass-through."""
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    w_deq = q * scale
    # Clamped STE mask: gradients flow only within [qmin * scale, qmax * scale]
    ste_mask = ((w >= qmin * scale) & (w <= qmax * scale)).float()
    return w_deq, ste_mask


def pact_activation(
    x: torch.Tensor,
    alpha: torch.Tensor,
    nbits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parameterized Activation Clipping (PACT) with learnable clipping threshold alpha."""
    qmax = float(2 ** nbits - 1)
    # Clip activation to [0, alpha]
    x_clipped = torch.clamp(x, 0.0, float(alpha))
    step = alpha / qmax
    q = torch.round(x_clipped / step)
    x_deq = q * step
    # PACT gradient wrt alpha: 1.0 where x >= alpha, 0.0 otherwise
    grad_alpha_mask = (x >= alpha).float()
    return x_deq, grad_alpha_mask


def lsq_quantize(
    w: torch.Tensor,
    scale: torch.Tensor,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Learned Step Size Quantization (LSQ) with analytical gradient scaling.
    
    scale_grad = sum(dLoss/dw * d_w_bar/ds) * (1 / sqrt(N * qmax))
    """
    v = w / scale
    q = torch.clamp(torch.round(v), qmin, qmax)
    w_deq = q * scale

    # Analytical d_w_bar / d_scale:
    # where v < qmin: qmin
    # where v > qmax: qmax
    # otherwise: round(v) - v
    d_w_bar_ds = torch.where(
        v < qmin,
        torch.full_like(v, qmin),
        torch.where(
            v > qmax,
            torch.full_like(v, qmax),
            torch.round(v) - v,
        ),
    )
    # Analytical LSQ gradient scale factor
    g = 1.0 / (float(w.numel() * qmax) ** 0.5)
    return w_deq, d_w_bar_ds * g


def lsq_plus_quantize(
    w: torch.Tensor,
    scale: torch.Tensor,
    beta: torch.Tensor,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> torch.Tensor:
    """LSQ+ quantization with learnable scale and asymmetric offset beta."""
    v = (w - beta) / scale
    q = torch.clamp(torch.round(v), qmin, qmax)
    return q * scale + beta


def qat_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_w: torch.Tensor,
    alpha_x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QAT end-to-end forward pass combining PACT activations and LSQ weights."""
    x_q, grad_alpha_mask = pact_activation(x, alpha_x, nbits=4)
    w_q, d_w_bar_ds = lsq_quantize(weight, scale_w, qmin=-8.0, qmax=7.0)
    y = x_q @ w_q
    if bias is not None:
        y = y + bias
    return y, x_q, w_q


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, d_in, d_out = 16, 16, 8
    x_calib = torch.randn(b, d_in).abs()  # positive activations for PACT
    w_orig = torch.randn(d_in, d_out) * 0.35
    b_orig = torch.randn(d_out) * 0.05

    scale_w = torch.tensor([[0.05]])
    alpha_x = torch.tensor(2.50)
    beta_w = torch.tensor([[0.02]])

    # 1. STE uniform quantization
    w_ste_deq, ste_mask = ste_quantize(w_orig, scale_w, qmin=-8.0, qmax=7.0)

    # 2. PACT activation quantization
    x_pact_deq, pact_grad_mask = pact_activation(x_calib, alpha_x, nbits=4)

    # 3. LSQ weight quantization
    w_lsq_deq, lsq_grad_scale = lsq_quantize(w_orig, scale_w, qmin=-8.0, qmax=7.0)

    # 4. LSQ+ asymmetric quantization
    w_lsq_plus = lsq_plus_quantize(w_orig, scale_w, beta_w, qmin=-8.0, qmax=7.0)

    # 5. QAT end-to-end forward pass
    out_qat, x_q_pass, w_q_pass = qat_forward(x_calib, w_orig, scale_w, alpha_x, b_orig)

    # 6. FP32 unquantized ground truth
    out_fp32 = x_calib @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "scale_w": scale_w,
        "alpha_x": alpha_x,
        "beta_w": beta_w,
        "w_ste_deq": w_ste_deq,
        "ste_mask": ste_mask,
        "x_pact_deq": x_pact_deq,
        "pact_grad_mask": pact_grad_mask,
        "w_lsq_deq": w_lsq_deq,
        "lsq_grad_scale": lsq_grad_scale,
        "w_lsq_plus": w_lsq_plus,
        "out_qat": out_qat,
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
