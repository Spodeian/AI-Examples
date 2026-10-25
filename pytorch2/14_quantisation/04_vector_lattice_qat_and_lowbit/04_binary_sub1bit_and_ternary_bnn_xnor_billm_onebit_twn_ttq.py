"""Binary Sub1bit and Ternary BNN XNOR BiLLM OneBit TWN TTQ — PyTorch 2.x reference implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq
Golden weights: weights/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.safetensors

``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def bnn_xnor_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binary Neural Network (BNN / XNOR-Net) 1-bit quantization.
    
    B = sign(W) in {-1, +1}
    alpha = mean(|W|)
    returns: (w_bnn, b_signs, alpha)
    """
    signs = torch.where(w >= 0.0, 1.0, -1.0)
    alpha = torch.mean(torch.abs(w), dim=0, keepdim=True)
    w_bnn = signs * alpha
    return w_bnn, signs, alpha


def twn_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ternary Weight Networks (TWN) 1.58-bit {-1, 0, +1} quantization.
    
    Delta = 0.7 * mean(|W|)
    T_i = +1 if W_i > Delta, -1 if W_i < -Delta, 0 otherwise
    alpha = mean_{|W| > Delta}(|W|)
    returns: (w_twn, t_ternary, alpha)
    """
    delta = 0.7 * torch.mean(torch.abs(w), dim=0, keepdim=True)
    t = torch.where(w > delta, 1.0, torch.where(w < -delta, -1.0, 0.0))
    nonzeros = (torch.abs(t) > 0.0).float()
    nz_counts = torch.sum(nonzeros, dim=0, keepdim=True)
    nz_counts = torch.clamp(nz_counts, min=1.0)
    alpha = torch.sum(torch.abs(w) * nonzeros, dim=0, keepdim=True) / nz_counts
    w_twn = t * alpha
    return w_twn, t, alpha


def ttq_quantize(w: torch.Tensor, delta_factor: float = 0.35) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trained Ternary Quantization (TTQ) with asymmetric positive and negative scales.
    
    returns: (w_ttq, t_ternary, w_pos_scale, w_neg_scale)
    """
    delta = delta_factor * torch.max(torch.abs(w), dim=0, keepdim=True).values
    pos_mask = (w > delta).float()
    neg_mask = (w < -delta).float()
    t = pos_mask - neg_mask

    pos_count = torch.clamp(torch.sum(pos_mask, dim=0, keepdim=True), min=1.0)
    neg_count = torch.clamp(torch.sum(neg_mask, dim=0, keepdim=True), min=1.0)

    w_pos = torch.sum(w * pos_mask, dim=0, keepdim=True) / pos_count
    w_neg = torch.sum(torch.abs(w) * neg_mask, dim=0, keepdim=True) / neg_count

    w_ttq = pos_mask * w_pos - neg_mask * w_neg
    return w_ttq, t, w_pos, w_neg


def onebit_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""OneBit: 1-bit value-sign rank-1 scaling decomposition:
    
    W \approx S \odot (g h^T)
    S in {-1, +1}, g in R^{d_in x 1}, h in R^{1 x d_out}
    returns: (w_onebit, signs, g, h)
    """
    signs = torch.where(w >= 0.0, 1.0, -1.0)
    abs_w = torch.abs(w)
    # Row-wise mean g [d_in, 1]
    g = torch.mean(abs_w, dim=1, keepdim=True)
    # Column-wise normalization factor h [1, d_out]
    total_abs = torch.clamp(torch.sum(abs_w), min=1e-5)
    h = (torch.sum(abs_w, dim=0, keepdim=True) / total_abs) * float(w.shape[0])

    scale_matrix = g @ h
    w_onebit = signs * scale_matrix
    return w_onebit, signs, g, h


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, d_in, d_out = 16, 16, 8
    x_calib = torch.randn(b, d_in)
    w_orig = torch.randn(d_in, d_out) * 0.4
    b_orig = torch.randn(d_out) * 0.05

    # 1. BNN 1-bit quantization
    w_bnn, b_signs, b_alpha = bnn_xnor_quantize(w_orig)
    out_bnn = x_calib @ w_bnn + b_orig

    # 2. TWN ternary quantization
    w_twn, t_ternary, t_alpha = twn_quantize(w_orig)
    out_twn = x_calib @ w_twn + b_orig

    # 3. TTQ asymmetric ternary quantization
    w_ttq, ttq_t, w_pos, w_neg = ttq_quantize(w_orig)
    out_ttq = x_calib @ w_ttq + b_orig

    # 4. OneBit rank-1 sign-magnitude decomposition
    w_onebit, onebit_signs, onebit_g, onebit_h = onebit_quantize(w_orig)
    out_onebit = x_calib @ w_onebit + b_orig

    # 5. Exact FP32 ground truth
    out_fp32 = x_calib @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "w_bnn": w_bnn,
        "b_signs": b_signs,
        "b_alpha": b_alpha,
        "out_bnn": out_bnn,
        "w_twn": w_twn,
        "t_ternary": t_ternary,
        "t_alpha": t_alpha,
        "out_twn": out_twn,
        "w_ttq": w_ttq,
        "ttq_t": ttq_t,
        "w_pos": w_pos,
        "w_neg": w_neg,
        "out_ttq": out_ttq,
        "w_onebit": w_onebit,
        "onebit_signs": onebit_signs,
        "onebit_g": onebit_g,
        "onebit_h": onebit_h,
        "out_onebit": out_onebit,
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
