"""RTN vs GPTQ Hessian Error Compensation — PyTorch 2.x reference implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation
Golden weights: weights/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.safetensors

``python pytorch2/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def rtn_quantize(
    weight: torch.Tensor,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> torch.Tensor:
    """Standard Round-To-Nearest (RTN) per-column symmetric quantization."""
    scale = torch.max(torch.abs(weight), dim=0, keepdim=True).values / qmax
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    return torch.clamp(torch.round(weight / scale), qmin, qmax) * scale


def compute_damped_hessian_inv(
    x: torch.Tensor,
    damp_factor: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes empirical activation Hessian H = 2 * X^T X / N and damped inverse."""
    n_samples = x.shape[0]
    d_in = x.shape[1]
    H = 2.0 * (x.T @ x) / float(n_samples)
    diag_mean = float(torch.mean(torch.diag(H)))
    damp = damp_factor * diag_mean if diag_mean > 0.0 else 1e-4
    H_damped = H + damp * torch.eye(d_in, dtype=H.dtype, device=H.device)
    H_inv = torch.inverse(H_damped)
    return H, H_inv


def gptq_quantize(
    weight: torch.Tensor,
    h_inv: torch.Tensor,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> torch.Tensor:
    """Second-order error compensation GPTQ column-by-column update."""
    w = weight.clone()
    d_in, _ = w.shape
    scale = torch.max(torch.abs(weight), dim=0, keepdim=True).values / qmax
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)

    for c in range(d_in):
        w_c = w[c, :]
        s = scale[0, :]
        q_c = torch.clamp(torch.round(w_c / s), qmin, qmax) * s
        err_c = w_c - q_c
        h_inv_cc = float(h_inv[c, c])
        if h_inv_cc != 0.0 and c + 1 < d_in:
            step = err_c.unsqueeze(0) / h_inv_cc
            delta = step * h_inv[c + 1 :, c].unsqueeze(1)
            w[c + 1 :, :] -= delta
        w[c, :] = q_c

    return w


def gptq_with_act_order(
    x: torch.Tensor,
    weight: torch.Tensor,
    damp_factor: float = 0.01,
    qmin: float = -8.0,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ with act-order (descending Hessian diagonal activation sorting)."""
    n_samples = x.shape[0]
    d_in = x.shape[1]
    H = 2.0 * (x.T @ x) / float(n_samples)
    diag_h = torch.diag(H)
    perm = torch.argsort(diag_h, descending=True)
    inv_perm = torch.argsort(perm)

    x_perm = x[:, perm]
    w_perm = weight[perm, :]

    H_perm, H_inv_perm = compute_damped_hessian_inv(x_perm, damp_factor=damp_factor)
    w_perm_quant = gptq_quantize(w_perm, H_inv_perm, qmin=qmin, qmax=qmax)
    w_final = w_perm_quant[inv_perm, :]

    return w_final, perm.float(), H_inv_perm


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Calibration activations and weight
    n_samples, d_in, d_out = 64, 16, 8
    x_calib = torch.randn(n_samples, d_in)
    # Skew first two channels to demonstrate act-order benefit
    x_calib[:, 0] *= 8.0
    x_calib[:, 1] *= 4.0

    w_orig = torch.randn(d_in, d_out) * 0.2
    b_orig = torch.randn(d_out) * 0.1

    # 1. Hessian computation
    h_matrix, h_inv = compute_damped_hessian_inv(x_calib, damp_factor=0.01)

    # 2. RTN baseline
    w_rtn = rtn_quantize(w_orig, qmin=-8.0, qmax=7.0)
    out_rtn = x_calib[:8] @ w_rtn + b_orig

    # 3. Standard GPTQ
    w_gptq = gptq_quantize(w_orig, h_inv, qmin=-8.0, qmax=7.0)
    out_gptq = x_calib[:8] @ w_gptq + b_orig

    # 4. GPTQ with Act-Order
    w_act_gptq, perm_indices, _ = gptq_with_act_order(x_calib, w_orig, damp_factor=0.01, qmin=-8.0, qmax=7.0)
    out_act_gptq = x_calib[:8] @ w_act_gptq + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "h_matrix": h_matrix,
        "h_inv": h_inv,
        "w_rtn": w_rtn,
        "out_rtn": out_rtn,
        "w_gptq": w_gptq,
        "out_gptq": out_gptq,
        "w_act_gptq": w_act_gptq,
        "perm_indices": perm_indices,
        "out_act_gptq": out_act_gptq,
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
