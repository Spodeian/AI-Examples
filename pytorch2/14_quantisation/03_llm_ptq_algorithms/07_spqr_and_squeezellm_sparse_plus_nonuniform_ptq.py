"""SpQR and SqueezeLLM: Sparse Plus Nonuniform PTQ — PyTorch 2.x reference implementation.

Topic: 14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq
Golden weights: weights/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.safetensors

``python pytorch2/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def compute_hessian_sensitivity(
    x_calib: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Fisher/Hessian diagonal sensitivity S_ij = H_ii * |W_ij|."""
    n_samples = x_calib.shape[0]
    # H = 2 * X^T X / N
    h_diag = 2.0 * torch.sum(x_calib ** 2, dim=0) / float(n_samples)
    sensitivity = h_diag.unsqueeze(1) * torch.abs(weight)
    return h_diag, sensitivity


def extract_sparse_outliers(
    weight: torch.Tensor,
    sensitivity: torch.Tensor,
    top_k_count: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isolates the top-k most sensitive weights into a high-precision sparse component."""
    flat_sens = sensitivity.flatten()
    threshold = float(torch.topk(flat_sens, k=top_k_count).values[-1])
    # Sparse mask for top-k sensitive weights
    sparse_mask = sensitivity >= threshold

    w_sparse = torch.where(sparse_mask, weight, torch.zeros_like(weight))
    w_dense = torch.where(sparse_mask, torch.zeros_like(weight), weight)
    return w_sparse, w_dense, sparse_mask.float()


def quantize_dense_int3(
    w_dense: torch.Tensor,
    sparse_mask: torch.Tensor,
    qmin: float = -4.0,
    qmax: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric 3-bit quantization of the dense non-outlier weights."""
    # Scale computed only over non-outlier dense elements
    scale = torch.max(torch.abs(w_dense), dim=0, keepdim=True).values / qmax
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)

    q = torch.clamp(torch.round(w_dense / scale), qmin, qmax)
    # Zero out positions that belong to the sparse outlier matrix
    q = torch.where(sparse_mask > 0.0, torch.zeros_like(q), q)
    w_dense_deq = q * scale
    return w_dense_deq, scale


def spqr_forward(
    x: torch.Tensor,
    w_dense_deq: torch.Tensor,
    w_sparse: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SpQR decomposition forward pass: Y = X @ W_dense + X @ W_sparse + bias."""
    y_dense = x @ w_dense_deq
    y_sparse = x @ w_sparse
    y_total = y_dense + y_sparse
    if bias is not None:
        y_total = y_total + bias
    return y_total, y_dense, y_sparse


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, d_in, d_out = 16, 16, 8
    x_calib = torch.randn(b, d_in)
    # Outlier channel 0 and 3
    x_calib[:, 0] *= 8.0
    x_calib[:, 3] *= 5.0

    w_orig = torch.randn(d_in, d_out) * 0.25
    b_orig = torch.randn(d_out) * 0.05

    # 1. Hessian sensitivity calculation
    h_diag, sensitivity = compute_hessian_sensitivity(x_calib, w_orig)

    # 2. Sparse outlier extraction (top 8 sensitive weights out of 128 = 6.25%)
    w_sparse, w_dense, sparse_mask = extract_sparse_outliers(w_orig, sensitivity, top_k_count=8)

    # 3. Dense 3-bit quantization
    w_dense_deq, dense_scale = quantize_dense_int3(w_dense, sparse_mask, qmin=-4.0, qmax=3.0)

    # 4. SpQR forward output
    out_spqr, y_dense, y_sparse = spqr_forward(x_calib, w_dense_deq, w_sparse, b_orig)

    # 5. Baseline pure 3-bit quantization (without sparse outlier isolation)
    w_pure_dense_deq, _ = quantize_dense_int3(w_orig, torch.zeros_like(sparse_mask), qmin=-4.0, qmax=3.0)
    out_pure_3bit = x_calib @ w_pure_dense_deq + b_orig

    # 6. Exact FP32 ground truth
    out_fp32 = x_calib @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "h_diag": h_diag,
        "sensitivity": sensitivity,
        "w_sparse": w_sparse,
        "w_dense": w_dense,
        "sparse_mask": sparse_mask,
        "w_dense_deq": w_dense_deq,
        "dense_scale": dense_scale,
        "out_spqr": out_spqr,
        "y_dense": y_dense,
        "y_sparse": y_sparse,
        "out_pure_3bit": out_pure_3bit,
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
