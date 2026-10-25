"""AI Analysis Loss Landscape Hessian Lanczos and Empirical NTK — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk
Golden weights: weights/10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def filter_normalize_direction(theta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Filter-normalization for scale-invariant loss landscape visualization (Li et al., 2018).
    
    delta_{i} <- delta_{i} * (||theta_{i}||_2 / (||delta_{i}||_2 + eps))
    """
    norm_theta = torch.sqrt((theta * theta).sum(dim=-1, keepdim=True) + 1e-8)
    norm_delta = torch.sqrt((delta * delta).sum(dim=-1, keepdim=True) + 1e-8)
    return delta * (norm_theta / norm_delta)


def loss_surface_coord(
    theta: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Evaluates sliced 2D coordinate on loss landscape: theta(alpha, beta) = theta + alpha * d1 + beta * d2."""
    return theta + float(alpha) * d1 + float(beta) * d2


def lanczos_tridiagonal(
    h: torch.Tensor,
    v0: torch.Tensor,
    num_steps: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lanczos iteration for Hessian spectral estimation (Ghorbani et al., 2019 / PyHessian).
    
    Generates tridiagonal matrix T with diagonal alpha and sub-diagonal beta.
    """
    p = h.shape[0]
    q = torch.zeros(num_steps + 1, p, dtype=h.dtype, device=h.device)
    v_norm = torch.sqrt((v0 * v0).sum() + 1e-8)
    q[0] = v0 / v_norm

    alpha_diag = torch.zeros(num_steps, dtype=h.dtype, device=h.device)
    beta_diag = torch.zeros(num_steps - 1, dtype=h.dtype, device=h.device)

    for j in range(num_steps):
        w = h @ q[j]
        if j > 0:
            w = w - beta_diag[j - 1] * q[j - 1]
        a_j = (q[j] * w).sum()
        alpha_diag[j] = a_j
        w = w - a_j * q[j]
        if j < num_steps - 1:
            b_j = torch.sqrt((w * w).sum() + 1e-8)
            beta_diag[j] = b_j
            q[j + 1] = w / b_j

    return alpha_diag, beta_diag


def empirical_ntk(jacobians: torch.Tensor) -> torch.Tensor:
    """Empirical Neural Tangent Kernel (eNTK) Gram matrix Theta = J J^T (Jacot et al., 2018).
    
    jacobians: [N, P] sample gradient vectors.
    Returns Theta: [N, N] kernel Gram matrix.
    """
    return jacobians @ jacobians.t()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. Filter-normalized loss landscape slicing: [3 filters, 4 weights]
    ls_theta = torch.tensor([
        [1.0, -2.0, 3.0, -4.0],
        [0.5, 0.5, -0.5, -0.5],
        [2.0, 0.0, -2.0, 1.0],
    ], dtype=torch.float32)
    ls_d1_raw = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [1.0, -1.0, 1.0, -1.0],
        [-0.5, 0.5, -0.5, 0.5],
    ], dtype=torch.float32)
    ls_d2_raw = torch.tensor([
        [-0.4, 0.3, -0.2, 0.1],
        [0.2, 0.4, 0.6, 0.8],
        [1.0, 1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    ls_d1_norm = filter_normalize_direction(ls_theta, ls_d1_raw)
    ls_d2_norm = filter_normalize_direction(ls_theta, ls_d2_raw)
    ls_slice_ref = loss_surface_coord(ls_theta, ls_d1_norm, ls_d2_norm, alpha=0.5, beta=-0.25)

    # 2. Lanczos tridiagonalization: 4x4 symmetric Hessian matrix
    h_mat = torch.tensor([
        [4.0, 1.0, 0.5, 0.2],
        [1.0, 3.0, 0.8, 0.4],
        [0.5, 0.8, 2.5, 0.1],
        [0.2, 0.4, 0.1, 1.5],
    ], dtype=torch.float32)
    v0 = torch.tensor([1.0, 0.5, -0.5, 0.2], dtype=torch.float32)
    lanczos_alpha_ref, lanczos_beta_ref = lanczos_tridiagonal(h_mat, v0, num_steps=3)

    # 3. Empirical NTK: N=4 samples, P=6 parameters
    ntk_j = torch.tensor([
        [1.0, -0.5, 0.2, 0.8, -0.1, 0.4],
        [0.5, 1.2, -0.4, 0.1, 0.9, -0.3],
        [-0.8, 0.2, 1.1, -0.6, 0.3, 0.5],
        [0.2, -0.7, 0.3, 1.0, -0.8, 0.2],
    ], dtype=torch.float32)
    ntk_gram_ref = empirical_ntk(ntk_j)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "ls_theta": ls_theta,
        "ls_d1_raw": ls_d1_raw,
        "ls_d2_raw": ls_d2_raw,
        "ls_d1_norm": ls_d1_norm,
        "ls_d2_norm": ls_d2_norm,
        "ls_slice_ref": ls_slice_ref,
        "h_mat": h_mat,
        "v0": v0,
        "lanczos_alpha_ref": lanczos_alpha_ref,
        "lanczos_beta_ref": lanczos_beta_ref,
        "ntk_j": ntk_j,
        "ntk_gram_ref": ntk_gram_ref,
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
