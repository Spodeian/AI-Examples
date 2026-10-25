"""Smoothness Regularization: Sobolev Norms, Jacobian Penalties, and Shift Equivariance — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance
Golden weights: weights/06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def sobolev_gradient(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Computes exact analytical input gradient nabla_x y for scalar potential y = sum(gelu(x W1 + b1) W2).
    
    x: [B, D_in]
    w1: [D_in, D_h]
    b1: [D_h]
    w2: [D_h, 1]
    Returns grad_x of shape [B, D_in].
    """
    pre = x @ w1 + b1  # [B, D_h]
    # GELU derivative: d/dx [x * Phi(x)] = Phi(x) + x * phi(x)
    # Using tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Or exact autograd evaluation:
    x_in = x.detach().clone().requires_grad_(True)
    y = (F.gelu(x_in @ w1 + b1) @ w2).sum()
    grad_x = torch.autograd.grad(y, x_in, create_graph=False)[0]
    return grad_x


def jacobian_frobenius_sq(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Computes squared Frobenius norm of the Jacobian ||J_f(x)||_F^2 for f(x) = gelu(x W1 + b1) W2.
    
    x: [B, D_in]
    w1: [D_in, D_h]
    b1: [D_h]
    w2: [D_h, D_out]
    Returns frobenius_sq of shape [B].
    """
    b, d_in = x.shape
    d_out = w2.shape[1]
    
    frobenius_sq = []
    for i in range(b):
        x_i = x[i : i + 1].clone().requires_grad_(True)
        h = F.gelu(x_i @ w1 + b1)
        y = h @ w2  # [1, D_out]
        
        j_rows = []
        for o in range(d_out):
            grad_o = torch.autograd.grad(y[0, o], x_i, retain_graph=True)[0]
            j_rows.append(grad_o)
        j_matrix = torch.cat(j_rows, dim=0)  # [D_out, D_in]
        frobenius_sq.append(torch.sum(j_matrix ** 2))
    return torch.stack(frobenius_sq)


def blur_downsample_1d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Anti-aliased 1D BlurPool downsampling: convolve with low-pass filter with stride 2.
    
    x: [B, C, L]
    kernel: [K] e.g. [0.25, 0.5, 0.25]
    Returns downsampled tensor of shape [B, C, L // 2].
    """
    b, c, l = x.shape
    k_len = kernel.shape[0]
    weight = kernel.view(1, 1, k_len).repeat(c, 1, 1)
    # Same padding to preserve size before stride
    pad = (k_len - 1) // 2
    return F.conv1d(x, weight, stride=2, padding=pad, groups=c)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Sobolev & Jacobian setup: B=2, D_in=4, D_h=6, D_out=3
    B, D_in, D_h, D_out = 2, 4, 6, 3
    x_feat = torch.randn(B, D_in)
    w1 = torch.randn(D_in, D_h) * 0.2
    b1 = torch.randn(D_h) * 0.1
    w2_scalar = torch.randn(D_h, 1) * 0.2
    w2_multi = torch.randn(D_h, D_out) * 0.2

    sobolev_grad = sobolev_gradient(x_feat, w1, b1, w2_scalar)
    j_frob_sq = jacobian_frobenius_sq(x_feat, w1, b1, w2_multi)

    # Blur downsampling: B=2, C=2, L=8
    x_signal = torch.randn(2, 2, 8)
    kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
    y_blur = blur_downsample_1d(x_signal, kernel_1d)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_feat": x_feat,
        "w1": w1,
        "b1": b1,
        "w2_scalar": w2_scalar,
        "w2_multi": w2_multi,
        "sobolev_grad": sobolev_grad,
        "j_frob_sq": j_frob_sq,
        "x_signal": x_signal,
        "kernel_1d": kernel_1d,
        "y_blur": y_blur,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous().cpu() for k, v in build_golden_tensors().items()}
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
