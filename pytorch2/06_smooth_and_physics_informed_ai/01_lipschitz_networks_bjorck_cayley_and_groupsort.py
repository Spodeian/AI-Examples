"""Lipschitz Networks: Björck Orthonormalization, Cayley Transform, and GroupSort — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort
Golden weights: weights/06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def bjorck_orthonormalize(w: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Iterative Björck orthonormalization: projects W onto the Stiefel manifold.
    
    W has shape [D_in, D_out] with D_in >= D_out.
    Iteratively computes W_{k+1} = W_k (1.5 * I - 0.5 * W_k^T W_k).
    """
    norm = torch.linalg.matrix_norm(w, ord="fro") + 1e-6
    w_k = w / norm
    d_out = w.shape[1]
    eye = torch.eye(d_out, dtype=w.dtype, device=w.device)

    for _ in range(iters):
        gram = w_k.t() @ w_k
        w_k = w_k @ (1.5 * eye - 0.5 * gram)
    return w_k


def cayley_transform(m: torch.Tensor) -> torch.Tensor:
    """Cayley transform: constructs an exact orthogonal matrix from unconstrained M.
    
    A = M - M^T is skew-symmetric.
    Q = (I - A)(I + A)^(-1) is exactly orthogonal (Q^T Q = I).
    """
    d = m.shape[0]
    eye = torch.eye(d, dtype=m.dtype, device=m.device)
    a = m - m.t()
    q = (eye - a) @ torch.linalg.inv(eye + a)
    return q


def groupsort2(x: torch.Tensor) -> torch.Tensor:
    """GroupSort-2 (PairSort) activation function: sorts adjacent pairs in descending order.
    
    Preserves the L2 norm (isometry) and is strictly 1-Lipschitz.
    x has shape [B, D] where D is even.
    """
    b, d = x.shape
    pairs = x.view(b, d // 2, 2)
    sorted_pairs, _ = torch.sort(pairs, dim=-1, descending=True)
    return sorted_pairs.reshape(b, d)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Björck orthonormalization: D=6
    D = 6
    w_raw = torch.randn(D, D)
    w_ortho = bjorck_orthonormalize(w_raw, iters=5)

    # Cayley transform: D_cayley=4
    D_c = 4
    m_raw = torch.randn(D_c, D_c) * 0.5
    q_cayley = cayley_transform(m_raw)

    # GroupSort2 input and forward: B=3, D=6
    B = 3
    x_feat = torch.randn(B, D)
    h_bjorck = x_feat @ w_ortho
    y_groupsort = groupsort2(h_bjorck)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "w_raw": w_raw,
        "w_ortho": w_ortho,
        "m_raw": m_raw,
        "q_cayley": q_cayley,
        "x_feat": x_feat,
        "h_bjorck": h_bjorck,
        "y_groupsort": y_groupsort,
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
