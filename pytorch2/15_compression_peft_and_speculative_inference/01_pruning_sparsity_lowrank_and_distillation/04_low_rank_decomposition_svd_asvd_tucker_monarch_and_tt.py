"""Low Rank Decomposition SVD Asvd Tucker Monarch and Tt — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt
Golden weights: weights/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def truncated_svd(w: torch.Tensor, rank: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated SVD rank-r factorization (Eckart-Young-Mirsky optimal approximation).

    Args:
        w: Weight matrix of shape [rows, cols].
        rank: Target low rank r <= min(rows, cols).

    Returns:
        w_recon: Low-rank reconstructed matrix of shape [rows, cols].
        singular_values: Top-r singular values.
        frobenius_error: ||w - w_recon||_F.
    """
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    w_recon = (U_r * S_r.unsqueeze(0)) @ Vh_r
    frob_err = torch.norm(w - w_recon, p="fro")
    return w_recon, S_r, frob_err


def asvd_decompose(
    w: torch.Tensor, act_scales: torch.Tensor, rank: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Activation-aware SVD (ASVD). Scales weights by input variance prior to SVD.

    Args:
        w: Weight matrix of shape [rows, cols].
        act_scales: Per-column scaling vector (e.g. activation L2 norms) of shape [cols].
        rank: Target rank r.

    Returns:
        w_asvd: Activation-weighted reconstructed matrix.
        singular_values: Singular values of the scaled system.
    """
    s = torch.clamp(act_scales, min=1e-5)
    # W_scaled = W @ diag(s)
    w_scaled = w * s.unsqueeze(0)
    U, S, Vh = torch.linalg.svd(w_scaled, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    w_scaled_r = (U_r * S_r.unsqueeze(0)) @ Vh_r
    # Unscale: W_asvd = W_scaled_r @ diag(s^-1)
    w_asvd = w_scaled_r / s.unsqueeze(0)
    return w_asvd, S_r


def monarch_forward(x: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Monarch factorized linear transformation using two block-diagonal stages.

    Given dim = 16 partitioned into 4 blocks of size 4:
    Stage 1: Apply block-diagonal B1 of shape [4, 4, 4] to x.view(B, 4, 4).
    Stage 2: Transpose blocks (spatial permutation).
    Stage 3: Apply block-diagonal B2 of shape [4, 4, 4] and flatten.

    Args:
        x: Input tensor of shape [batch, 16].
        b1: Block-diagonal kernel 1 of shape [4, 4, 4].
        b2: Block-diagonal kernel 2 of shape [4, 4, 4].

    Returns:
        y: Output tensor of shape [batch, 16].
    """
    batch = x.shape[0]
    # Reshape x to [batch, 4 blocks, 4 elements]
    x_blk = x.view(batch, 4, 4)
    # Stage 1: bmm per block -> x_blk * b1
    # einsum: "b k i, k i j -> b k j"
    h1 = torch.einsum("bki,kij->bkj", x_blk, b1)
    # Stage 2: Transpose block axes
    h1_perm = h1.permute(0, 2, 1).contiguous()
    # Stage 3: bmm with b2
    h2 = torch.einsum("bki,kij->bkj", h1_perm, b2)
    return h2.reshape(batch, 16)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Weights: 16 rows, 16 cols
    w = torch.randn(16, 16) * 0.2
    x = torch.randn(8, 16)
    act_scales = torch.norm(x, p=2, dim=0) + 0.1 * torch.randn(16).abs()

    # 1. Truncated SVD (rank 4)
    w_svd, s_svd, frob_svd = truncated_svd(w, rank=4)

    # 2. ASVD (rank 4)
    w_asvd, s_asvd = asvd_decompose(w, act_scales, rank=4)

    # 3. Output reconstruction errors on x
    err_svd = float(torch.norm(x @ w.T - x @ w_svd.T) ** 2 / (torch.norm(x @ w.T) ** 2 + 1e-8))
    err_asvd = float(torch.norm(x @ w.T - x @ w_asvd.T) ** 2 / (torch.norm(x @ w.T) ** 2 + 1e-8))
    err_metrics = torch.tensor([err_svd, err_asvd, float(frob_svd)], dtype=torch.float32)

    # 4. Monarch factorized matrices (dim=16, 4 blocks of 4x4)
    b1 = torch.randn(4, 4, 4) * 0.2
    b2 = torch.randn(4, 4, 4) * 0.2
    y_monarch = monarch_forward(x, b1, b2)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "w": w,
        "x": x,
        "act_scales": act_scales,
        "w_svd": w_svd,
        "s_svd": s_svd,
        "w_asvd": w_asvd,
        "s_asvd": s_asvd,
        "err_metrics": err_metrics,
        "b1": b1,
        "b2": b2,
        "y_monarch": y_monarch,
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
        assert err < 1e-5, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
