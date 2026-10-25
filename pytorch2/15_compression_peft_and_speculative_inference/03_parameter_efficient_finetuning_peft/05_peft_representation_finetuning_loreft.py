"""PEFT Representation Finetuning Loreft — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft
Golden weights: weights/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def loreft_subspace_delta(
    h: torch.Tensor,
    r_matrix: torch.Tensor,
    w_matrix: torch.Tensor,
    b_vec: torch.Tensor,
) -> torch.Tensor:
    """Compute LoReFT representation delta: delta_h = R^T (W h + b - R h).

    Args:
        h: Hidden representation [batch, seq_len, dim].
        r_matrix: Orthonormal subspace projector [rank, dim].
        w_matrix: Learned linear map [rank, dim].
        b_vec: Subspace offset [rank].

    Returns:
        delta_h: Intervened representation shift [batch, seq_len, dim].
    """
    # Project to subspace: [batch, seq_len, rank]
    proj_r = torch.matmul(h, r_matrix.t())
    proj_w = torch.matmul(h, w_matrix.t()) + b_vec
    delta_s = proj_w - proj_r
    # Project back to representation space: [batch, seq_len, dim]
    delta_h = torch.matmul(delta_s, r_matrix)
    return delta_h


def loreft_intervene(
    h: torch.Tensor,
    r_matrix: torch.Tensor,
    w_matrix: torch.Tensor,
    b_vec: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    """LoReFT intervention at selective token positions.

    h_out = h + mask * delta_h

    Args:
        h: Hidden representation [batch, seq_len, dim].
        r_matrix: Orthonormal subspace projector [rank, dim].
        w_matrix: Learned linear map [rank, dim].
        b_vec: Subspace offset [rank].
        position_mask: Binary or soft position mask [batch, seq_len].

    Returns:
        h_out: Intervened hidden representation [batch, seq_len, dim].
    """
    delta_h = loreft_subspace_delta(h, r_matrix, w_matrix, b_vec)
    mask = position_mask.unsqueeze(-1)  # [batch, seq_len, 1]
    return h + mask * delta_h


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    batch = 2
    seq_len = 4
    dim = 8
    rank = 2

    h = torch.randn(batch, seq_len, dim)

    # Construct orthonormal R [rank, dim] via QR
    raw_r = torch.randn(dim, rank)
    q_mat, _ = torch.linalg.qr(raw_r)
    r_matrix = q_mat[:, :rank].t().contiguous()  # [rank, dim]

    w_matrix = torch.randn(rank, dim) * 0.25
    b_vec = torch.randn(rank) * 0.1

    position_mask = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    delta_h = loreft_subspace_delta(h, r_matrix, w_matrix, b_vec)
    loreft_out = loreft_intervene(h, r_matrix, w_matrix, b_vec, position_mask)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "h": h,
        "r_matrix": r_matrix,
        "w_matrix": w_matrix,
        "b_vec": b_vec,
        "position_mask": position_mask,
        "delta_h": delta_h,
        "loreft_out": loreft_out,
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
