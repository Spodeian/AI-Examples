"""2 4 Semi Structured Sparsity and Rigl Dynamic Sparse — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse
Golden weights: weights/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def prune_2_to_4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies NVIDIA Ampere/Hopper 2:4 semi-structured sparsity pattern.

    In every contiguous block of 4 elements along the last dimension,
    retains the 2 elements with largest absolute magnitude and zeroes the other 2.

    Args:
        w: Weight tensor of shape [rows, cols] where cols is divisible by 4.

    Returns:
        w_sparse: Sparse weight tensor of shape [rows, cols].
        mask: Binary indicator tensor of shape [rows, cols] (two 1s per 4-block).
    """
    rows, cols = w.shape
    assert cols % 4 == 0, f"Columns ({cols}) must be divisible by 4"
    w_blocks = w.view(rows, cols // 4, 4)
    abs_blocks = torch.abs(w_blocks)
    top2_indices = torch.topk(abs_blocks, k=2, dim=-1, largest=True).indices
    mask_blocks = torch.zeros_like(w_blocks)
    mask_blocks.scatter_(-1, top2_indices, 1.0)
    mask = mask_blocks.view(rows, cols)
    w_sparse = w * mask
    return w_sparse, mask


def compress_2_to_4(w: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs 2:4 sparse weights into halved dense storage and 2-bit index metadata.

    Args:
        w: Weight tensor of shape [rows, cols].
        mask: Binary 2:4 mask of shape [rows, cols].

    Returns:
        values: Packed non-zero values of shape [rows, cols // 2].
        indices: Positional column indices (0..3) of shape [rows, (cols // 4) * 2].
    """
    rows, cols = w.shape
    num_blocks = cols // 4
    w_blocks = w.view(rows, num_blocks, 4)
    mask_blocks = mask.view(rows, num_blocks, 4)

    vals_list = []
    idxs_list = []
    for r in range(rows):
        row_vals = []
        row_idxs = []
        for b in range(num_blocks):
            m_block = mask_blocks[r, b]
            w_block = w_blocks[r, b]
            kept_indices = torch.nonzero(m_block > 0.5, as_tuple=False).flatten()
            for idx in kept_indices:
                row_vals.append(w_block[idx])
                row_idxs.append(idx.to(torch.float32))
        vals_list.append(torch.stack(row_vals))
        idxs_list.append(torch.stack(row_idxs))

    values = torch.stack(vals_list)
    indices = torch.stack(idxs_list)
    return values, indices


def rigl_update_mask(
    w: torch.Tensor,
    grad: torch.Tensor,
    current_mask: torch.Tensor,
    drop_fraction: float = 0.3,
) -> torch.Tensor:
    """RigL (Rigging the Lottery) dynamic sparse topology exploration step.

    Drops active connections with smallest absolute weight |W| and
    regrows inactive connections with largest absolute gradient |∇W L|.

    Args:
        w: Current weight tensor of shape [rows, cols].
        grad: Current gradient tensor of shape [rows, cols].
        current_mask: Current binary mask with fixed sparsity.
        drop_fraction: Fraction of active connections to drop and regrow.

    Returns:
        new_mask: Updated binary mask with identical total sparsity.
    """
    flat_w = w.flatten()
    flat_grad = grad.flatten()
    flat_mask = current_mask.flatten()

    active_indices = torch.nonzero(flat_mask > 0.5, as_tuple=False).flatten()
    inactive_indices = torch.nonzero(flat_mask <= 0.5, as_tuple=False).flatten()
    n_active = len(active_indices)
    k_drop = max(1, int(drop_fraction * n_active))

    # 1. Drop active connections with smallest |w|
    active_mags = torch.abs(flat_w[active_indices])
    drop_subindices = torch.topk(active_mags, k_drop, largest=False).indices
    to_drop = active_indices[drop_subindices]

    # 2. Regrow inactive connections with largest |grad|
    inactive_grad_mags = torch.abs(flat_grad[inactive_indices])
    regrow_subindices = torch.topk(inactive_grad_mags, k_drop, largest=True).indices
    to_regrow = inactive_indices[regrow_subindices]

    # 3. Form new mask
    new_mask = flat_mask.clone()
    new_mask[to_drop] = 0.0
    new_mask[to_regrow] = 1.0
    return new_mask.view_as(current_mask)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Weight tensor: 16 rows, 16 cols (4 blocks of 4 per row)
    w = torch.randn(16, 16) * 0.2
    grad = torch.randn(16, 16) * 0.05

    # 1. 2:4 Semi-structured Pruning
    w_2_to_4, mask_2_to_4 = prune_2_to_4(w)

    # 2. 2:4 Storage Compression
    packed_values, packed_indices = compress_2_to_4(w, mask_2_to_4)

    # 3. RigL Dynamic Sparsity Update on a 50% random initial mask
    initial_mask = (torch.rand(16, 16) > 0.5).to(torch.float32)
    # Ensure at least some active and inactive
    if initial_mask.sum() == 0 or initial_mask.sum() == 256:
        initial_mask[0, :8] = 1.0
        initial_mask[0, 8:] = 0.0

    rigl_mask_step1 = rigl_update_mask(w, grad, initial_mask, drop_fraction=0.25)
    # Step 2: evolve further with perturbed grad
    grad_step2 = grad + 0.02 * torch.randn(16, 16)
    rigl_mask_step2 = rigl_update_mask(w, grad_step2, rigl_mask_step1, drop_fraction=0.25)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "w": w,
        "grad": grad,
        "w_2_to_4": w_2_to_4,
        "mask_2_to_4": mask_2_to_4,
        "packed_values": packed_values,
        "packed_indices": packed_indices,
        "initial_mask": initial_mask,
        "rigl_mask_step1": rigl_mask_step1,
        "rigl_mask_step2": rigl_mask_step2,
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
