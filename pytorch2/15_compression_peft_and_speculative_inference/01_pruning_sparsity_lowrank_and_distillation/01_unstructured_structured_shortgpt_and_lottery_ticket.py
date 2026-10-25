"""Unstructured Structured Shortgpt and Lottery Ticket — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket
Golden weights: weights/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def unstructured_magnitude_prune(w: torch.Tensor, sparsity: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Prunes individual smallest magnitude weights regardless of structure.

    Args:
        w: Weight tensor of shape [rows, cols].
        sparsity: Fraction of weights to prune to zero (default 0.5).

    Returns:
        w_pruned: Weight tensor with pruned entries zeroed.
        mask: Binary indicator tensor (1 = kept, 0 = pruned).
    """
    k = max(1, int(sparsity * w.numel()))
    tau = torch.kthvalue(torch.abs(w).flatten(), k).values
    mask = (torch.abs(w) >= tau).to(torch.float32)
    return w * mask, mask


def structured_row_prune(w: torch.Tensor, keep_ratio: float = 0.75) -> tuple[torch.Tensor, torch.Tensor]:
    """Prunes entire rows/channels with smallest L2 norm.

    Args:
        w: Weight tensor of shape [rows, cols].
        keep_ratio: Fraction of rows to retain.

    Returns:
        w_sliced: Physically sliced weight tensor of shape [k_rows, cols].
        indices: 1D indices of retained rows.
    """
    row_norms = torch.norm(w, p=2, dim=1)
    k_rows = max(1, int(keep_ratio * w.shape[0]))
    indices = torch.topk(row_norms, k_rows, largest=True).indices.sort().values
    w_sliced = w[indices, :]
    return w_sliced, indices.to(torch.float32)


def compute_block_influence(x_in: torch.Tensor, x_out: torch.Tensor) -> float:
    """Computes ShortGPT Block Influence (BI): 1 - mean(cosine_sim(x_in, x_out))."""
    dot = torch.sum(x_in * x_out, dim=-1)
    norm_in = torch.norm(x_in, p=2, dim=-1)
    norm_out = torch.norm(x_out, p=2, dim=-1)
    cos = dot / (norm_in * norm_out + 1e-8)
    return float(1.0 - torch.mean(cos))


def lottery_ticket_rewind(
    w_initial: torch.Tensor, w_trained: torch.Tensor, sparsity: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts winning ticket subnetwork: mask from trained weights applied to initial weights."""
    k = max(1, int(sparsity * w_trained.numel()))
    tau = torch.kthvalue(torch.abs(w_trained).flatten(), k).values
    mask = (torch.abs(w_trained) >= tau).to(torch.float32)
    return w_initial * mask, mask


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    w = torch.randn(16, 16) * 0.1
    w_trained = w + torch.randn(16, 16) * 0.05
    x_in = torch.randn(8, 16)
    x_out_red = x_in + 0.01 * torch.randn(8, 16)
    x_out_act = x_in + 0.5 * torch.randn(8, 16)

    # 1. Unstructured Pruning
    w_unstr, mask_unstr = unstructured_magnitude_prune(w, sparsity=0.5)

    # 2. Structured Pruning
    w_str, indices_str = structured_row_prune(w, keep_ratio=0.75)

    # 3. ShortGPT Block Influence
    bi_red = compute_block_influence(x_in, x_out_red)
    bi_act = compute_block_influence(x_in, x_out_act)
    bi_scores = torch.tensor([bi_red, bi_act], dtype=torch.float32)

    # 4. Lottery Ticket
    w_subnetwork, mask_lottery = lottery_ticket_rewind(w, w_trained, sparsity=0.5)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "w": w,
        "w_trained": w_trained,
        "x_in": x_in,
        "x_out_red": x_out_red,
        "x_out_act": x_out_act,
        "w_unstructured": w_unstr,
        "mask_unstructured": mask_unstr,
        "w_structured": w_str,
        "indices_structured": indices_str,
        "w_lottery_subnetwork": w_subnetwork,
        "mask_lottery": mask_lottery,
        "bi_scores": bi_scores,
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
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
