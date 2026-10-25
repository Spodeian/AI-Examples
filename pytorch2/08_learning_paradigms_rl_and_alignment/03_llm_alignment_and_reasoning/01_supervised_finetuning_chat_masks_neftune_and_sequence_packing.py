"""Supervised Finetuning Chat Masks Neftune and Sequence Packing — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing
Golden weights: weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def neftune_add_noise(
    embeddings: torch.Tensor,
    noise: torch.Tensor,
    alpha: float = 5.0,
) -> torch.Tensor:
    """NEFTune: Noisy Embedding Instruction Fine-Tuning.
    Adds scaled uniform noise to token embeddings to regularize conversational fine-tuning:
    e_noisy = e + (alpha / sqrt(L * D)) * noise
    embeddings: [B, L, D]
    noise:      [B, L, D] uniform noise in [-1, 1]
    alpha:      hyperparameter noise scale (typically 5.0)
    Returns:    [B, L, D] perturbed embeddings.
    """
    b, seq_len, d_dim = embeddings.shape
    scale = alpha / math.sqrt(seq_len * d_dim)
    return embeddings + scale * noise


def masked_completion_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Chat template completion-only loss masking:
    Evaluates cross-entropy loss exclusively on assistant response tokens, masking out prompt/system tokens.
    logits:    [B, L, V] vocabulary logits
    targets:   [B, L] ground truth target token ids
    loss_mask: [B, L] binary mask (1.0 for completion tokens, 0.0 for prompt tokens)
    Returns:   scalar masked cross-entropy loss.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    nll = -target_log_probs
    masked_nll = nll * loss_mask
    return torch.sum(masked_nll) / (torch.sum(loss_mask) + eps)


def packed_block_diagonal_causal_mask(
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Sequence Packing: 2D Block-Diagonal Causal Attention Mask.
    Prevents cross-sequence attention leakage when concatenating multiple sequences into one packed buffer.
    segment_ids: [L] integer document/sequence identifier per token (e.g. [0, 0, 1, 1, 1, 2])
    Returns:     [L, L] binary causal mask (1.0 = attend, 0.0 = mask).
    """
    seq_len = segment_ids.shape[0]
    # Causal lower-triangular mask
    indices = torch.arange(seq_len, device=segment_ids.device)
    causal_mask = indices.unsqueeze(0) <= indices.unsqueeze(1)  # [L, L] where j <= i

    # Same segment mask
    same_segment = segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)  # [L, L] where seg[j] == seg[i]

    # Combined mask: attend iff j <= i and seg[j] == seg[i]
    return (causal_mask & same_segment).float()


def packed_position_ids(
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Sequence Packing: Reset Position IDs per Sub-Sequence.
    Generates [0, 1, 2, ..., len_i - 1] position IDs for each packed sequence in the contiguous buffer.
    segment_ids: [L] integer document/sequence identifier per token
    Returns:     [L] integer position IDs resetting at each sequence boundary.
    """
    seq_len = segment_ids.shape[0]
    pos_ids = torch.zeros(seq_len, dtype=torch.long, device=segment_ids.device)
    current_pos = 0

    for i in range(seq_len):
        if i > 0 and segment_ids[i] != segment_ids[i - 1]:
            current_pos = 0
        pos_ids[i] = current_pos
        current_pos += 1

    return pos_ids


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. NEFTune Noise Injection
    # B=4, L=8, D=16
    embeddings = torch.randn(4, 8, 16)
    noise = torch.rand(4, 8, 16) * 2.0 - 1.0  # Uniform(-1, 1)
    neftune_noisy_ref = neftune_add_noise(embeddings, noise, alpha=5.0)

    # 2. Masked Completion Cross-Entropy
    # B=4, L=8, V=32
    logits = torch.randn(4, 8, 32) * 2.0
    targets = torch.randint(0, 32, (4, 8))
    # Prompt tokens (first 3 tokens) masked out (0.0), completion tokens (last 5) active (1.0)
    loss_mask = torch.zeros(4, 8)
    loss_mask[:, 3:] = 1.0
    masked_loss_ref = masked_completion_cross_entropy(logits, targets, loss_mask)

    # 3. Packed Block-Diagonal Causal Mask & Position IDs
    # Packed sequence of length 8 consisting of 3 sequences: lens = [3, 2, 3]
    segment_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
    packed_mask_ref = packed_block_diagonal_causal_mask(segment_ids)
    packed_positions_ref = packed_position_ids(segment_ids)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "embeddings": embeddings,
        "noise": noise,
        "neftune_noisy_ref": neftune_noisy_ref,
        "logits": logits,
        "targets": targets,
        "loss_mask": loss_mask,
        "masked_loss_ref": masked_loss_ref,
        "segment_ids": segment_ids,
        "packed_mask_ref": packed_mask_ref,
        "packed_positions_ref": packed_positions_ref,
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
