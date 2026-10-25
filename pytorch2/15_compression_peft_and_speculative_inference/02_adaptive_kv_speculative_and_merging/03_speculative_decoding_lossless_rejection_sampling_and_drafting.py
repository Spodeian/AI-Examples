"""Speculative Decoding Lossless Rejection Sampling and Drafting — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting
Golden weights: weights/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def speculative_rejection_step(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    uniform_rands: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute lossless speculative rejection sampling across gamma candidate tokens.

    Args:
        target_probs: Target model probabilities of shape [batch, gamma, vocab].
        draft_probs: Draft model probabilities of shape [batch, gamma, vocab].
        draft_tokens: Candidate token IDs of shape [batch, gamma].
        uniform_rands: Deterministic uniform samples in [0, 1] of shape [batch, gamma].

    Returns:
        accepted_mask: Binary indicator of shape [batch, gamma] (1 = accepted, 0 = rejected).
        num_accepted: Count of sequentially accepted tokens of shape [batch].
        residual_probs: Adjusted target distribution for the first rejected token, shape [batch, vocab].
    """
    batch, gamma, vocab = target_probs.shape
    accepted_mask = torch.zeros((batch, gamma), dtype=torch.float32, device=target_probs.device)
    num_accepted = torch.zeros((batch,), dtype=torch.float32, device=target_probs.device)
    residual_probs = torch.zeros((batch, vocab), dtype=torch.float32, device=target_probs.device)

    for b in range(batch):
        rejected_pos = -1
        for i in range(gamma):
            t_id = draft_tokens[b, i].item()
            p_val = target_probs[b, i, t_id].item()
            q_val = draft_probs[b, i, t_id].item()
            ratio = min(1.0, p_val / max(q_val, 1e-8))

            if uniform_rands[b, i].item() <= ratio:
                accepted_mask[b, i] = 1.0
                num_accepted[b] += 1.0
            else:
                rejected_pos = i
                break

        if rejected_pos >= 0:
            # Resample from adjusted distribution: max(0, p - q) normalized
            diff = torch.clamp(target_probs[b, rejected_pos] - draft_probs[b, rejected_pos], min=0.0)
            norm = torch.sum(diff)
            if norm > 1e-8:
                residual_probs[b] = diff / norm
            else:
                residual_probs[b] = target_probs[b, rejected_pos]
        else:
            # All gamma tokens accepted; next token comes from target_probs at final step
            residual_probs[b] = target_probs[b, -1]

    return accepted_mask, num_accepted, residual_probs


def speculative_speedup_theoretical(
    gamma: int,
    beta: float,
    cost_ratio: float,
) -> float:
    """Compute theoretical speculative decoding wall-clock speedup.

    Args:
        gamma: Speculative draft length (lookahead horizon).
        beta: Acceptance rate (mean token agreement prob).
        cost_ratio: Relative cost ratio cost(M_draft) / cost(M_target).

    Returns:
        speedup: Wall-clock speedup factor relative to autoregressive target baseline.
    """
    if abs(beta - 1.0) < 1e-6:
        expected_tokens = float(gamma + 1)
    else:
        expected_tokens = (1.0 - beta ** (gamma + 1)) / (1.0 - beta)

    step_cost = gamma * cost_ratio + 1.0
    return expected_tokens / step_cost


def prompt_lookup_ngram_draft(
    input_ids: torch.Tensor,
    ngram_size: int = 3,
    gamma: int = 4,
) -> torch.Tensor:
    """Zero-parameter speculative drafting via n-gram prompt prefix lookup.

    Scans the prompt history for matches to the current suffix n-gram and extracts
    the subsequent gamma tokens as speculative draft candidates.

    Args:
        input_ids: 1D sequence tensor of tokens [seq_len].
        ngram_size: Match window size.
        gamma: Number of candidate tokens to draft.

    Returns:
        draft_tokens: Selected candidate tokens of shape [gamma].
    """
    seq_len = input_ids.shape[0]
    if seq_len < ngram_size + 1:
        return torch.zeros(gamma, dtype=torch.int64, device=input_ids.device)

    suffix = input_ids[-ngram_size:]
    best_pos = -1

    # Search backwards for the most recent matching n-gram preceding suffix
    for i in range(seq_len - ngram_size - 1, -1, -1):
        if torch.equal(input_ids[i : i + ngram_size], suffix):
            best_pos = i + ngram_size
            break

    draft = []
    if best_pos >= 0:
        for k in range(gamma):
            idx = best_pos + k
            if idx < seq_len:
                draft.append(input_ids[idx].item())
            else:
                draft.append(0)
    else:
        draft = [0] * gamma

    return torch.tensor(draft, dtype=torch.int64, device=input_ids.device)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Speculative Rejection Sampling (batch=3, gamma=4, vocab=8)
    batch = 3
    gamma = 4
    vocab = 8
    target_logits = torch.randn(batch, gamma, vocab)
    draft_logits = torch.randn(batch, gamma, vocab)
    target_probs = F.softmax(target_logits, dim=-1)
    draft_probs = F.softmax(draft_logits, dim=-1)

    draft_tokens = torch.tensor([
        [2, 5, 1, 0],
        [3, 4, 7, 2],
        [1, 2, 3, 4],
    ], dtype=torch.int64)

    uniform_rands = torch.tensor([
        [0.10, 0.20, 0.95, 0.50],  # Accepts token 0, 1; rejects at token 2
        [0.05, 0.10, 0.15, 0.20],  # Low rands -> all 4 accepted
        [0.90, 0.50, 0.10, 0.05],  # High rand at pos 0 -> immediately rejects token 0
    ], dtype=torch.float32)

    accepted_mask, num_accepted, residual_probs = speculative_rejection_step(
        target_probs, draft_probs, draft_tokens, uniform_rands
    )

    # 2. Speculative Speedup Theoretical
    speedup_curve = torch.tensor([
        speculative_speedup_theoretical(gamma=4, beta=0.8, cost_ratio=0.05),
        speculative_speedup_theoretical(gamma=4, beta=0.6, cost_ratio=0.05),
        speculative_speedup_theoretical(gamma=6, beta=0.8, cost_ratio=0.05),
    ], dtype=torch.float32)

    # 3. Prompt Lookup Drafting
    # Sequence with repeated phrase: [10, 20, 30, 42, 55, 60, 10, 20, 30] -> suffix is [10, 20, 30], next is [42, 55, 60, 0]
    prompt_seq = torch.tensor([1, 5, 9, 10, 20, 30, 42, 55, 60, 77, 88, 10, 20, 30], dtype=torch.int64)
    draft_ngram = prompt_lookup_ngram_draft(prompt_seq, ngram_size=3, gamma=4)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "target_probs": target_probs,
        "draft_probs": draft_probs,
        "draft_tokens": draft_tokens.to(torch.float32),
        "uniform_rands": uniform_rands,
        "accepted_mask": accepted_mask,
        "num_accepted": num_accepted,
        "residual_probs": residual_probs,
        "speedup_curve": speedup_curve,
        "prompt_seq": prompt_seq.to(torch.float32),
        "draft_ngram": draft_ngram.to(torch.float32),
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
