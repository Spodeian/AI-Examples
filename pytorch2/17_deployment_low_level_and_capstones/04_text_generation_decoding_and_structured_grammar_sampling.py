"""Text Generation Decoding and Structured Grammar Sampling — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling
Golden weights: weights/17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"

MASK_VALUE = -10000.0


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def apply_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor,
    penalty: float = 1.25,
) -> torch.Tensor:
    """Applies Keskar et al. multiplicative repetition penalty based on past token history.
    
    For token t in history:
      if logit > 0: logit = logit / penalty
      else: logit = logit * penalty
    """
    out = logits.clone()
    batch_size = logits.shape[0]
    for b in range(batch_size):
        for tok in history[b]:
            idx = int(tok.item())
            val = out[b, idx]
            if val > 0:
                out[b, idx] = val / penalty
            else:
                out[b, idx] = val * penalty
    return out


def apply_grammar_mask(
    logits: torch.Tensor,
    grammar_mask: torch.Tensor,
    mask_value: float = MASK_VALUE,
) -> torch.Tensor:
    """Pre-softmax grammar/FSM bitmask: invalid transition tokens are set to mask_value."""
    return torch.where(grammar_mask > 0.5, logits, torch.full_like(logits, mask_value))


def apply_temperature(logits: torch.Tensor, temperature: float = 0.8) -> torch.Tensor:
    """Scales logits by temperature T before softmax."""
    return logits / max(temperature, 1e-5)


def apply_top_k(logits: torch.Tensor, k: int = 4, mask_value: float = MASK_VALUE) -> torch.Tensor:
    """Keeps top-k logits per row, masking out all lower logits to mask_value."""
    val, _ = torch.topk(logits, k=k, dim=-1)
    k_th = val[:, -1:]
    return torch.where(logits >= k_th, logits, torch.full_like(logits, mask_value))


def apply_min_p(probs: torch.Tensor, min_p: float = 0.1) -> torch.Tensor:
    """Min-P sampling filter: retains tokens with p_i >= p_max * min_p, then renormalizes."""
    p_max, _ = torch.max(probs, dim=-1, keepdim=True)
    threshold = p_max * min_p
    filtered = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    sums = torch.sum(filtered, dim=-1, keepdim=True).clamp_min(1e-12)
    return filtered / sums


def decoding_pipeline(
    logits: torch.Tensor,
    history: torch.Tensor,
    grammar_mask: torch.Tensor,
    temperature: float = 0.8,
    repetition_penalty: float = 1.25,
    top_k: int = 4,
    min_p: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete decoding pipeline: rep penalty -> grammar mask -> temp -> top-k -> softmax -> min-p."""
    x = apply_repetition_penalty(logits, history, repetition_penalty)
    x = apply_grammar_mask(x, grammar_mask)
    x = apply_temperature(x, temperature)
    x = apply_top_k(x, top_k)
    probs = F.softmax(x, dim=-1)
    probs = apply_min_p(probs, min_p)
    greedy_tokens = torch.argmax(probs, dim=-1).to(torch.float32)
    return probs, greedy_tokens


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    batch_size = 2
    vocab_size = 16
    seq_len = 4

    logits = torch.randn(batch_size, vocab_size)
    # Token history with token IDs in [0, vocab_size)
    history = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Grammar mask: binary mask with ~50% allowed transitions
    grammar_mask = (torch.rand(batch_size, vocab_size) > 0.4).to(torch.float32)
    # Ensure at least one token is allowed
    grammar_mask[:, 0] = 1.0

    rep_penalized = apply_repetition_penalty(logits, history, 1.25)
    masked_logits = apply_grammar_mask(rep_penalized, grammar_mask)
    temp_logits = apply_temperature(masked_logits, 0.8)
    top_k_logits = apply_top_k(temp_logits, 4)
    probs, greedy_tokens = decoding_pipeline(
        logits, history, grammar_mask, temperature=0.8, repetition_penalty=1.25, top_k=4, min_p=0.1
    )

    linear_w = torch.randn(vocab_size, vocab_size) * 0.1
    linear_b = torch.randn(vocab_size) * 0.1
    test_out = linear_anchor(probs, linear_w, linear_b)

    tensors = {
        "logits": logits,
        "history": history.to(torch.float32),
        "grammar_mask": grammar_mask,
        "rep_penalized": rep_penalized,
        "masked_logits": masked_logits,
        "temp_logits": temp_logits,
        "top_k_logits": top_k_logits,
        "probs": probs,
        "greedy_tokens": greedy_tokens,
        "linear_w": linear_w,
        "linear_b": linear_b,
        "test_input": probs.clone(),
        "weight": linear_w.clone(),
        "bias": linear_b.clone(),
        "test_output": test_out,
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
