"""Nlp Tokenizers Bpe Wordpiece Unigram and Byte Latent — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent
Golden weights: weights/07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent.safetensors

``python pytorch2/07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def wordpiece_pair_score(
    bigram_counts: torch.Tensor,
    unigram_counts: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""WordPiece Mutual Information Pair Score:
    S_{ij} = C_{ij} / (C_i * C_j + eps)
    bigram_counts: [V, V]
    unigram_counts: [V]
    """
    outer = unigram_counts.unsqueeze(1) * unigram_counts.unsqueeze(0) # [V, V]
    return bigram_counts / (outer + eps)


def byte_patch_pooling(
    byte_embeddings: torch.Tensor,
    patch_size: int = 4,
) -> torch.Tensor:
    r"""Byte-Level Model Patch Aggregator (MegaByte / BLT):
    byte_embeddings: [B, T, D] where T is divisible by patch_size.
    Returns patch-level embeddings: [B, T // patch_size, D] via average pooling.
    """
    b, t, d = byte_embeddings.shape
    num_patches = t // patch_size
    reshaped = byte_embeddings.view(b, num_patches, patch_size, d)
    return torch.mean(reshaped, dim=2)


def entropy_boundary_detector(
    probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Local Byte Entropy for Dynamic Boundary Detection (BLT):
    H = -sum_v p_v * log2(p_v + eps)
    probs: [B, T, V]
    Returns entropy: [B, T]
    """
    log_probs = torch.log2(probs + eps)
    return -torch.sum(probs * log_probs, dim=-1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B = 2
    V = 8
    T = 16
    D = 8
    patch_size = 4
    
    # 0. Linear anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)
    
    # 1. WordPiece Pair Score
    bigram_counts = torch.rand(V, V) * 100.0 + 5.0
    unigram_counts = torch.sum(bigram_counts, dim=1) + 10.0
    wp_scores = wordpiece_pair_score(bigram_counts, unigram_counts)
    
    # 2. Byte Patch Pooling
    byte_embeds = torch.randn(B, T, D) * 0.5
    patch_embeds = byte_patch_pooling(byte_embeds, patch_size=patch_size)
    
    # 3. Dynamic Boundary Entropy
    logits = torch.randn(B, T, V)
    probs = torch.softmax(logits, dim=-1)
    entropy_ref = entropy_boundary_detector(probs)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        
        "bigram_counts": bigram_counts,
        "unigram_counts": unigram_counts,
        "wp_scores": wp_scores,
        
        "byte_embeds": byte_embeds,
        "patch_embeds": patch_embeds,
        
        "probs": probs,
        "entropy_ref": entropy_ref,
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
