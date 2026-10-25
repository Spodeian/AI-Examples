"""Nlp Sequence Tagging Bilstm Crf and Span Ner — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner
Golden weights: weights/07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner.safetensors

``python pytorch2/07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def crf_log_partition(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    r"""Linear-Chain CRF Partition Function (Log-Sum-Exp forward pass):
    emissions: [B, T, K]
    transitions: [K, K] where transitions[i, j] is score for transition i -> j.
    Returns log Z: [B]
    """
    b, t_len, k = emissions.shape
    alpha = emissions[:, 0, :] # [B, K]
    trans = transitions.unsqueeze(0) # [1, K, K]

    for t in range(1, t_len):
        emit = emissions[:, t, :].unsqueeze(1) # [B, 1, K]
        # alpha.unsqueeze(2): [B, K, 1], trans: [1, K, K], emit: [B, 1, K]
        # scores[b, i, j] = alpha[b, i] + trans[i, j] + emit[b, j]
        scores = alpha.unsqueeze(2) + trans + emit # [B, K, K]
        alpha = torch.logsumexp(scores, dim=1) # sum over i -> [B, K]

    return torch.logsumexp(alpha, dim=-1) # [B]


def crf_sequence_score(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
    tags: torch.Tensor,
) -> torch.Tensor:
    r"""Score of gold tag sequence under CRF:
    Score = sum_t emit(t, tag_t) + sum_t trans(tag_t, tag_{t+1})
    tags: [B, T]
    Returns score: [B]
    """
    b, t_len, _k = emissions.shape
    score = torch.zeros(b, dtype=emissions.dtype, device=emissions.device)

    # First emission
    first_tags = tags[:, 0]
    score = score + emissions[torch.arange(b), 0, first_tags]

    for t in range(1, t_len):
        prev_tag = tags[:, t - 1]
        curr_tag = tags[:, t]
        emit_score = emissions[torch.arange(b), t, curr_tag]
        trans_score = transitions[prev_tag, curr_tag]
        score = score + emit_score + trans_score

    return score


def crf_nll_loss(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
    tags: torch.Tensor,
) -> torch.Tensor:
    r"""CRF Negative Log-Likelihood Loss:
    Loss = log Z - Score(y)
    """
    log_z = crf_log_partition(emissions, transitions)
    gold_score = crf_sequence_score(emissions, transitions, tags)
    return log_z - gold_score


def span_endpoint_features(
    seq_embeds: torch.Tensor,
    start_idx: torch.Tensor,
    end_idx: torch.Tensor,
) -> torch.Tensor:
    r"""Span-based NER boundary feature representation:
    span = [H[start]; H[end]; H[end] - H[start]]
    seq_embeds: [B, T, D]
    start_idx, end_idx: [B, S]
    Returns span representations: [B, S, 3 * D]
    """
    b, s_len = start_idx.shape
    d = seq_embeds.shape[-1]
    
    batch_idx = torch.arange(b).unsqueeze(1).expand(b, s_len)
    h_start = seq_embeds[batch_idx, start_idx] # [B, S, D]
    h_end = seq_embeds[batch_idx, end_idx] # [B, S, D]
    h_diff = h_end - h_start
    return torch.cat([h_start, h_end, h_diff], dim=-1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B = 2
    T = 4
    K = 3
    D = 4
    S = 2
    
    # 0. Linear anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)
    
    # 1. CRF components
    emissions = torch.randn(B, T, K) * 0.8
    transitions = torch.randn(K, K) * 0.5
    tags = torch.randint(0, K, (B, T))
    
    log_z = crf_log_partition(emissions, transitions)
    gold_score = crf_sequence_score(emissions, transitions, tags)
    crf_loss = crf_nll_loss(emissions, transitions, tags)
    
    # 2. Span NER representation
    seq_embeds = torch.randn(B, T, D) * 0.7
    start_idx = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    end_idx = torch.tensor([[2, 3], [2, 3]], dtype=torch.long)
    span_feats = span_endpoint_features(seq_embeds, start_idx, end_idx)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        
        "emissions": emissions,
        "transitions": transitions,
        "tags": tags.float(),
        "log_z": log_z,
        "gold_score": gold_score,
        "crf_loss": crf_loss,
        
        "seq_embeds": seq_embeds,
        "start_idx": start_idx.float(),
        "end_idx": end_idx.float(),
        "span_feats": span_feats,
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
