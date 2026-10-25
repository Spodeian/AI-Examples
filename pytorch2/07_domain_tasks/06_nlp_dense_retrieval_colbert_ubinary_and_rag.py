"""Nlp Dense Retrieval Colbert Ubinary and RAG — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag
Golden weights: weights/07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag.safetensors

``python pytorch2/07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def colbert_maxsim(
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
) -> torch.Tensor:
    r"""ColBERT Multi-Vector Late Interaction (MaxSim):
    Score(Q, D) = sum_i max_j (q_i . d_j)
    query_embeds: [B, Q_len, D]
    doc_embeds: [B, D_len, D]
    Returns MaxSim score: [B]
    """
    # [B, Q_len, D] @ [B, D, D_len] -> [B, Q_len, D_len]
    sim_matrix = query_embeds @ doc_embeds.transpose(-2, -1)
    max_per_query = torch.max(sim_matrix, dim=-1).values # [B, Q_len]
    return torch.sum(max_per_query, dim=-1) # [B]


def ubinary_quantize_and_dot(
    q_float: torch.Tensor,
    d_float: torch.Tensor,
) -> torch.Tensor:
    r"""1-Bit Unsigned Binary (UBinary) Quantization & Similarity:
    b = sign(x) in {-1, +1}
    Dot product: sum_k (b_q * b_d)
    q_float, d_float: [B, D]
    Returns score: [B]
    """
    q_b = torch.where(q_float >= 0, 1.0, -1.0)
    d_b = torch.where(d_float >= 0, 1.0, -1.0)
    return torch.sum(q_b * d_b, dim=-1)


def reciprocal_rank_fusion(
    ranks1: torch.Tensor,
    ranks2: torch.Tensor,
    k: float = 60.0,
) -> torch.Tensor:
    r"""Reciprocal Rank Fusion (RRF) for hybrid sparse-dense retrieval:
    RRF(d) = 1 / (k + rank1) + 1 / (k + rank2)
    ranks1, ranks2: [B, N]
    """
    return (1.0 / (k + ranks1)) + (1.0 / (k + ranks2))


def dense_infonce_loss(
    query_vecs: torch.Tensor,
    pos_vecs: torch.Tensor,
    tau: float = 0.05,
) -> torch.Tensor:
    r"""In-Batch Contrastive InfoNCE Loss for Dense Bi-Encoders:
    L = -log( exp(q_i . p_i / tau) / sum_j exp(q_i . p_j / tau) )
    query_vecs: [B, D]
    pos_vecs: [B, D]
    """
    b = query_vecs.shape[0]
    logits = (query_vecs @ pos_vecs.t()) / tau # [B, B]
    targets = torch.arange(b, device=query_vecs.device)
    return torch.nn.functional.cross_entropy(logits, targets)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B = 2
    Q_len = 3
    D_len = 5
    D = 4
    N = 4
    
    # 0. Linear anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)
    
    # 1. ColBERT MaxSim
    q_raw = torch.randn(B, Q_len, D)
    d_raw = torch.randn(B, D_len, D)
    query_embeds = torch.nn.functional.normalize(q_raw, p=2, dim=-1)
    doc_embeds = torch.nn.functional.normalize(d_raw, p=2, dim=-1)
    maxsim_ref = colbert_maxsim(query_embeds, doc_embeds)
    
    # 2. UBinary 1-bit dot product
    q_float = torch.randn(B, D)
    d_float = torch.randn(B, D)
    ubinary_ref = ubinary_quantize_and_dot(q_float, d_float)
    
    # 3. Reciprocal Rank Fusion
    ranks1 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]])
    ranks2 = torch.tensor([[2.0, 1.0, 4.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    rrf_ref = reciprocal_rank_fusion(ranks1, ranks2, k=60.0)
    
    # 4. Dense InfoNCE Loss
    q_vecs = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=-1)
    p_vecs = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=-1)
    infonce_ref = dense_infonce_loss(q_vecs, p_vecs, tau=0.05)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        
        "query_embeds": query_embeds,
        "doc_embeds": doc_embeds,
        "maxsim_ref": maxsim_ref,
        
        "q_float": q_float,
        "d_float": d_float,
        "ubinary_ref": ubinary_ref,
        
        "ranks1": ranks1,
        "ranks2": ranks2,
        "rrf_ref": rrf_ref,
        
        "q_vecs": q_vecs,
        "p_vecs": p_vecs,
        "infonce_ref": infonce_ref,
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
