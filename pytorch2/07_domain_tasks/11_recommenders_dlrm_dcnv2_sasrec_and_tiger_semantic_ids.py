"""Recommenders Dlrm Dcnv2 Sasrec and Tiger Semantic Ids — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids
Golden weights: weights/07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids.safetensors

``python pytorch2/07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dlrm_feature_interaction(v_dense: torch.Tensor, v_sparse: torch.Tensor) -> torch.Tensor:
    """DLRM Explicit Pairwise Dot-Product Feature Interaction:
    v_dense: [B, D]
    v_sparse: [B, S, D]
    Returns: [B, D + S*(S+1)/2]
    """
    b, d = v_dense.shape
    s = v_sparse.shape[1]
    # Stack: [B, 1 + S, D]
    v_all = torch.cat([v_dense.unsqueeze(1), v_sparse], dim=1)
    # Dot product matrix: [B, 1 + S, 1 + S]
    interaction_mat = torch.bmm(v_all, v_all.transpose(1, 2))

    # Extract lower triangular elements (strictly below diagonal)
    tril_indices = torch.tril_indices(s + 1, s + 1, offset=-1)
    interactions = interaction_mat[:, tril_indices[0], tril_indices[1]]  # [B, (1+S)*S/2]

    # Concatenate dense features with interactions
    return torch.cat([v_dense, interactions], dim=1)


def dcn_v2_cross_layer(
    x0: torch.Tensor,
    xl: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """DCN-v2 (Deep & Cross Network v2) Cross Layer:
    x_{l+1} = x0 * (xl @ weight + bias) + xl
    x0, xl: [B, D]
    weight: [D, D]
    bias: [D]
    Returns: [B, D]
    """
    proj = xl @ weight + bias
    return x0 * proj + xl


def sasrec_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SASRec Causally-Masked Self-Attention for Sequential Recommendation.
    q, k, v: [B, T, D]
    Returns: [B, T, D]
    """
    d = q.shape[-1]
    t = q.shape[1]
    scores = torch.bmm(q, k.transpose(1, 2)) / (d ** 0.5)  # [B, T, T]

    # Causal mask: upper triangle is -inf
    mask = torch.triu(torch.ones(t, t, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

    attn_weights = F.softmax(scores, dim=-1)
    return torch.bmm(attn_weights, v)


def rq_vae_quantize_step(
    residual: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TIGER / RQ-VAE Residual Vector Quantization Step:
    residual: [B, D]
    codebook: [V, D]
    Returns: (quantized [B, D], indices [B], next_residual [B, D])
    """
    # Squared Euclidean distances: ||r - c||^2 = ||r||^2 - 2 r c^T + ||c||^2
    r_sq = torch.sum(residual ** 2, dim=-1, keepdim=True)  # [B, 1]
    c_sq = torch.sum(codebook ** 2, dim=-1, keepdim=True).t()  # [1, V]
    dist = r_sq - 2.0 * (residual @ codebook.t()) + c_sq  # [B, V]

    indices = torch.argmin(dist, dim=-1)  # [B]
    quantized = codebook[indices]  # [B, D]
    next_residual = residual - quantized
    return quantized, indices, next_residual


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }

    # 1. DLRM Feature Interaction
    b, d, s = 2, 4, 3
    v_dense = torch.randn(b, d)
    v_sparse = torch.randn(b, s, d)
    dlrm_ref = dlrm_feature_interaction(v_dense, v_sparse)
    tensors["v_dense"] = v_dense
    tensors["v_sparse"] = v_sparse
    tensors["dlrm_ref"] = dlrm_ref

    # 2. DCN-v2 Cross Layer
    x0 = torch.randn(b, d)
    xl = torch.randn(b, d)
    w_cross = torch.randn(d, d) * 0.1
    b_cross = torch.randn(d) * 0.1
    dcn_ref = dcn_v2_cross_layer(x0, xl, w_cross, b_cross)
    tensors["x0"] = x0
    tensors["xl"] = xl
    tensors["w_cross"] = w_cross
    tensors["b_cross"] = b_cross
    tensors["dcn_ref"] = dcn_ref

    # 3. SASRec Causal Attention
    t_seq, d_attn = 4, 8
    q = torch.randn(b, t_seq, d_attn)
    k = torch.randn(b, t_seq, d_attn)
    v = torch.randn(b, t_seq, d_attn)
    sasrec_ref = sasrec_causal_attention(q, k, v)
    tensors["q"] = q
    tensors["k"] = k
    tensors["v"] = v
    tensors["sasrec_ref"] = sasrec_ref

    # 4. RQ-VAE Quantize Step
    v_vocab, d_rq = 16, 8
    residual = torch.randn(b, d_rq)
    codebook = torch.randn(v_vocab, d_rq) * 0.5
    quantized_ref, indices_ref, next_res_ref = rq_vae_quantize_step(residual, codebook)
    tensors["residual"] = residual
    tensors["codebook"] = codebook
    tensors["quantized_ref"] = quantized_ref
    tensors["indices_ref"] = indices_ref
    tensors["next_res_ref"] = next_res_ref

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
