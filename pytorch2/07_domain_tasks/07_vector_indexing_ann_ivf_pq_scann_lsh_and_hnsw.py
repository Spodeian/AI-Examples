"""Vector Indexing Ann IVF PQ Scann Lsh and HNSW — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw
Golden weights: weights/07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw.safetensors

``python pytorch2/07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def pq_distance_table(
    query_subvectors: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    r"""Product Quantization (PQ) Asymmetric Distance Table:
    query_subvectors: [B, M, d]
    codebooks: [M, K, d]
    Returns squared L2 distance table: [B, M, K]
    """
    # [B, M, 1, d] - [1, M, K, d] -> [B, M, K, d]
    diff = query_subvectors.unsqueeze(2) - codebooks.unsqueeze(0)
    return torch.sum(diff**2, dim=-1)


def pq_asymmetric_distance(
    dist_table: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    r"""Asymmetric Distance Computation (ADC) via table lookup:
    dist_table: [B, M, K]
    codes: [B, N_docs, M] with integer cluster indices in [0, K-1]
    Returns accumulated L2 distances: [B, N_docs]
    """
    b, n_docs, m = codes.shape
    codes_long = codes.long()
    
    # We gather from dist_table: [B, M, K]
    # Transpose dist_table to [B, K, M] or iterate over M
    out = torch.zeros(b, n_docs, dtype=dist_table.dtype, device=dist_table.device)
    for mi in range(m):
        c_m = codes_long[:, :, mi] # [B, N_docs]
        t_m = dist_table[:, mi, :] # [B, K]
        # Gather distances: [B, N_docs]
        d_m = torch.gather(t_m, 1, c_m)
        out = out + d_m
    return out


def scann_anisotropic_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    omega: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""ScaNN Anisotropic Vector Quantization Distortion Loss (Guo et al.):
    L = ||e_perp||^2 + (1 + omega) * ||e_parallel||^2
    where e = x - x_tilde, e_parallel = (e . x / ||x||^2) * x, e_perp = e - e_parallel
    x, x_tilde: [B, D]
    Returns loss: [B]
    """
    e = x - x_tilde # [B, D]
    x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True) + eps # [B, 1]
    dot_ex = torch.sum(e * x, dim=-1, keepdim=True) # [B, 1]
    e_parallel = (dot_ex / x_norm_sq) * x # [B, D]
    e_perp = e - e_parallel # [B, D]
    
    loss_perp = torch.sum(e_perp**2, dim=-1) # [B]
    loss_parallel = torch.sum(e_parallel**2, dim=-1) # [B]
    return loss_perp + (1.0 + omega) * loss_parallel


def lsh_hash_hamming_distance(
    q: torch.Tensor,
    d: torch.Tensor,
    hyperplanes: torch.Tensor,
) -> torch.Tensor:
    r"""Locality-Sensitive Hashing (LSH) Random Projection Cosine Similarity:
    b_q = sign(q @ R), b_d = sign(d @ R)
    Sim = mean(b_q * b_d)
    q, d: [B, D]
    hyperplanes: [D, N_bits]
    Returns similarity in [-1, +1]: [B]
    """
    bits_q = torch.where(q @ hyperplanes >= 0, 1.0, -1.0) # [B, N_bits]
    bits_d = torch.where(d @ hyperplanes >= 0, 1.0, -1.0) # [B, N_bits]
    return torch.mean(bits_q * bits_d, dim=-1) # [B]


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B = 2
    M = 4 # subvectors
    K = 8 # centroids per subvector
    d_sub = 2 # subvector dimension
    N_docs = 6
    D = M * d_sub # total dim = 8
    N_bits = 16
    
    # 0. Linear anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)
    
    # 1. Product Quantization
    query_subvectors = torch.randn(B, M, d_sub) * 0.7
    codebooks = torch.randn(M, K, d_sub) * 0.7
    dist_table = pq_distance_table(query_subvectors, codebooks)
    
    codes = torch.randint(0, K, (B, N_docs, M)).float()
    adc_distances = pq_asymmetric_distance(dist_table, codes)
    
    # 2. ScaNN Anisotropic Loss
    x = torch.randn(B, D) + 1.0
    x_tilde = x + torch.randn(B, D) * 0.2
    scann_loss = scann_anisotropic_loss(x, x_tilde, omega=2.0)
    
    # 3. LSH Random Hyperplane Hashing
    q = torch.randn(B, D)
    d = torch.randn(B, D)
    hyperplanes = torch.randn(D, N_bits) * 0.5
    lsh_sim = lsh_hash_hamming_distance(q, d, hyperplanes)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        
        "query_subvectors": query_subvectors,
        "codebooks": codebooks,
        "dist_table": dist_table,
        
        "codes": codes,
        "adc_distances": adc_distances,
        
        "x": x,
        "x_tilde": x_tilde,
        "scann_loss": scann_loss,
        
        "q": q,
        "d": d,
        "hyperplanes": hyperplanes,
        "lsh_sim": lsh_sim,
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
