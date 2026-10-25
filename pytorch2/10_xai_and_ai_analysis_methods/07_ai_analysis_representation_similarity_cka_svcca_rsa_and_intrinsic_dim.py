"""AI Analysis Representation Similarity CKA SVCCA RSA and Intrinsic Dim — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim
Golden weights: weights/10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def center_gram_matrix(gram: torch.Tensor) -> torch.Tensor:
    """Double centering of Gram matrix: H K H = K - row_mean - col_mean + total_mean."""
    mean_row = gram.mean(dim=0, keepdim=True)
    mean_col = gram.mean(dim=1, keepdim=True)
    mean_total = gram.mean()
    return gram - mean_row - mean_col + mean_total


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear Centered Kernel Alignment (CKA — Kornblith et al., 2019).
    
    CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    where K = X X^T, L = Y Y^T are centered Gram matrices.
    """
    k = x @ x.t()
    l_mat = y @ y.t()
    
    k_c = center_gram_matrix(k)
    l_c = center_gram_matrix(l_mat)
    
    hsic_xy = (k_c * l_c).sum()
    hsic_xx = (k_c * k_c).sum()
    hsic_yy = (l_c * l_c).sum()
    
    denom = torch.sqrt(hsic_xx * hsic_yy) + 1e-8
    return hsic_xy / denom


def rdm_euclidean(x: torch.Tensor) -> torch.Tensor:
    """Representational Dissimilarity Matrix (RDM — Kriegeskorte et al., 2008)."""
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
    return torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)


def rsa_correlation(rdm_a: torch.Tensor, rdm_b: torch.Tensor) -> torch.Tensor:
    """Representational Similarity Analysis Pearson correlation on upper-triangular RDM entries."""
    n = rdm_a.shape[0]
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    vec_a = rdm_a[idx_i, idx_j]
    vec_b = rdm_b[idx_i, idx_j]
    
    va_c = vec_a - vec_a.mean()
    vb_c = vec_b - vec_b.mean()
    
    num = (va_c * vb_c).sum()
    denom = torch.sqrt((va_c * va_c).sum() * (vb_c * vb_c).sum()) + 1e-8
    return num / denom


def twonn_intrinsic_dimension(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """TwoNN intrinsic manifold dimension estimator (Facco et al., 2017).
    
    mu_i = r2_i / r1_i
    d_hat = N / sum_i ln(mu_i)
    """
    mu = r2 / (r1 + 1e-8)
    log_mu = torch.log(mu)
    n = float(r1.shape[0])
    return n / (log_mu.sum() + 1e-8)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. CKA setup: N=5 samples, D_x=4, D_y=3
    cka_x = torch.tensor([
        [1.2, -0.5, 0.8, 0.4],
        [-0.3, 1.5, 0.1, -0.7],
        [0.6, 0.2, 1.4, 0.5],
        [-0.1, -0.8, 0.7, 1.2],
        [0.9, 0.4, -0.3, -0.2],
    ], dtype=torch.float32)
    cka_y = torch.tensor([
        [0.9, -0.2, 0.7],
        [-0.1, 1.2, -0.5],
        [0.4, 0.3, 1.1],
        [-0.3, -0.6, 0.9],
        [0.7, 0.5, -0.1],
    ], dtype=torch.float32)
    cka_score_ref = linear_cka(cka_x, cka_y)

    # 2. RSA setup: N=4 samples, compute RDM and RSA correlation
    rsa_x = torch.tensor([
        [1.0, 0.5, -0.2],
        [-0.4, 1.2, 0.3],
        [0.7, -0.6, 1.4],
        [0.2, 0.9, -0.8],
    ], dtype=torch.float32)
    rsa_y = torch.tensor([
        [0.8, 0.4, -0.1],
        [-0.3, 1.0, 0.2],
        [0.6, -0.5, 1.2],
        [0.1, 0.8, -0.7],
    ], dtype=torch.float32)
    rdm_a = rdm_euclidean(rsa_x)
    rdm_b = rdm_euclidean(rsa_y)
    rsa_corr_ref = rsa_correlation(rdm_a, rdm_b)

    # 3. TwoNN setup: N=6 samples
    twonn_r1 = torch.tensor([0.45, 0.32, 0.58, 0.29, 0.41, 0.52], dtype=torch.float32)
    twonn_r2 = torch.tensor([0.82, 0.61, 0.99, 0.54, 0.77, 0.89], dtype=torch.float32)
    twonn_id_ref = twonn_intrinsic_dimension(twonn_r1, twonn_r2)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "cka_x": cka_x,
        "cka_y": cka_y,
        "cka_score_ref": cka_score_ref,
        "rsa_x": rsa_x,
        "rsa_y": rsa_y,
        "rdm_a": rdm_a,
        "rdm_b": rdm_b,
        "rsa_corr_ref": rsa_corr_ref,
        "twonn_r1": twonn_r1,
        "twonn_r2": twonn_r2,
        "twonn_id_ref": twonn_id_ref,
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
