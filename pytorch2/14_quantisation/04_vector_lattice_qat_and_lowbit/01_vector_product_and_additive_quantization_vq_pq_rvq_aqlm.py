"""Vector Product and Additive Quantization VQ PQ RVQ AQLM — PyTorch 2.x reference implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm
Golden weights: weights/14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm.safetensors

``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def vector_quantize(vectors: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-codebook Vector Quantization (VQ): nearest Euclidean centroid.
    
    vectors: [N, D]
    codebook: [K, D]
    returns: (recon [N, D], indices [N])
    """
    # ||v - c||^2 = ||v||^2 - 2 v c^T + ||c||^2
    dist = (
        torch.sum(vectors ** 2, dim=1, keepdim=True)
        - 2.0 * (vectors @ codebook.T)
        + torch.sum(codebook ** 2, dim=1, keepdim=True).T
    )
    indices = torch.argmin(dist, dim=1)
    recon = codebook[indices]
    return recon, indices


def product_quantize(vectors: torch.Tensor, codebooks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Product Quantization (PQ): splits D-dim vector into M subvectors.
    
    vectors: [N, D]
    codebooks: [M, K, D_sub] where D_sub = D // M
    returns: (recon [N, D], indices [N, M])
    """
    n, d = vectors.shape
    m, k, d_sub = codebooks.shape
    assert d == m * d_sub, f"Dimension mismatch: {d} != {m} * {d_sub}"

    recon_subs = []
    indices_list = []
    for i in range(m):
        sub_vec = vectors[:, i * d_sub : (i + 1) * d_sub]
        sub_cb = codebooks[i]
        sub_recon, sub_idx = vector_quantize(sub_vec, sub_cb)
        recon_subs.append(sub_recon)
        indices_list.append(sub_idx.unsqueeze(1))

    recon = torch.cat(recon_subs, dim=1)
    indices = torch.cat(indices_list, dim=1)
    return recon, indices


def residual_vector_quantize(vectors: torch.Tensor, codebooks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual Vector Quantization (RVQ): sequential multi-stage codebook refinement.
    
    vectors: [N, D]
    codebooks: [M, K, D] (M sequential residual codebooks)
    returns: (recon [N, D], indices [N, M])
    """
    residual = vectors.clone()
    recon = torch.zeros_like(vectors)
    indices_list = []

    m = codebooks.shape[0]
    for i in range(m):
        cb = codebooks[i]
        stage_recon, stage_idx = vector_quantize(residual, cb)
        recon = recon + stage_recon
        residual = residual - stage_recon
        indices_list.append(stage_idx.unsqueeze(1))

    indices = torch.cat(indices_list, dim=1)
    return recon, indices


def aqlm_dequantize(indices: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Additive Multi-Codebook Quantization (AQLM) dequantization:
    
    recon = sum_{m=0}^{M-1} codebooks[m][indices[:, m]]
    indices: [N, M]
    codebooks: [M, K, D]
    returns: recon [N, D]
    """
    n, m = indices.shape
    _, _, d = codebooks.shape
    recon = torch.zeros((n, d), dtype=codebooks.dtype, device=codebooks.device)
    for i in range(m):
        idx = indices[:, i]
        recon = recon + codebooks[i][idx]
    return recon


def aqlm_forward(
    x: torch.Tensor,
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    d_in: int,
    d_out: int,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AQLM forward pass: reconstructs [d_in, d_out] weight matrix and computes Y = X @ W + bias."""
    w_recon_vecs = aqlm_dequantize(indices, codebooks)
    w_deq = w_recon_vecs.reshape(d_in, d_out)
    y = x @ w_deq
    if bias is not None:
        y = y + bias
    return y, w_deq


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, d_in, d_out = 16, 16, 8
    x_calib = torch.randn(b, d_in)
    w_orig = torch.randn(d_in, d_out) * 0.25
    b_orig = torch.randn(d_out) * 0.05

    # Group into N = 32 vectors of dimension D = 4
    w_vecs = w_orig.reshape(32, 4).clone()

    # 1. Single VQ codebook (K=8, D=4)
    vq_codebook = torch.randn(8, 4) * 0.2
    vq_recon, vq_indices = vector_quantize(w_vecs, vq_codebook)

    # 2. Product Quantization (M=2, K=8, D_sub=2)
    pq_codebooks = torch.randn(2, 8, 2) * 0.2
    pq_recon, pq_indices = product_quantize(w_vecs, pq_codebooks)

    # 3. Residual Vector Quantization (M=2, K=8, D=4)
    rvq_codebooks = torch.randn(2, 8, 4) * 0.15
    rvq_recon, rvq_indices = residual_vector_quantize(w_vecs, rvq_codebooks)

    # 4. AQLM multi-codebook forward pass using RVQ-derived indices & codebooks
    aqlm_indices = rvq_indices.clone()
    aqlm_codebooks = rvq_codebooks.clone()
    out_aqlm, w_aqlm_deq = aqlm_forward(x_calib, aqlm_indices, aqlm_codebooks, d_in, d_out, b_orig)

    # 5. Exact FP32 ground truth
    out_fp32 = x_calib @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "w_vecs": w_vecs,
        "vq_codebook": vq_codebook,
        "vq_recon": vq_recon,
        "vq_indices": vq_indices.float(),
        "pq_codebooks": pq_codebooks,
        "pq_recon": pq_recon,
        "pq_indices": pq_indices.float(),
        "rvq_codebooks": rvq_codebooks,
        "rvq_recon": rvq_recon,
        "rvq_indices": rvq_indices.float(),
        "aqlm_indices": aqlm_indices.float(),
        "aqlm_codebooks": aqlm_codebooks,
        "w_aqlm_deq": w_aqlm_deq,
        "out_aqlm": out_aqlm,
        "out_fp32": out_fp32,
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
