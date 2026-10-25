"""AI Brain Surgery Knowledge Editing ROME MEMIT and Stitching — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching
Golden weights: weights/10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def rome_rank_one_edit(
    weight: torch.Tensor,
    key_star: torch.Tensor,
    val_star: torch.Tensor,
    cov_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-One Model Editing (ROME — Meng et al., 2022).
    
    Delta_W = (v_* - W k_*) @ (C^(-1) k_*)^T / (k_*^T C^(-1) k_*)
    W_new = W + Delta_W
    """
    v_curr = weight @ key_star  # [D_out]
    residual = val_star - v_curr  # [D_out]
    u = cov_inv @ key_star  # [D_in]
    denom = (key_star * u).sum() + 1e-8
    delta_w = (residual.unsqueeze(1) @ u.unsqueeze(0)) / denom  # [D_out, D_in]
    w_new = weight + delta_w
    return delta_w, w_new


def memit_mass_edit(
    weight: torch.Tensor,
    keys: torch.Tensor,
    vals: torch.Tensor,
    cov_inv: torch.Tensor,
    reg: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mass-Editing Memory in a Transformer (MEMIT — Meng et al., 2023).
    
    Delta = V_* - W K_*
    Delta_W = Delta @ (K_*^T C^(-1) K_* + reg * I)^(-1) @ (C^(-1) K_*)^T
    """
    # keys: [D_in, M], vals: [D_out, M]
    residuals = vals - weight @ keys  # [D_out, M]
    u_mat = cov_inv @ keys  # [D_in, M]
    m_dim = keys.shape[1]
    gram = keys.t() @ u_mat + reg * torch.eye(m_dim, dtype=keys.dtype, device=keys.device)  # [M, M]
    # Solve G @ A = residuals^T -> A^T @ G^T = residuals
    coeff = torch.linalg.solve(gram.t(), residuals.t()).t()  # [D_out, M]
    delta_w = coeff @ u_mat.t()  # [D_out, D_in]
    w_new = weight + delta_w
    return delta_w, w_new


def model_stitching_layer(
    acts_a: torch.Tensor,
    acts_b: torch.Tensor,
    reg: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model Stitching (Lenc & Vedaldi, 2015; Csiszarik et al., 2021).
    
    S = (A^T A + reg * I)^(-1) A^T B
    B_hat = A @ S
    """
    d_a = acts_a.shape[1]
    lhs = acts_a.t() @ acts_a + reg * torch.eye(d_a, dtype=acts_a.dtype, device=acts_a.device)
    rhs = acts_a.t() @ acts_b
    s_matrix = torch.linalg.solve(lhs, rhs)  # [D_a, D_b]
    b_hat = acts_a @ s_matrix
    return s_matrix, b_hat


def neuron_knockout(
    weight: torch.Tensor,
    knockout_mask: torch.Tensor,
) -> torch.Tensor:
    """Surgical neuron pruning / knockout of specific input channels.
    
    W_pruned = W * (1 - mask)
    """
    keep_mask = 1.0 - knockout_mask.unsqueeze(0)
    return weight * keep_mask


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. ROME setup: D_out=4, D_in=4
    rome_w = torch.tensor([
        [0.5, -0.2, 0.4, 0.1],
        [-0.3, 0.6, 0.0, -0.4],
        [0.2, 0.1, 0.7, 0.3],
        [-0.1, -0.5, 0.2, 0.8],
    ], dtype=torch.float32)
    rome_k_star = torch.tensor([1.2, -0.8, 0.5, 2.0], dtype=torch.float32)
    rome_v_star = torch.tensor([2.5, 1.0, -0.4, 3.2], dtype=torch.float32)
    rome_cov_inv = torch.tensor([
        [1.5, 0.2, -0.1, 0.0],
        [0.2, 1.8, 0.3, -0.2],
        [-0.1, 0.3, 2.0, 0.1],
        [0.0, -0.2, 0.1, 1.2],
    ], dtype=torch.float32)
    rome_delta_ref, rome_wnew_ref = rome_rank_one_edit(rome_w, rome_k_star, rome_v_star, rome_cov_inv)

    # 2. MEMIT setup: M=3 facts concurrently, D_out=4, D_in=4
    memit_w = rome_w.clone()
    memit_keys = torch.tensor([
        [1.2, 0.4, -0.6],
        [-0.8, 1.5, 0.2],
        [0.5, -0.3, 1.8],
        [2.0, 0.1, -0.5],
    ], dtype=torch.float32)
    memit_vals = torch.tensor([
        [2.5, -0.5, 1.2],
        [1.0, 2.2, -0.8],
        [-0.4, 1.4, 0.6],
        [3.2, 0.0, 1.9],
    ], dtype=torch.float32)
    memit_delta_ref, memit_wnew_ref = memit_mass_edit(memit_w, memit_keys, memit_vals, rome_cov_inv, reg=1e-4)

    # 3. Model Stitching setup: N=5 samples, D_a=4, D_b=3
    stitch_a = torch.tensor([
        [1.0, 0.5, -0.2, 0.8],
        [-0.4, 1.2, 0.3, -0.1],
        [0.7, -0.6, 1.4, 0.5],
        [0.2, 0.9, -0.8, 1.1],
        [-0.5, 0.0, 0.6, -0.4],
    ], dtype=torch.float32)
    stitch_b = torch.tensor([
        [0.8, -0.3, 1.1],
        [-0.2, 0.9, 0.4],
        [1.2, -0.5, 0.6],
        [0.4, 0.7, 0.2],
        [-0.6, 0.1, -0.3],
    ], dtype=torch.float32)
    stitch_s_ref, stitch_bhat_ref = model_stitching_layer(stitch_a, stitch_b, reg=1e-4)

    # 4. Neuron Knockout setup
    ko_mask = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)  # disable channel 1 and 3
    ko_pruned_ref = neuron_knockout(rome_w, ko_mask)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "rome_w": rome_w,
        "rome_k_star": rome_k_star,
        "rome_v_star": rome_v_star,
        "rome_cov_inv": rome_cov_inv,
        "rome_delta_ref": rome_delta_ref,
        "rome_wnew_ref": rome_wnew_ref,
        "memit_keys": memit_keys,
        "memit_vals": memit_vals,
        "memit_delta_ref": memit_delta_ref,
        "memit_wnew_ref": memit_wnew_ref,
        "stitch_a": stitch_a,
        "stitch_b": stitch_b,
        "stitch_s_ref": stitch_s_ref,
        "stitch_bhat_ref": stitch_bhat_ref,
        "ko_mask": ko_mask,
        "ko_pruned_ref": ko_pruned_ref,
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
