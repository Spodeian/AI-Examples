"""AI Analysis Neural Collapse Superposition and WeightWatcher RMT — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt
Golden weights: weights/10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def neural_collapse_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Neural Collapse NC1 (within-class variability) and NC2 (Simplex ETF) metrics (Papyan et al., 2020).
    
    NC1 = (1 / N) * sum_i ||h_i - mu_{y_i}||^2
    NC2_err = (1 / (C(C-1))) * sum_{c != c'} (cos(mu_c, mu_c') - (-1/(C-1)))^2
    """
    n, d = features.shape
    c = num_classes
    
    # Compute class means
    class_means = torch.zeros(c, d, dtype=features.dtype, device=features.device)
    class_counts = torch.zeros(c, dtype=features.dtype, device=features.device)
    for i in range(n):
        lbl = int(labels[i].item())
        class_means[lbl] += features[i]
        class_counts[lbl] += 1.0
    for k in range(c):
        if class_counts[k] > 0:
            class_means[k] /= class_counts[k]
            
    # NC1: within-class variability
    nc1_sum = 0.0
    for i in range(n):
        lbl = int(labels[i].item())
        diff = features[i] - class_means[lbl]
        nc1_sum = nc1_sum + (diff * diff).sum()
    nc1_var = nc1_sum / float(n)
    
    # NC2: Simplex ETF cosine alignment
    global_mean = class_means.mean(dim=0, keepdim=True)
    centered_means = class_means - global_mean
    norms = torch.sqrt((centered_means * centered_means).sum(dim=-1, keepdim=True) + 1e-8)
    norm_means = centered_means / norms
    
    cos_matrix = norm_means @ norm_means.t()  # [C, C]
    ideal_cos = -1.0 / float(c - 1)
    
    nc2_err_sum = 0.0
    for i in range(c):
        for j in range(c):
            if i != j:
                diff = cos_matrix[i, j] - ideal_cos
                nc2_err_sum = nc2_err_sum + diff * diff
    nc2_etf_err = nc2_err_sum / float(c * (c - 1))
    
    return nc1_var, nc2_etf_err


def superposition_interference(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Toy Model of Superposition (Elhage et al., 2022) feature interference energy.
    
    G = W^T W in R^{M x M}
    Interference = sum_{i != j} G_{ij}^2
    """
    # w: [D, M]
    gram = w.t() @ w  # [M, M]
    m = gram.shape[0]
    diag = torch.diag(gram)
    off_diag = gram - torch.diag_embed(diag)
    interf_energy = (off_diag * off_diag).sum()
    return gram, interf_energy


def weightwatcher_alpha_hill(
    eigenvalues: torch.Tensor,
    lambda_min: float = 0.5,
) -> torch.Tensor:
    """WeightWatcher / Heavy-Tailed RMT power-law tail exponent via Hill estimator (Martin & Mahoney, 2021).
    
    alpha = 1 + K / (sum_{k=1}^K ln(lambda_k / lambda_min))
    """
    tail = eigenvalues[eigenvalues >= lambda_min]
    k_tail = float(tail.shape[0])
    log_ratios = torch.log(tail / lambda_min)
    alpha = 1.0 + k_tail / (log_ratios.sum() + 1e-8)
    return alpha


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. Neural Collapse setup: N=6 instances, D=3 dimensions, C=3 classes
    nc_feats = torch.tensor([
        [1.0, 0.1, -0.1],
        [0.9, -0.1, 0.0],
        [-0.5, 0.85, 0.1],
        [-0.45, 0.90, -0.05],
        [-0.5, -0.85, 0.0],
        [-0.48, -0.88, 0.05],
    ], dtype=torch.float32)
    nc_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.float32)
    nc1_ref, nc2_ref = neural_collapse_metrics(nc_feats, nc_labels, num_classes=3)

    # 2. Superposition setup: D=2 hidden dimensions, M=4 features
    sp_w = torch.tensor([
        [1.0, 0.0, -0.7071, 0.7071],
        [0.0, 1.0, 0.7071, 0.7071],
    ], dtype=torch.float32)
    sp_gram_ref, sp_interf_ref = superposition_interference(sp_w)

    # 3. WeightWatcher RMT setup: eigenvalues of empirical correlation matrix
    ww_evals = torch.tensor([4.5, 2.8, 1.6, 1.1, 0.8, 0.55, 0.3, 0.15], dtype=torch.float32)
    ww_alpha_ref = weightwatcher_alpha_hill(ww_evals, lambda_min=0.5)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "nc_feats": nc_feats,
        "nc_labels": nc_labels,
        "nc1_ref": nc1_ref,
        "nc2_ref": nc2_ref,
        "sp_w": sp_w,
        "sp_gram_ref": sp_gram_ref,
        "sp_interf_ref": sp_interf_ref,
        "ww_evals": ww_evals,
        "ww_alpha_ref": ww_alpha_ref,
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
