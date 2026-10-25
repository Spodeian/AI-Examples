"""XAI Concepts TCAV CBM Counterfactuals DiCE and Global ALE — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale
Golden weights: weights/10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def tcav_directional_derivatives(
    activation_grads: torch.Tensor,
    cav: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Testing with Concept Activation Vectors (Kim et al., 2018).
    
    S_{C, k}(x) = grad_a f_k(x) . v_C
    TCAV_score = (1 / N) * sum_i I(S_{C, k}(x_i) > 0)
    """
    cav_norm = cav / torch.sqrt((cav * cav).sum() + 1e-8)
    dir_derivs = activation_grads @ cav_norm  # [N]
    positive_mask = torch.where(dir_derivs > 0.0, 1.0, 0.0)
    tcav_score = positive_mask.mean()
    return dir_derivs, tcav_score


def cbm_test_time_intervention(
    pred_concepts: torch.Tensor,
    true_concepts: torch.Tensor,
    intervention_mask: torch.Tensor,
    task_weights: torch.Tensor,
    task_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concept Bottleneck Model (Koh et al., 2020) test-time expert concept intervention.
    
    c_int = (1 - mask) * c_pred + mask * c_true
    y_pred = c_int @ W_task + b_task
    """
    intervened_concepts = (1.0 - intervention_mask) * pred_concepts + intervention_mask * true_concepts
    task_preds = intervened_concepts @ task_weights + task_bias
    return intervened_concepts, task_preds


def dice_dpp_diversity(counterfactuals: torch.Tensor) -> torch.Tensor:
    """DiCE Determinantal Point Process diversity metric (Mothilal et al., 2020).
    
    K_{i, j} = 1 / (1 + ||c_i - c_j||_2)
    Diversity = det(K)
    """
    k_cf = counterfactuals.shape[0]
    # Pairwise Euclidean distances
    diff = counterfactuals.unsqueeze(1) - counterfactuals.unsqueeze(0)  # [K, K, D]
    dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
    kernel_mat = 1.0 / (1.0 + dist)
    return torch.linalg.det(kernel_mat)


def ale_accumulated_local_effects(interval_diffs: torch.Tensor) -> torch.Tensor:
    """Accumulated Local Effects (Apley & Zhu, 2020) on 1D feature grid.
    
    ALE(x_k) = cumsum(Delta_k) - mean(cumsum(Delta_k))
    """
    cumsum_ale = torch.cumsum(interval_diffs, dim=0)
    centered_ale = cumsum_ale - cumsum_ale.mean()
    return centered_ale


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. TCAV setup: N=6 samples, D=4 activations, CAV concept vector
    tcav_grads = torch.tensor([
        [0.4, -0.2, 0.8, 0.1],
        [-0.5, 0.3, -0.1, 0.2],
        [0.7, 0.1, 0.5, -0.3],
        [-0.1, -0.4, 0.2, 0.6],
        [0.3, 0.5, -0.2, 0.1],
        [-0.6, -0.1, 0.3, -0.4],
    ], dtype=torch.float32)
    tcav_cav = torch.tensor([0.5, 0.2, 0.8, -0.1], dtype=torch.float32)
    tcav_derivs_ref, tcav_score_ref = tcav_directional_derivatives(tcav_grads, tcav_cav)

    # 2. CBM setup: N=4 instances, K=3 concepts, C=2 task classes
    cbm_pred_c = torch.tensor([
        [0.2, 0.8, 0.1],
        [0.7, 0.3, 0.9],
        [0.4, 0.5, 0.2],
        [0.9, 0.1, 0.6],
    ], dtype=torch.float32)
    cbm_true_c = torch.tensor([
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ], dtype=torch.float32)
    cbm_mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)  # Intervene on concept 0 and 2
    cbm_w = torch.tensor([
        [0.5, -0.2],
        [-0.3, 0.7],
        [0.4, 0.1],
    ], dtype=torch.float32)
    cbm_b = torch.tensor([0.1, -0.1], dtype=torch.float32)
    cbm_c_int_ref, cbm_y_ref = cbm_test_time_intervention(cbm_pred_c, cbm_true_c, cbm_mask, cbm_w, cbm_b)

    # 3. DiCE DPP Diversity setup: K=3 counterfactual candidates, D=4 features
    dice_cfs = torch.tensor([
        [1.0, 0.5, -0.2, 2.0],
        [0.2, 1.8, 0.4, -0.5],
        [-1.2, 0.1, 1.5, 0.8],
    ], dtype=torch.float32)
    dice_div_ref = dice_dpp_diversity(dice_cfs)

    # 4. ALE setup: 5 intervals
    ale_diffs = torch.tensor([0.15, 0.32, -0.08, 0.45, -0.12], dtype=torch.float32)
    ale_ref = ale_accumulated_local_effects(ale_diffs)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "tcav_grads": tcav_grads,
        "tcav_cav": tcav_cav,
        "tcav_derivs_ref": tcav_derivs_ref,
        "tcav_score_ref": tcav_score_ref,
        "cbm_pred_c": cbm_pred_c,
        "cbm_true_c": cbm_true_c,
        "cbm_mask": cbm_mask,
        "cbm_w": cbm_w,
        "cbm_b": cbm_b,
        "cbm_c_int_ref": cbm_c_int_ref,
        "cbm_y_ref": cbm_y_ref,
        "dice_cfs": dice_cfs,
        "dice_div_ref": dice_div_ref,
        "ale_diffs": ale_diffs,
        "ale_ref": ale_ref,
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
