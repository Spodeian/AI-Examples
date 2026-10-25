"""XAI Local Surrogates LIME SHAP and Perturbation Masks — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks
Golden weights: weights/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def lime_weighted_regression(
    z_prime: torch.Tensor,
    y: torch.Tensor,
    pi: torch.Tensor,
    reg: float = 1e-4,
) -> torch.Tensor:
    """Closed-form weighted ridge regression for LIME feature attribution.
    
    w* = (Z'^T Pi Z' + reg * I)^(-1) Z'^T Pi y
    """
    sqrt_pi = torch.sqrt(pi).unsqueeze(-1)
    z_w = z_prime * sqrt_pi
    y_w = y * sqrt_pi.squeeze(-1)
    d = z_prime.shape[1]
    lhs = z_w.t() @ z_w + reg * torch.eye(d, dtype=z_prime.dtype, device=z_prime.device)
    rhs = z_w.t() @ y_w
    w = torch.linalg.solve(lhs, rhs)
    return w


def shapley_kernel_weights(
    coalition_sizes: torch.Tensor,
    m_features: int,
) -> torch.Tensor:
    """Compute exact KernelSHAP combinatorial weights for subsets of size |z'| in [1, M-1].
    
    pi(z') = (M - 1) / (binom(M, |z'|) * |z'| * (M - |z'|))
    """
    weights = torch.zeros_like(coalition_sizes, dtype=torch.float32)
    for i in range(coalition_sizes.shape[0]):
        s = int(coalition_sizes[i].item())
        if 0 < s < m_features:
            binom = math.comb(m_features, s)
            weights[i] = float(m_features - 1) / (float(binom) * float(s) * float(m_features - s))
        else:
            weights[i] = 1000.0  # Constraint boundary weight
    return weights


def rise_feature_importance(
    masks: torch.Tensor,
    model_outputs: torch.Tensor,
) -> torch.Tensor:
    """RISE (Randomized Input Sampling for Explanation) importance attribution.
    
    S_i = (1 / N) * sum_k f(x . m_k) * m_{k, i}
    """
    n = float(masks.shape[0])
    return (model_outputs.unsqueeze(0) @ masks).squeeze(0) / n


def faithfulness_degradation(
    y_orig: torch.Tensor,
    degraded_y: torch.Tensor,
) -> torch.Tensor:
    """Measure attribution faithfulness via monotonic confidence degradation curve."""
    return y_orig - degraded_y


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. LIME setup: N=8 perturbations, M=4 features
    lime_z = torch.tensor([
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    lime_y = torch.tensor([2.5, 1.8, -0.4, 1.2, 0.5, 0.1, 0.9, 3.2], dtype=torch.float32)
    lime_pi = torch.tensor([0.9, 0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 1.0], dtype=torch.float32)

    lime_w_ref = lime_weighted_regression(lime_z, lime_y, lime_pi, reg=1e-4)

    # 2. KernelSHAP weights setup (M=4 features, coalition sizes 1, 2, 3)
    shap_sizes = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 4.0], dtype=torch.float32)
    shap_weights_ref = shapley_kernel_weights(shap_sizes, m_features=4)

    # 3. RISE feature importance setup
    rise_masks = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    rise_outputs = torch.tensor([0.8, 0.4, 0.9, 0.6], dtype=torch.float32)
    rise_importance_ref = rise_feature_importance(rise_masks, rise_outputs)

    # 4. ROAR faithfulness degradation curve
    roar_orig = torch.tensor(0.95, dtype=torch.float32)
    roar_degraded = torch.tensor([0.82, 0.54, 0.31, 0.12], dtype=torch.float32)
    roar_degradation_ref = faithfulness_degradation(roar_orig, roar_degraded)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "lime_z": lime_z,
        "lime_y": lime_y,
        "lime_pi": lime_pi,
        "lime_w_ref": lime_w_ref,
        "shap_sizes": shap_sizes,
        "shap_weights_ref": shap_weights_ref,
        "rise_masks": rise_masks,
        "rise_outputs": rise_outputs,
        "rise_importance_ref": rise_importance_ref,
        "roar_orig": roar_orig,
        "roar_degraded": roar_degraded,
        "roar_degradation_ref": roar_degradation_ref,
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
