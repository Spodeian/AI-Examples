"""XAI Gradient Propagation IG LRP DeepLIFT Grad-CAM and Attention — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention
Golden weights: weights/10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def integrated_gradients(
    x: torch.Tensor,
    x_baseline: torch.Tensor,
    interpolated_grads: torch.Tensor,
) -> torch.Tensor:
    """Riemann sum approximation of Integrated Gradients (Sundararajan et al., 2017).
    
    IG_i(x) = (x_i - x'_i) * (1 / M) * sum_{m=0}^{M-1} grad_{m, i}
    """
    diff = x - x_baseline
    avg_grad = interpolated_grads.mean(dim=0)
    return diff * avg_grad


def lrp_epsilon_rule(
    activations: torch.Tensor,
    weights: torch.Tensor,
    relevance_next: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Layer-wise Relevance Propagation epsilon-rule for linear layer.
    
    z_{ij} = a_i * w_{ij}
    z_j = sum_i z_{ij} + eps * sign(sum_i z_{ij})
    R_i = sum_j (z_{ij} / z_j) * R_j^{next}
    """
    z_ij = activations.unsqueeze(-1) * weights  # [d_in, d_out]
    z_j = z_ij.sum(dim=0)  # [d_out]
    sign_z_j = torch.where(z_j >= 0.0, torch.ones_like(z_j), -torch.ones_like(z_j))
    z_j_stab = z_j + epsilon * sign_z_j
    s = relevance_next / z_j_stab  # [d_out]
    return (z_ij @ s.unsqueeze(-1)).squeeze(-1)  # [d_in]


def grad_cam_2d(
    activations: torch.Tensor,
    grads: torch.Tensor,
) -> torch.Tensor:
    """Grad-CAM attribution for 2D spatial feature activations.
    
    alpha_c = (1 / HW) * sum_{h, w} grad_{c, h, w}
    L_GradCAM = ReLU(sum_c alpha_c * A_c)
    """
    # Global average pooling over spatial dimensions [H, W]
    alpha = grads.mean(dim=(-2, -1))  # [C]
    # Linear combination of forward activation maps
    cam = (alpha.view(-1, 1, 1) * activations).sum(dim=0)  # [H, W]
    return torch.relu(cam)


def cosine_similarity(
    s_orig: torch.Tensor,
    s_rand: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Adebayo parameter randomization test alignment metric."""
    dot = (s_orig * s_rand).sum()
    norm_orig = torch.sqrt((s_orig * s_orig).sum())
    norm_rand = torch.sqrt((s_rand * s_rand).sum())
    return dot / (norm_orig * norm_rand + eps)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. Integrated Gradients setup: M=5 path steps, D=4 features
    ig_x = torch.tensor([1.2, -0.8, 2.5, 0.4], dtype=torch.float32)
    ig_baseline = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    ig_grads = torch.tensor([
        [0.10, 0.25, 0.40, 0.05],
        [0.12, 0.28, 0.45, 0.08],
        [0.15, 0.30, 0.52, 0.10],
        [0.18, 0.35, 0.58, 0.12],
        [0.20, 0.40, 0.65, 0.15],
    ], dtype=torch.float32)
    ig_ref = integrated_gradients(ig_x, ig_baseline, ig_grads)

    # 2. LRP setup: 4 input activations, 3 output relevances
    lrp_acts = torch.tensor([0.5, 1.2, 0.8, 2.0], dtype=torch.float32)
    lrp_weights = torch.tensor([
        [0.3, -0.1, 0.5],
        [-0.2, 0.4, 0.1],
        [0.6, 0.2, -0.3],
        [0.1, 0.7, 0.4],
    ], dtype=torch.float32)
    lrp_rel_next = torch.tensor([1.5, 2.2, 0.8], dtype=torch.float32)
    lrp_ref = lrp_epsilon_rule(lrp_acts, lrp_weights, lrp_rel_next, epsilon=1e-4)

    # 3. Grad-CAM setup: C=4, H=3, W=3
    cam_acts = torch.tensor([
        [[0.1, 0.5, 0.2], [0.8, 1.2, 0.4], [0.0, 0.3, 0.7]],
        [[0.4, 0.1, 0.9], [0.3, 0.7, 0.5], [0.6, 0.2, 0.1]],
        [[0.7, 0.8, 0.2], [0.5, 0.0, 0.4], [0.3, 0.9, 0.6]],
        [[0.2, 0.4, 0.6], [0.9, 0.5, 0.1], [0.8, 0.3, 0.2]],
    ], dtype=torch.float32)
    cam_grads = torch.tensor([
        [[0.2, 0.4, 0.1], [0.3, 0.5, 0.2], [0.1, 0.3, 0.4]],
        [[-0.1, 0.0, 0.2], [0.1, -0.2, 0.3], [0.0, 0.1, -0.1]],
        [[0.5, 0.6, 0.4], [0.7, 0.3, 0.5], [0.4, 0.8, 0.6]],
        [[-0.3, -0.1, 0.0], [-0.2, 0.1, -0.4], [0.0, -0.2, -0.1]],
    ], dtype=torch.float32)
    cam_ref = grad_cam_2d(cam_acts, cam_grads)

    # 4. Adebayo randomization check
    adebayo_orig = torch.tensor([0.85, 0.72, 0.91, 0.45, 0.60, 0.88, 0.76, 0.95], dtype=torch.float32)
    adebayo_rand = torch.tensor([0.12, -0.05, 0.18, 0.22, -0.15, 0.08, 0.03, -0.10], dtype=torch.float32)
    adebayo_sim_ref = cosine_similarity(adebayo_orig, adebayo_rand)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "ig_x": ig_x,
        "ig_baseline": ig_baseline,
        "ig_grads": ig_grads,
        "ig_ref": ig_ref,
        "lrp_acts": lrp_acts,
        "lrp_weights": lrp_weights,
        "lrp_rel_next": lrp_rel_next,
        "lrp_ref": lrp_ref,
        "cam_acts": cam_acts,
        "cam_grads": cam_grads,
        "cam_ref": cam_ref,
        "adebayo_orig": adebayo_orig,
        "adebayo_rand": adebayo_rand,
        "adebayo_sim_ref": adebayo_sim_ref,
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
