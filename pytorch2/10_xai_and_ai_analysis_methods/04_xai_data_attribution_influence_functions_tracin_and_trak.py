"""XAI Data Attribution Influence Functions TracIn and TRAK — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak
Golden weights: weights/10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def influence_score(
    hessian: torch.Tensor,
    grad_test: torch.Tensor,
    grad_train: torch.Tensor,
    damping: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Influence function computation via damped inverse Hessian-vector product (Koh & Liang, 2017).
    
    ihvp = (H + lambda * I)^(-1) * grad_test
    influence = - grad_train . ihvp
    """
    d = hessian.shape[0]
    h_damped = hessian + damping * torch.eye(d, dtype=hessian.dtype, device=hessian.device)
    ihvp = torch.linalg.solve(h_damped, grad_test)
    inf = -(grad_train * ihvp).sum()
    return ihvp, inf


def tracin_trajectory_attribution(
    train_grads: torch.Tensor,
    test_grads: torch.Tensor,
    learning_rates: torch.Tensor,
) -> torch.Tensor:
    """TracIn: Tracing gradient descent checkpoint dot-products (Pruthi et al., 2020).
    
    score = sum_{t=0}^{T-1} lr_t * (grad_train_t . grad_test_t)
    """
    # train_grads: [T, D], test_grads: [T, D], learning_rates: [T]
    dot_per_checkpoint = (train_grads * test_grads).sum(dim=-1)  # [T]
    return (learning_rates * dot_per_checkpoint).sum()


def trak_projected_attribution(
    train_grads: torch.Tensor,
    test_grad: torch.Tensor,
    proj_matrix: torch.Tensor,
) -> torch.Tensor:
    """TRAK: Tracing with random JL projections at scale (Park et al., 2023).
    
    phi_train = train_grads @ P   # [N, K]
    phi_test = test_grad @ P      # [K]
    score_i = (phi_train_i . phi_test) / (||phi_train_i|| * ||phi_test|| + eps)
    """
    phi_train = train_grads @ proj_matrix  # [N, K]
    phi_test = test_grad @ proj_matrix    # [K]
    
    norm_train = torch.sqrt((phi_train * phi_train).sum(dim=-1, keepdim=True) + 1e-8)  # [N, 1]
    norm_test = torch.sqrt((phi_test * phi_test).sum() + 1e-8)                          # scalar
    
    scores = (phi_train @ phi_test.unsqueeze(-1)) / (norm_train * norm_test)
    return scores.squeeze(-1)  # [N]


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. Influence Functions setup: D=4 parameters
    inf_hessian = torch.tensor([
        [2.0, 0.4, 0.1, -0.2],
        [0.4, 1.8, 0.3, 0.0],
        [0.1, 0.3, 2.5, 0.5],
        [-0.2, 0.0, 0.5, 1.5],
    ], dtype=torch.float32)
    inf_grad_test = torch.tensor([0.8, -0.5, 1.2, 0.3], dtype=torch.float32)
    inf_grad_train = torch.tensor([0.6, -0.2, 0.9, -0.1], dtype=torch.float32)
    inf_ihvp_ref, inf_score_ref = influence_score(inf_hessian, inf_grad_test, inf_grad_train, damping=1e-3)

    # 2. TracIn setup: T=4 checkpoints, D=4 parameters
    tracin_train_g = torch.tensor([
        [0.5, 0.8, -0.2, 0.4],
        [0.4, 0.6, -0.1, 0.3],
        [0.2, 0.3, 0.0, 0.2],
        [0.1, 0.1, 0.1, 0.1],
    ], dtype=torch.float32)
    tracin_test_g = torch.tensor([
        [0.6, 0.7, -0.3, 0.5],
        [0.3, 0.5, -0.2, 0.2],
        [0.1, 0.2, -0.1, 0.1],
        [0.05, 0.1, 0.0, 0.05],
    ], dtype=torch.float32)
    tracin_lrs = torch.tensor([0.1, 0.05, 0.02, 0.01], dtype=torch.float32)
    tracin_score_ref = tracin_trajectory_attribution(tracin_train_g, tracin_test_g, tracin_lrs)

    # 3. TRAK setup: N=5 training samples, D=4 parameters, K=3 projected dim
    trak_train_g = torch.tensor([
        [1.2, 0.5, -0.8, 0.3],
        [-0.4, 1.5, 0.2, -0.6],
        [0.8, -0.2, 1.4, 0.5],
        [0.1, 0.9, -0.5, 1.1],
        [-0.9, -0.6, 0.7, -0.2],
    ], dtype=torch.float32)
    trak_test_g = torch.tensor([0.9, 0.4, -0.6, 0.2], dtype=torch.float32)
    trak_proj = torch.tensor([
        [0.5, -0.2, 0.8],
        [-0.3, 0.7, 0.1],
        [0.6, 0.4, -0.5],
        [0.2, -0.8, 0.3],
    ], dtype=torch.float32)
    trak_scores_ref = trak_projected_attribution(trak_train_g, trak_test_g, trak_proj)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "inf_hessian": inf_hessian,
        "inf_grad_test": inf_grad_test,
        "inf_grad_train": inf_grad_train,
        "inf_ihvp_ref": inf_ihvp_ref,
        "inf_score_ref": inf_score_ref,
        "tracin_train_g": tracin_train_g,
        "tracin_test_g": tracin_test_g,
        "tracin_lrs": tracin_lrs,
        "tracin_score_ref": tracin_score_ref,
        "trak_train_g": trak_train_g,
        "trak_test_g": trak_test_g,
        "trak_proj": trak_proj,
        "trak_scores_ref": trak_scores_ref,
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
