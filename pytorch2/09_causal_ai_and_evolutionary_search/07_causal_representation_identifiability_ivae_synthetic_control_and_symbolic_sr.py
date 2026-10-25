"""Causal Representation Identifiability iVAE Synthetic Control and Symbolic Sr — PyTorch 2.x reference implementation.

Topic: 09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr
Golden weights: weights/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.safetensors

``python pytorch2/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.py`` verifies against the golden checkpoint;
``python pytorch2/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ivae_kl_divergence(
    mu_z: torch.Tensor,
    logvar_z: torch.Tensor,
    mu_u: torch.Tensor,
    logvar_u: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL divergence between posterior q(z|x, u) and conditional prior p(z|u).
    
    KL(N(mu_z, sigma_z^2) || N(mu_u, sigma_u^2))
    = 0.5 * sum_j [ (sigma_z^2 + (mu_z - mu_u)^2) / sigma_u^2 - 1 - (logvar_z - logvar_u) ]
    """
    var_z = torch.exp(logvar_z)
    var_u = torch.exp(logvar_u)
    kl_per_dim = 0.5 * (
        (var_z + (mu_z - mu_u) ** 2) / var_u - 1.0 - (logvar_z - logvar_u)
    )
    return torch.sum(kl_per_dim, dim=-1)


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex {w >= 0, sum(w) = 1} (Duchi et al. 2008)."""
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, n_features + 1, dtype=v.dtype, device=v.device)
    cond = u - cssv / ind > 0.0
    # Find last True index
    rho = torch.max(torch.where(cond, ind, torch.zeros_like(ind))).long()
    theta = cssv[rho - 1] / float(rho)
    w = torch.clamp(v - theta, min=0.0)
    return w


def synthetic_control_effect(
    y1: torch.Tensor,
    y0: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute synthetic counterfactual control trajectory and treatment effect.
    
    y1: [T_post]
    y0: [T_post, J]
    w: [J]
    y1_synth = y0 @ w
    tau = y1 - y1_synth
    """
    y1_synth = y0 @ w
    tau = y1 - y1_synth
    return y1_synth, tau


def symbolic_pareto_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    complexity: torch.Tensor,
    lambda_reg: torch.Tensor,
) -> torch.Tensor:
    """Symbolic regression Pareto multi-objective loss: MSE(y, y_hat) + lambda * complexity."""
    mse = torch.mean((y_true - y_pred) ** 2)
    return mse + lambda_reg * complexity


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. iVAE conditional factorial Gaussian KL
    # B=4 batch samples, D=3 latent dimensions
    mu_z = torch.tensor([
        [0.5, -0.2, 1.1],
        [-0.8, 0.4, 0.2],
        [1.2, -1.0, 0.0],
        [0.1, 0.3, -0.5],
    ], dtype=torch.float32)
    logvar_z = torch.tensor([
        [-0.5, 0.1, -0.2],
        [0.2, -0.4, 0.0],
        [-0.1, 0.3, -0.8],
        [0.0, -0.2, 0.1],
    ], dtype=torch.float32)
    mu_u = torch.tensor([
        [0.2, 0.0, 0.8],
        [-0.5, 0.2, 0.5],
        [0.9, -0.7, -0.2],
        [0.3, 0.1, -0.2],
    ], dtype=torch.float32)
    logvar_u = torch.tensor([
        [0.0, 0.2, -0.1],
        [0.1, -0.2, 0.3],
        [0.0, 0.1, -0.5],
        [-0.2, 0.0, 0.2],
    ], dtype=torch.float32)

    ivae_kl_ref = ivae_kl_divergence(mu_z, logvar_z, mu_u, logvar_u)

    # 2. Synthetic Control
    # J=4 donor units, T_post=3 time steps
    sc_v = torch.tensor([0.6, -0.2, 0.8, 0.1], dtype=torch.float32)
    sc_w_simplex_ref = project_simplex(sc_v)

    sc_y1_post = torch.tensor([15.2, 18.5, 21.0], dtype=torch.float32)
    sc_y0_post = torch.tensor([
        [10.0, 12.0, 14.0, 9.0],
        [11.5, 13.0, 16.2, 10.5],
        [12.0, 15.0, 18.0, 11.0],
    ], dtype=torch.float32)

    sc_y1_synth_ref, sc_tau_ref = synthetic_control_effect(
        sc_y1_post, sc_y0_post, sc_w_simplex_ref
    )

    # 3. Symbolic Regression Pareto Loss
    sr_y_true = torch.tensor([1.2, 2.5, 3.8, 4.1, 5.5, 6.0, 7.2, 8.4], dtype=torch.float32)
    sr_y_pred = torch.tensor([1.1, 2.7, 3.6, 4.3, 5.3, 6.2, 7.0, 8.5], dtype=torch.float32)
    sr_complexity = torch.tensor(7.0, dtype=torch.float32)
    sr_lambda = torch.tensor(0.05, dtype=torch.float32)

    sr_pareto_loss_ref = symbolic_pareto_loss(
        sr_y_true, sr_y_pred, sr_complexity, sr_lambda
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "mu_z": mu_z,
        "logvar_z": logvar_z,
        "mu_u": mu_u,
        "logvar_u": logvar_u,
        "ivae_kl_ref": ivae_kl_ref,
        "sc_v": sc_v,
        "sc_w_simplex_ref": sc_w_simplex_ref,
        "sc_y1_post": sc_y1_post,
        "sc_y0_post": sc_y0_post,
        "sc_y1_synth_ref": sc_y1_synth_ref,
        "sc_tau_ref": sc_tau_ref,
        "sr_y_true": sr_y_true,
        "sr_y_pred": sr_y_pred,
        "sr_complexity": sr_complexity,
        "sr_lambda": sr_lambda,
        "sr_pareto_loss_ref": sr_pareto_loss_ref,
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
