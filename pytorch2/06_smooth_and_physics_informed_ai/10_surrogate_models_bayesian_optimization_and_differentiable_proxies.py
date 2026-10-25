"""Surrogate Models Bayesian Optimization & Differentiable Proxies — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies
Golden weights: weights/06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def rbf_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscale: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    r"""Squared Exponential (RBF) Kernel:
    K(x1, x2) = variance * exp( - 0.5 * ||(x1 - x2) / lengthscale||^2 ).
    x1: [N1, D], x2: [N2, D] -> [N1, N2]
    """
    diff = (x1.unsqueeze(1) - x2.unsqueeze(0)) / lengthscale # [N1, N2, D]
    dist_sq = torch.sum(diff**2, dim=-1) # [N1, N2]
    return variance * torch.exp(-0.5 * dist_sq)


def gp_posterior(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    lengthscale: torch.Tensor,
    variance: torch.Tensor,
    noise_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Gaussian Process Posterior Mean and Variance:
    K_yy = K(X, X) + sigma_n^2 I
    K_sy = K(X_test, X)
    K_ss = K(X_test, X_test)
    mu_s = K_sy @ K_yy^{-1} y
    var_s = diag(K_ss - K_sy @ K_yy^{-1} K_sy^T)
    """
    N = x_train.shape[0]
    K_yy = rbf_kernel(x_train, x_train, lengthscale, variance) + noise_var * torch.eye(N, dtype=x_train.dtype, device=x_train.device)
    K_sy = rbf_kernel(x_test, x_train, lengthscale, variance)
    K_ss = rbf_kernel(x_test, x_test, lengthscale, variance)
    
    # Solve K_yy alpha = y_train
    alpha = torch.linalg.solve(K_yy, y_train.unsqueeze(-1)).squeeze(-1) # [N]
    mu_s = K_sy @ alpha # [N_test]
    
    # Solve K_yy V = K_sy^T -> V = K_yy^{-1} K_sy^T
    V = torch.linalg.solve(K_yy, K_sy.t()) # [N, N_test]
    cov_s = K_ss - K_sy @ V # [N_test, N_test]
    var_s = torch.clamp(torch.diag(cov_s), min=1e-6)
    return mu_s, var_s


def expected_improvement(
    mu: torch.Tensor,
    var: torch.Tensor,
    best_y: torch.Tensor,
    xi: float = 0.01,
) -> torch.Tensor:
    r"""Expected Improvement (EI) for maximization:
    EI(x) = (mu - best_y - xi) * Phi(Z) + sigma * phi(Z),
    where Z = (mu - best_y - xi) / sigma.
    Using standard Normal CDF approximation:
    Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
    """
    sigma = torch.sqrt(var)
    improvement = mu - best_y - xi
    Z = improvement / sigma
    
    # Standard normal PDF phi(Z) and CDF Phi(Z)
    phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * Z**2)
    # Abramowitz-Stegun formula 7.1.26 erf evaluation for cross-track determinism
    # Phi(Z) = 0.5 * (1 + torch.erf(Z / math.sqrt(2)))
    # In PyTorch:
    phi_cdf = 0.5 * (1.0 + torch.erf(Z / math.sqrt(2.0)))
    
    ei = improvement * phi_cdf + sigma * phi_z
    return torch.clamp(ei, min=0.0)


def ucb_acquisition(
    mu: torch.Tensor,
    var: torch.Tensor,
    beta: float = 2.0,
) -> torch.Tensor:
    r"""Upper Confidence Bound (UCB):
    UCB(x) = mu + beta * sigma.
    """
    sigma = torch.sqrt(var)
    return mu + beta * sigma


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    N_train, N_test, D = 6, 8, 2
    
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # GP Data
    x_train = torch.randn(N_train, D) * 1.2
    # Objective function with non-convex landscape: f(x) = sin(x_0) + cos(x_1) - 0.1 * ||x||^2
    y_train = torch.sin(x_train[:, 0]) + torch.cos(x_train[:, 1]) - 0.1 * torch.sum(x_train**2, dim=-1)
    x_test = torch.randn(N_test, D) * 1.5
    
    # Hyperparameters
    lengthscale = torch.tensor([1.2, 1.2])
    variance = torch.tensor(1.5)
    noise_var = torch.tensor(1e-3)
    
    # 1. GP Posterior Mean & Variance
    mu_s, var_s = gp_posterior(x_train, y_train, x_test, lengthscale, variance, noise_var)
    
    # 2. Acquisition Functions
    best_y = torch.max(y_train)
    ei_acq = expected_improvement(mu_s, var_s, best_y, xi=0.01)
    ucb_acq = ucb_acquisition(mu_s, var_s, beta=2.0)
    
    # 3. Kernel derivative / sensitivity with respect to query point x_test
    # dK/dx_test = - 1/lengthscale^2 * (x_test - x_train) * K
    k_sy = rbf_kernel(x_test, x_train, lengthscale, variance)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "lengthscale": lengthscale,
        "variance": variance,
        "noise_var": noise_var,
        "k_sy": k_sy,
        "mu_s": mu_s,
        "var_s": var_s,
        
        "best_y": best_y,
        "ei_acq": ei_acq,
        "ucb_acq": ucb_acq,
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
