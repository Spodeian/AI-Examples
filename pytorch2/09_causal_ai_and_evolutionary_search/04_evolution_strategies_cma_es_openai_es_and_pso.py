"""Evolution Strategies CMA ES OpenAI ES and PSO — PyTorch 2.x reference implementation.

Topic: 09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso
Golden weights: weights/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.safetensors

``python pytorch2/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.py`` verifies against the golden checkpoint;
``python pytorch2/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def openai_es_gradient_estimate(
    epsilons: torch.Tensor,
    pos_fitness: torch.Tensor,
    neg_fitness: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """OpenAI Evolution Strategies antithetic gradient estimator (Salimans et al. 2017).
    
    grad = 1 / (2 * P * sigma) * sum_i (F_i^+ - F_i^-) * eps_i
    """
    p = epsilons.shape[0]
    diff = (pos_fitness - neg_fitness).unsqueeze(-1)  # [P, 1]
    grad = torch.sum(diff * epsilons, dim=0) / (2.0 * float(p) * sigma)
    return grad


def cma_es_mean_and_cov_step(
    mu_old: torch.Tensor,
    cov_old: torch.Tensor,
    population: torch.Tensor,
    weights_elite: torch.Tensor,
    c_cov: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CMA-ES (Hansen 2006) elite recombination and rank-mu covariance matrix update.
    
    mu_new = sum_i w_i * x_i
    y_i = x_i - mu_old
    C_mu = sum_i w_i * (y_i y_i^T)
    C_new = (1 - c_cov) * C_old + c_cov * C_mu
    """
    w_col = weights_elite.unsqueeze(-1)  # [K, 1]
    mu_new = torch.sum(w_col * population, dim=0)

    y = population - mu_old.unsqueeze(0)  # [K, D]
    c_mu = y.t() @ (w_col * y)            # [D, D]
    cov_new = (1.0 - c_cov) * cov_old + c_cov * c_mu

    return mu_new, cov_new


def pso_velocity_and_position_step(
    x: torch.Tensor,
    v: torch.Tensor,
    p_best: torch.Tensor,
    g_best: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    w_inertia: float = 0.7,
    c1: float = 1.4,
    c2: float = 1.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Particle Swarm Optimization (Kennedy & Eberhart 1995) kinematic update.
    
    v_{t+1} = w * v_t + c1 * r1 * (p_best - x_t) + c2 * r2 * (g_best - x_t)
    x_{t+1} = x_t + v_{t+1}
    """
    cognitive = c1 * r1 * (p_best - x)
    social = c2 * r2 * (g_best - x)
    v_next = w_inertia * v + cognitive + social
    x_next = x + v_next
    return v_next, x_next


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. OpenAI ES antithetic inputs (P=4 perturbations, D=4 dimensions)
    es_epsilons = torch.tensor([
        [0.4, -0.8, 1.2, -0.3],
        [-0.5, 0.7, 0.2, 0.9],
        [1.1, -0.2, -0.6, 0.4],
        [-0.9, 0.3, 0.5, -0.7],
    ], dtype=torch.float32)
    es_pos_fit = torch.tensor([3.5, 2.1, 4.2, 1.8], dtype=torch.float32)
    es_neg_fit = torch.tensor([2.0, 2.9, 3.1, 2.5], dtype=torch.float32)
    es_grad_ref = openai_es_gradient_estimate(es_epsilons, es_pos_fit, es_neg_fit, sigma=0.1)

    # 2. CMA-ES inputs
    cma_mu_old = torch.tensor([0.5, -0.2, 1.0, 0.3], dtype=torch.float32)
    cma_cov_old = torch.eye(4, dtype=torch.float32) * 0.5
    cma_pop = torch.tensor([
        [0.8, 0.1, 1.3, 0.6],
        [0.6, -0.1, 0.9, 0.4],
        [0.4, -0.4, 0.8, 0.1],
        [0.3, -0.3, 0.7, 0.2],
    ], dtype=torch.float32)
    cma_weights = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    cma_mu_new_ref, cma_cov_new_ref = cma_es_mean_and_cov_step(
        cma_mu_old, cma_cov_old, cma_pop, cma_weights, c_cov=0.2
    )

    # 3. PSO inputs (N=4 particles, D=4 dimensions)
    pso_x = torch.tensor([
        [1.0, -0.5, 2.0, 0.2],
        [-1.2, 0.8, 0.5, -0.9],
        [0.3, 1.5, -0.4, 0.7],
        [1.8, -0.2, 1.1, -0.3],
    ], dtype=torch.float32)
    pso_v = torch.tensor([
        [0.1, -0.05, 0.2, 0.0],
        [-0.1, 0.1, 0.05, -0.1],
        [0.05, 0.2, -0.1, 0.1],
        [0.2, -0.1, 0.1, -0.05],
    ], dtype=torch.float32)
    pso_pbest = torch.tensor([
        [0.8, -0.2, 1.5, 0.1],
        [-0.9, 0.5, 0.3, -0.6],
        [0.1, 1.2, -0.2, 0.5],
        [1.4, -0.1, 0.8, -0.1],
    ], dtype=torch.float32)
    pso_gbest = torch.tensor([[0.5, 0.0, 1.0, 0.0]], dtype=torch.float32)
    pso_r1 = torch.tensor([
        [0.5, 0.3, 0.8, 0.2],
        [0.6, 0.4, 0.7, 0.5],
        [0.3, 0.8, 0.2, 0.6],
        [0.7, 0.5, 0.4, 0.3],
    ], dtype=torch.float32)
    pso_r2 = torch.tensor([
        [0.4, 0.6, 0.3, 0.7],
        [0.5, 0.5, 0.6, 0.4],
        [0.7, 0.2, 0.8, 0.3],
        [0.3, 0.4, 0.5, 0.6],
    ], dtype=torch.float32)

    pso_v_next_ref, pso_x_next_ref = pso_velocity_and_position_step(
        pso_x, pso_v, pso_pbest, pso_gbest, pso_r1, pso_r2, w_inertia=0.7, c1=1.4, c2=1.4
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "es_epsilons": es_epsilons,
        "es_pos_fit": es_pos_fit,
        "es_neg_fit": es_neg_fit,
        "es_grad_ref": es_grad_ref,
        "cma_mu_old": cma_mu_old,
        "cma_cov_old": cma_cov_old,
        "cma_pop": cma_pop,
        "cma_weights": cma_weights,
        "cma_mu_new_ref": cma_mu_new_ref,
        "cma_cov_new_ref": cma_cov_new_ref,
        "pso_x": pso_x,
        "pso_v": pso_v,
        "pso_pbest": pso_pbest,
        "pso_gbest": pso_gbest,
        "pso_r1": pso_r1,
        "pso_r2": pso_r2,
        "pso_v_next_ref": pso_v_next_ref,
        "pso_x_next_ref": pso_x_next_ref,
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
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
