"""JAX / Functional implementation for Category 04 Topic 11:
Combinatorial and Discrete: Simulated Annealing, Parallel Tempering, and Sinkhorn.

Provides pure functional, memory-efficient implementations of:
1. Gumbel-Sinkhorn continuous permutation relaxation via alternating row/column scalings.
2. Cross-Entropy Method (CEM) distribution parameter estimation.
3. Metropolis-Hastings acceptance probability for Simulated Annealing (SA).
4. Replica-Exchange Parallel Tempering swap probability.
"""

from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def run_sinkhorn(log_alpha: np.ndarray, n_iters: int = 10, tau: float = 1.0) -> np.ndarray:
    """Sinkhorn normalization over square cost/score matrix."""
    P = np.exp(log_alpha / tau)
    for _ in range(n_iters):
        P = P / (np.sum(P, axis=-1, keepdims=True) + 1e-12)
        P = P / (np.sum(P, axis=-2, keepdims=True) + 1e-12)
    return P


def step_cem(
    candidates: np.ndarray,
    scores: np.ndarray,
    elite_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-Entropy Method step: select elite candidates, update mean and std."""
    n = candidates.shape[0]
    n_elite = max(1, int(n * elite_frac))
    elite_indices = np.argsort(scores)[:n_elite]
    elite_samples = candidates[elite_indices]
    new_mean = np.mean(elite_samples, axis=0)
    new_std = np.std(elite_samples, axis=0, ddof=0) + 1e-6
    return new_mean, new_std


def sa_acceptance_prob(delta_e: np.ndarray, temp: float) -> np.ndarray:
    """Metropolis-Hastings acceptance probability: min(1, exp(-delta_e / T))."""
    return np.clip(np.exp(-delta_e / temp), a_min=None, a_max=1.0)


def parallel_tempering_swap_prob(e1: float, e2: float, temp1: float, temp2: float) -> float:
    """Replica exchange swap probability: min(1, exp((beta1 - beta2) * (e1 - e2)))."""
    beta1 = 1.0 / temp1
    beta2 = 1.0 / temp2
    delta = (beta1 - beta2) * (e1 - e2)
    return float(np.clip(np.exp(delta), a_min=None, a_max=1.0))


def linear_anchor_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear anchor forward pass: x @ w + b."""
    return np.matmul(x, w) + b


def verify_invariants(weights_path: str | Path, atol: float = 1e-5) -> dict[str, float]:
    """Verify parity against golden reference tensors."""
    weights = load_file(str(weights_path))
    errors: dict[str, float] = {}

    # 1. Sinkhorn Invariant
    sinkhorn_out = run_sinkhorn(weights["log_alpha"], n_iters=10, tau=0.5)
    err_sinkhorn = float(np.max(np.abs(sinkhorn_out - weights["golden_sinkhorn_out"])))
    assert err_sinkhorn < atol, f"Sinkhorn invariant failed: err={err_sinkhorn:.2e} >= {atol:.2e}"
    errors["sinkhorn_err"] = err_sinkhorn

    # 2. CEM Invariant
    mean, std = step_cem(weights["candidates"], weights["scores"], elite_frac=0.25)
    err_cem_m = float(np.max(np.abs(mean - weights["golden_cem_mean"])))
    err_cem_s = float(np.max(np.abs(std - weights["golden_cem_std"])))
    err_cem = max(err_cem_m, err_cem_s)
    assert err_cem < atol, f"CEM invariant failed: err={err_cem:.2e} >= {atol:.2e}"
    errors["cem_err"] = err_cem

    # 3. Simulated Annealing Invariant
    sa_temp = float(weights["sa_temp"][0])
    sa_probs = sa_acceptance_prob(weights["delta_e"], sa_temp)
    err_sa = float(np.max(np.abs(sa_probs - weights["golden_sa_probs"])))
    assert err_sa < atol, f"SA invariant failed: err={err_sa:.2e} >= {atol:.2e}"
    errors["sa_err"] = err_sa

    # 4. Parallel Tempering Invariant
    pt_e1 = float(weights["pt_e1"][0])
    pt_e2 = float(weights["pt_e2"][0])
    pt_t1 = float(weights["pt_t1"][0])
    pt_t2 = float(weights["pt_t2"][0])
    swap_prob = parallel_tempering_swap_prob(pt_e1, pt_e2, pt_t1, pt_t2)
    err_pt = abs(swap_prob - float(weights["golden_pt_swap_prob"][0]))
    assert err_pt < atol, f"PT invariant failed: err={err_pt:.2e} >= {atol:.2e}"
    errors["pt_err"] = err_pt

    # 5. Linear Anchor Invariant
    anchor_out = linear_anchor_forward(weights["anchor_x"], weights["anchor_w"], weights["anchor_b"])
    err_anchor = float(np.max(np.abs(anchor_out - weights["golden_anchor_out"])))
    assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
    errors["anchor_err"] = err_anchor

    return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.safetensors")
    errs = verify_invariants(p)
    print(f"JAX Invariant Parity Checks Passed: {errs}")
