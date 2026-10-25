"""Evolution Strategies CMA Es Openai Es and Pso — functional JAX implementation.

Topic: 09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso. Run ``python jax/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def openai_es_gradient_estimate(
    epsilons: jnp.ndarray,
    pos_fitness: jnp.ndarray,
    neg_fitness: jnp.ndarray,
    sigma: float = 0.1,
) -> jnp.ndarray:
    p = epsilons.shape[0]
    diff = jnp.expand_dims(pos_fitness - neg_fitness, -1)
    grad = jnp.sum(diff * epsilons, axis=0) / (2.0 * float(p) * sigma)
    return grad


def cma_es_mean_and_cov_step(
    mu_old: jnp.ndarray,
    cov_old: jnp.ndarray,
    population: jnp.ndarray,
    weights_elite: jnp.ndarray,
    c_cov: float = 0.2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    w_col = jnp.expand_dims(weights_elite, -1)
    mu_new = jnp.sum(w_col * population, axis=0)
    y = population - jnp.expand_dims(mu_old, 0)
    c_mu = y.T @ (w_col * y)
    cov_new = (1.0 - c_cov) * cov_old + c_cov * c_mu
    return mu_new, cov_new


def pso_velocity_and_position_step(
    x: jnp.ndarray,
    v: jnp.ndarray,
    p_best: jnp.ndarray,
    g_best: jnp.ndarray,
    r1: jnp.ndarray,
    r2: jnp.ndarray,
    w_inertia: float = 0.7,
    c1: float = 1.4,
    c2: float = 1.4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    cognitive = c1 * r1 * (p_best - x)
    social = c2 * r2 * (g_best - x)
    v_next = w_inertia * v + cognitive + social
    x_next = x + v_next
    return v_next, x_next


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # ES
    es_grad = openai_es_gradient_estimate(g["es_epsilons"], g["es_pos_fit"], g["es_neg_fit"], sigma=0.1)
    assert float(jnp.max(jnp.abs(es_grad - g["es_grad_ref"]))) < 1e-5, "ES gradient mismatch"

    # CMA-ES
    cma_mu, cma_cov = cma_es_mean_and_cov_step(g["cma_mu_old"], g["cma_cov_old"], g["cma_pop"], g["cma_weights"], c_cov=0.2)
    assert float(jnp.max(jnp.abs(cma_mu - g["cma_mu_new_ref"]))) < 1e-5, "CMA-ES mu mismatch"
    assert float(jnp.max(jnp.abs(cma_cov - g["cma_cov_new_ref"]))) < 1e-5, "CMA-ES cov mismatch"

    # PSO
    pso_v, pso_x = pso_velocity_and_position_step(
        g["pso_x"], g["pso_v"], g["pso_pbest"], g["pso_gbest"], g["pso_r1"], g["pso_r2"], w_inertia=0.7, c1=1.4, c2=1.4
    )
    assert float(jnp.max(jnp.abs(pso_v - g["pso_v_next_ref"]))) < 1e-5, "PSO velocity mismatch"
    assert float(jnp.max(jnp.abs(pso_x - g["pso_x_next_ref"]))) < 1e-5, "PSO position mismatch"

    print("09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
