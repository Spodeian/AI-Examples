"""Topological Data Analysis Persistent Homology and Singular Learning LLC — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc. Run ``python jax/10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def persistence_diagram_lifespans(birth_death_pairs: jnp.ndarray) -> jnp.ndarray:
    return birth_death_pairs[:, 1] - birth_death_pairs[:, 0]


def persistence_landscape_level1(
    birth_death_pairs: jnp.ndarray,
    eval_points: jnp.ndarray,
) -> jnp.ndarray:
    t = eval_points[None, :]
    b = birth_death_pairs[:, 0:1]
    d = birth_death_pairs[:, 1:2]
    tri = jnp.maximum(0.0, jnp.minimum(t - b, d - t))
    return jnp.max(tri, axis=0)


def singular_learning_llc_energy(
    sample_losses: jnp.ndarray,
    loss_star: jnp.ndarray,
    n_samples: float,
    beta_inv_temp: float = 1.0,
) -> jnp.ndarray:
    mean_diff = jnp.mean(sample_losses - loss_star)
    return (float(n_samples) * float(beta_inv_temp) / 2.0) * mean_diff


def betti_0_filtration_counts(
    pairwise_distances: jnp.ndarray,
    epsilon_thresholds: jnp.ndarray,
) -> jnp.ndarray:
    n = pairwise_distances.shape[0]
    e_count = epsilon_thresholds.shape[0]
    counts = []

    for idx in range(e_count):
        eps = float(epsilon_thresholds[idx])
        adj = pairwise_distances <= eps
        visited = [False] * n
        comp = 0
        for i in range(n):
            if not visited[i]:
                comp += 1
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    for nbr in range(n):
                        if bool(adj[curr, nbr]) and not visited[nbr]:
                            visited[nbr] = True
                            queue.append(nbr)
        counts.append(float(comp))

    return jnp.array(counts, dtype=jnp.float32)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Persistence lifespans
    life = persistence_diagram_lifespans(g["birth_death_pairs"])
    assert float(jnp.max(jnp.abs(life - g["lifespans_ref"]))) < 1e-5, "Lifespans mismatch"

    # 3. Persistence landscape level 1
    land = persistence_landscape_level1(g["birth_death_pairs"], g["eval_points"])
    assert float(jnp.max(jnp.abs(land - g["landscape_ref"]))) < 1e-5, "Landscape mismatch"

    # 4. Singular Learning Theory LLC
    llc = singular_learning_llc_energy(g["sample_losses"], g["loss_star"], 500.0, 1.0)
    assert float(jnp.max(jnp.abs(llc - g["llc_ref"]))) < 1e-5, "LLC mismatch"

    # 5. Betti-0 filtration
    betti0 = betti_0_filtration_counts(g["pairwise_distances"], g["epsilon_thresholds"])
    assert float(jnp.max(jnp.abs(betti0 - g["betti0_ref"]))) < 1e-5, "Betti-0 mismatch"

    print("10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
