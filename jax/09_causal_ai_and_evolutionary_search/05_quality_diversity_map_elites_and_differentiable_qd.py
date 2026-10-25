"""Quality Diversity Map Elites and Differentiable Qd — functional JAX implementation.

Topic: 09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd. Run ``python jax/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def map_elites_grid_indices(
    descriptors: jnp.ndarray,
    bounds_min: jnp.ndarray,
    bounds_max: jnp.ndarray,
    grid_size: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = (descriptors - jnp.expand_dims(bounds_min, 0)) / (
        jnp.expand_dims(bounds_max, 0) - jnp.expand_dims(bounds_min, 0) + 1e-8
    )
    indices = jnp.floor(scale * jnp.expand_dims(grid_size, 0)).astype(jnp.int32)
    indices = jnp.clip(indices, 0, (grid_size - 1).astype(jnp.int32))
    cell_idx = indices[:, 0] * grid_size[1].astype(jnp.int32) + indices[:, 1]
    return indices.astype(jnp.float32), cell_idx.astype(jnp.float32)


def map_elites_archive_update(
    archive_fitness: jnp.ndarray,
    candidate_fitness: jnp.ndarray,
    cell_indices: jnp.ndarray,
) -> jnp.ndarray:
    # Segment max update via jax.ops or index_max / loop
    res = archive_fitness
    indices = cell_indices.astype(jnp.int32)
    for i in range(candidate_fitness.shape[0]):
        c = indices[i]
        res = res.at[c].set(jnp.maximum(res[c], candidate_fitness[i]))
    return res


def dqd_gradient_arborescence_step(
    grad_f: jnp.ndarray,
    jac_b: jnp.ndarray,
    coeffs: jnp.ndarray,
) -> jnp.ndarray:
    c0 = coeffs[0]
    c_b = jnp.expand_dims(coeffs[1:], -1)
    step = c0 * grad_f + jnp.sum(c_b * jac_b, axis=0)
    return step


def dqd_orthogonalize_behavior_gradients(
    grad_f: jnp.ndarray,
    jac_b: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    norm_sq = jnp.sum(grad_f * grad_f) + eps
    proj_coeffs = (jac_b @ grad_f) / norm_sq
    ortho_jac = jac_b - jnp.expand_dims(proj_coeffs, -1) * jnp.expand_dims(grad_f, 0)
    return ortho_jac


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Grid indices
    g_idx, c_idx = map_elites_grid_indices(g["descriptors"], g["b_min"], g["b_max"], g["grid_size"])
    assert float(jnp.max(jnp.abs(g_idx - g["grid_indices_ref"]))) < 1e-5, "Grid indices mismatch"
    assert float(jnp.max(jnp.abs(c_idx - g["cell_indices_ref"]))) < 1e-5, "Cell indices mismatch"

    # Archive update
    up = map_elites_archive_update(g["archive_fit"], g["cand_fitness"], g["cell_indices_ref"])
    assert float(jnp.max(jnp.abs(up - g["archive_fit_updated_ref"]))) < 1e-5, "Archive update mismatch"

    # DQD step
    step = dqd_gradient_arborescence_step(g["grad_f"], g["jac_b"], g["dqd_coeffs"])
    assert float(jnp.max(jnp.abs(step - g["dqd_step_ref"]))) < 1e-5, "DQD step mismatch"

    # DQD ortho
    ortho = dqd_orthogonalize_behavior_gradients(g["grad_f"], g["jac_b"])
    assert float(jnp.max(jnp.abs(ortho - g["dqd_ortho_jac_ref"]))) < 1e-5, "DQD ortho mismatch"

    print("09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
