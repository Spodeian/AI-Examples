"""Population Based Training PBT and DARTS Neural Architecture Search — functional JAX implementation.

Topic: 09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search. Run ``python jax/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def pbt_exploit_explore_step(
    scores: jnp.ndarray,
    weights_pop: jnp.ndarray,
    hyperparams_pop: jnp.ndarray,
    perturb_factors: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    top_idx = int(jnp.argmax(scores))
    bot_idx = int(jnp.argmin(scores))

    w_new = weights_pop.at[bot_idx].set(weights_pop[top_idx])
    h_new = hyperparams_pop.at[bot_idx].set(hyperparams_pop[top_idx] * perturb_factors)

    return w_new, h_new


def darts_mixed_op_forward(
    alpha: jnp.ndarray,
    candidate_outputs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    weights = jax.nn.softmax(alpha, axis=-1)
    w_expanded = jnp.expand_dims(jnp.expand_dims(weights, -1), -1)
    mixed_outputs = jnp.sum(w_expanded * candidate_outputs, axis=1)
    return mixed_outputs, weights


def darts_discrete_decode(
    alpha: jnp.ndarray,
    candidate_outputs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    selected_ops = jnp.argmax(alpha, axis=-1)
    e_indices = jnp.arange(alpha.shape[0])
    discrete_outputs = candidate_outputs[e_indices, selected_ops]
    return selected_ops.astype(jnp.float32), discrete_outputs


def darts_architecture_entropy(softmax_weights: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return -jnp.sum(softmax_weights * jnp.log(softmax_weights + eps))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # PBT
    w_up, h_up = pbt_exploit_explore_step(g["pbt_scores"], g["pbt_weights"], g["pbt_hyperparams"], g["pbt_perturb"])
    assert float(jnp.max(jnp.abs(w_up - g["pbt_weights_updated_ref"]))) < 1e-5, "PBT weights mismatch"
    assert float(jnp.max(jnp.abs(h_up - g["pbt_hyperparams_updated_ref"]))) < 1e-5, "PBT hyperparams mismatch"

    # DARTS mixed op
    mixed, weights = darts_mixed_op_forward(g["darts_alpha"], g["darts_candidate_ops"])
    assert float(jnp.max(jnp.abs(mixed - g["darts_mixed_out_ref"]))) < 1e-5, "DARTS mixed out mismatch"
    assert float(jnp.max(jnp.abs(weights - g["darts_softmax_weights_ref"]))) < 1e-5, "DARTS weights mismatch"

    # DARTS discrete decode
    ops, disc = darts_discrete_decode(g["darts_alpha"], g["darts_candidate_ops"])
    assert float(jnp.max(jnp.abs(ops - g["darts_selected_ops_ref"]))) < 1e-5, "DARTS ops mismatch"
    assert float(jnp.max(jnp.abs(disc - g["darts_discrete_out_ref"]))) < 1e-5, "DARTS discrete out mismatch"

    # DARTS entropy
    ent = darts_architecture_entropy(weights)
    assert float(jnp.max(jnp.abs(ent - g["darts_entropy_ref"]))) < 1e-5, "DARTS entropy mismatch"

    print("09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
