"""Active Learning Bald Coreset and Badge — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def bald_score(mc_probs: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    mean_probs = jnp.mean(mc_probs, axis=0)
    entropy_mean = -jnp.sum(mean_probs * jnp.log(mean_probs + eps), axis=-1)
    mean_entropy = jnp.mean(-jnp.sum(mc_probs * jnp.log(mc_probs + eps), axis=-1), axis=0)
    return entropy_mean - mean_entropy


def coreset_min_distances(unlabeled_feats: jnp.ndarray, labeled_feats: jnp.ndarray) -> jnp.ndarray:
    diff = jnp.expand_dims(unlabeled_feats, 1) - jnp.expand_dims(labeled_feats, 0)
    dists = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)
    return jnp.min(dists, axis=1)


def badge_gradient_embeddings(feats: jnp.ndarray, probs: jnp.ndarray) -> jnp.ndarray:
    pred_class = jnp.argmax(probs, axis=-1)
    c = probs.shape[-1]
    one_hot = jax.nn.one_hot(pred_class, c)
    prob_diff = probs - one_hot
    outer = jnp.expand_dims(prob_diff, 2) * jnp.expand_dims(feats, 1)
    b = feats.shape[0]
    return jnp.reshape(outer, (b, -1))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. BALD
    bald = bald_score(g["mc_probs"])
    err_bald = float(jnp.max(jnp.abs(bald - g["bald_score_ref"])))
    assert err_bald < 1e-5, f"bald error {err_bald:.2e}"

    # 3. Core-Set
    min_dists = coreset_min_distances(g["unlabeled_feats"], g["labeled_feats"])
    err_core = float(jnp.max(jnp.abs(min_dists - g["coreset_min_dists_ref"])))
    assert err_core < 1e-5, f"coreset error {err_core:.2e}"

    # 4. BADGE
    badge = badge_gradient_embeddings(g["feats"], g["probs"])
    err_badge = float(jnp.max(jnp.abs(badge - g["badge_embeds_ref"])))
    assert err_badge < 1e-5, f"badge error {err_badge:.2e}"

    print("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
