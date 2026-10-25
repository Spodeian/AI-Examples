"""Abliteration Refusal Surgery Circuit Breakers and Sleeper Agents — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents. Run ``python jax/11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def extract_refusal_direction(
    activations_harmful: jnp.ndarray,
    activations_harmless: jnp.ndarray,
) -> jnp.ndarray:
    """Mean refusal direction r_hat."""
    mu_harmful = jnp.mean(activations_harmful, axis=0)
    mu_harmless = jnp.mean(activations_harmless, axis=0)
    diff = mu_harmful - mu_harmless
    norm = jnp.linalg.norm(diff) + 1e-8
    return diff / norm


def abliterate_weights(
    weight: jnp.ndarray,
    refusal_direction: jnp.ndarray,
) -> jnp.ndarray:
    """Abliterates refusal mechanism: W' = W - r_hat * (r_hat^T W)."""
    r = refusal_direction.reshape(-1, 1)  # [D_in, 1]
    proj = r @ (r.T @ weight)
    return weight - proj


def contrastive_activation_addition(
    h: jnp.ndarray,
    steering_vector: jnp.ndarray,
    alpha: float = 1.0,
) -> jnp.ndarray:
    """CAA / RepE dynamic steering."""
    return h + float(alpha) * steering_vector


def circuit_breaker_loss(
    h_harmful: jnp.ndarray,
    h_target_orthogonal: jnp.ndarray,
    h_benign: jnp.ndarray,
    h_ref_benign: jnp.ndarray,
    lambda_cb: float = 1.0,
) -> jnp.ndarray:
    """Safety Circuit Breakers loss."""
    dot = jnp.sum(h_harmful * h_target_orthogonal, axis=-1)
    norm_h = jnp.linalg.norm(h_harmful, axis=-1) + 1e-8
    norm_t = jnp.linalg.norm(h_target_orthogonal, axis=-1) + 1e-8
    cos_sim = dot / (norm_h * norm_t)
    l_breaker = jnp.mean(cos_sim ** 2)
    
    l_benign = jnp.mean((h_benign - h_ref_benign) ** 2)
    return l_breaker + float(lambda_cb) * l_benign


def sleeper_agent_probe(
    h: jnp.ndarray,
    probe_w: jnp.ndarray,
    probe_b: jnp.ndarray,
) -> jnp.ndarray:
    """Linear probe for auditing backdoor sleeper agents."""
    return h @ probe_w + probe_b


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. Refusal direction
    r_hat = extract_refusal_direction(g["act_harmful"], g["act_harmless"])
    assert float(jnp.max(jnp.abs(r_hat - g["refusal_dir_ref"]))) < 1e-5
    
    # 3. Abliterate weights
    w_abl = abliterate_weights(g["w_mlp"], r_hat)
    assert float(jnp.max(jnp.abs(w_abl - g["w_ablated_ref"]))) < 1e-5
    
    # 4. CAA steering
    h_steered = contrastive_activation_addition(g["h_query"], r_hat, alpha=2.0)
    assert float(jnp.max(jnp.abs(h_steered - g["h_steered_ref"]))) < 1e-5
    
    # 5. Circuit Breakers
    cb_val = circuit_breaker_loss(g["h_harmful"], g["h_target_ortho"], g["h_benign"], g["h_ref_benign"], lambda_cb=1.5)
    assert float(jnp.abs(cb_val - g["cb_loss_ref"])) < 1e-5
    
    # 6. Sleeper probe
    scores = sleeper_agent_probe(g["h_query"], g["probe_w"], g["probe_b"])
    assert float(jnp.max(jnp.abs(scores - g["probe_scores_ref"]))) < 1e-5
    
    print("11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
