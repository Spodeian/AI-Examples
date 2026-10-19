"""JAX / Equinox implementation for Category 04 Topic 21:
Scaling Laws, Data Deduplication (MinHash), DoReMi, and Model Collapse.

Provides functional, jit-compiled implementations of:
1. Chinchilla parametric scaling laws and compute-optimal parameter/data allocation.
2. DoReMi domain weight optimization via exponentiated gradient updates.
3. MinHash pairwise Jaccard similarity estimation.
4. Model collapse variance decay and real-data accumulation cure.
5. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


@jax.jit
def compute_chinchilla_loss_and_opt_jax(
    n: jnp.ndarray,
    d: jnp.ndarray,
    e_const: float = 1.69,
    a_const: float = 40.0,
    b_const: float = 50.0,
    alpha: float = 0.34,
    beta: float = 0.28,
    g_n: float = 0.6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Chinchilla scaling calculation in JAX."""
    loss = e_const + a_const * (n ** (-alpha)) + b_const * (d ** (-beta))

    exp_a = beta / (alpha + beta)
    exp_b = alpha / (alpha + beta)
    budget_c = 6.0 * n * d
    g_d = 1.0 / (6.0 * g_n)

    n_opt = g_n * (budget_c ** exp_a)
    d_opt = g_d * (budget_c ** exp_b)
    return loss, n_opt, d_opt


@jax.jit
def doremi_update_step_jax(
    w_init: jnp.ndarray,
    losses_model: jnp.ndarray,
    losses_ref: jnp.ndarray,
    eta: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """DoReMi domain weight update in JAX."""
    excess_loss = losses_model - losses_ref
    log_w = jnp.log(w_init) + eta * excess_loss
    w_next = jax.nn.softmax(log_w, axis=-1)
    return excess_loss, w_next


@jax.jit
def minhash_pairwise_jaccard_jax(
    signatures: jnp.ndarray,
) -> jnp.ndarray:
    """MinHash pairwise Jaccard similarity matrix in JAX."""
    # signatures: [B, K]
    matches = (jnp.expand_dims(signatures, 1) == jnp.expand_dims(signatures, 0)).astype(jnp.float32)
    return jnp.mean(matches, axis=-1)


@jax.jit
def model_collapse_variance_cure_jax(
    var_synth: jnp.ndarray,
    var_real: jnp.ndarray,
    lambda_real: float = 0.25,
) -> jnp.ndarray:
    """Model collapse accumulation cure in JAX."""
    return (1.0 - lambda_real) * var_synth + lambda_real * var_real


@jax.jit
def linear_anchor_forward_jax(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Linear anchor forward pass in JAX."""
    return x @ w + b


class ScalingLawsUnifiedModuleJAX:
    """Unified container for JAX Topic 21 verification."""

    def __init__(self, weights_path: Path):
        weights = load_file(str(weights_path))

        # Chinchilla
        self.n_val = jnp.array(weights["n_val"])
        self.d_val = jnp.array(weights["d_val"])
        self.golden_chinchilla_loss = jnp.array(weights["golden_chinchilla_loss"])
        self.golden_n_opt = jnp.array(weights["golden_n_opt"])
        self.golden_d_opt = jnp.array(weights["golden_d_opt"])

        # DoReMi
        self.w_doremi_init = jnp.array(weights["w_doremi_init"])
        self.losses_model = jnp.array(weights["losses_model"])
        self.losses_ref = jnp.array(weights["losses_ref"])
        self.golden_excess_loss = jnp.array(weights["golden_excess_loss"])
        self.golden_w_doremi_next = jnp.array(weights["golden_w_doremi_next"])

        # MinHash
        self.signatures = jnp.array(weights["signatures"])
        self.golden_jaccard_matrix = jnp.array(weights["golden_jaccard_matrix"])

        # Model collapse
        self.var_synth = jnp.array(weights["var_synth"])
        self.var_real = jnp.array(weights["var_real"])
        self.golden_cured_var = jnp.array(weights["golden_cured_var"])

        # Anchor
        self.anchor_x = jnp.array(weights["anchor_x"])
        self.anchor_w = jnp.array(weights["anchor_w"])
        self.anchor_b = jnp.array(weights["anchor_b"])
        self.golden_anchor_out = jnp.array(weights["golden_anchor_out"])

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. Chinchilla
        loss, n_opt, d_opt = compute_chinchilla_loss_and_opt_jax(self.n_val, self.d_val)
        err_l = float(jnp.max(jnp.abs(loss - self.golden_chinchilla_loss)))
        err_n = float(jnp.max(jnp.abs(n_opt - self.golden_n_opt)))
        err_d = float(jnp.max(jnp.abs(d_opt - self.golden_d_opt)))
        err_chinchilla = max(err_l, err_n, err_d)
        assert err_chinchilla < atol, f"JAX Chinchilla invariant failed: {err_chinchilla:.2e}"
        errors["chinchilla_err"] = err_chinchilla

        # 2. DoReMi
        excess_l, w_next = doremi_update_step_jax(self.w_doremi_init, self.losses_model, self.losses_ref, eta=1.0)
        err_ex = float(jnp.max(jnp.abs(excess_l - self.golden_excess_loss)))
        err_wn = float(jnp.max(jnp.abs(w_next - self.golden_w_doremi_next)))
        err_doremi = max(err_ex, err_wn)
        assert err_doremi < atol, f"JAX DoReMi invariant failed: {err_doremi:.2e}"
        errors["doremi_err"] = err_doremi

        # 3. MinHash
        jaccard_m = minhash_pairwise_jaccard_jax(self.signatures)
        err_minhash = float(jnp.max(jnp.abs(jaccard_m - self.golden_jaccard_matrix)))
        assert err_minhash < atol, f"JAX MinHash invariant failed: {err_minhash:.2e}"
        errors["minhash_err"] = err_minhash

        # 4. Model collapse
        cured_v = model_collapse_variance_cure_jax(self.var_synth, self.var_real, lambda_real=0.25)
        err_collapse = float(jnp.max(jnp.abs(cured_v - self.golden_cured_var)))
        assert err_collapse < atol, f"JAX Model Collapse invariant failed: {err_collapse:.2e}"
        errors["collapse_err"] = err_collapse

        # 5. Anchor
        anchor_out = linear_anchor_forward_jax(self.anchor_x, self.anchor_w, self.anchor_b)
        err_anchor = float(jnp.max(jnp.abs(anchor_out - self.golden_anchor_out)))
        assert err_anchor < atol, f"JAX Anchor invariant failed: {err_anchor:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse.safetensors")
    module = ScalingLawsUnifiedModuleJAX(p)
    errs = module.verify_invariants(1e-5)
    print(f"JAX Topic 21 Invariant Parity Checks Passed: {errs}")
