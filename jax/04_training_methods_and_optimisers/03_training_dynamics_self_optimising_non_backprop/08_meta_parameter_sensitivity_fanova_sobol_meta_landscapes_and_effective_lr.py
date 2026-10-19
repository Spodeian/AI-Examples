"""JAX / Equinox implementation for Category 04 Topic 22:
Meta-Parameter Sensitivity, fANOVA, Sobol' Indices, and Effective Learning Rate.

Provides functional, jit-compiled implementations of:
1. Functional ANOVA (fANOVA) variance decomposition and Sobol' indices.
2. Scale-invariant effective learning rate and equilibrium norm.
3. Chaotic unrolled meta-gradient exponential explosion.
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


@jax.jit
def compute_fanova_sobol_indices_jax(
    y_grid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes fANOVA variance decomposition and Sobol' indices in JAX."""
    total_var = jnp.var(y_grid)

    e_y_given_x1 = jnp.mean(y_grid, axis=1)
    e_y_given_x2 = jnp.mean(y_grid, axis=0)

    v1 = jnp.var(e_y_given_x1)
    v2 = jnp.var(e_y_given_x2)
    v12 = total_var - (v1 + v2)

    s1 = v1 / total_var
    s2 = v2 / total_var
    st1 = (v1 + v12) / total_var
    st2 = (v2 + v12) / total_var

    return total_var, v1, v2, s1, s2, st1, st2


@jax.jit
def compute_effective_lr_and_eq_norm_jax(
    w: jnp.ndarray,
    eta_nominal: jnp.ndarray,
    weight_decay: jnp.ndarray,
    grad_norm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scale-invariant effective learning rate and equilibrium norm in JAX."""
    w_norm_sq = jnp.sum(w ** 2)
    effective_lr = eta_nominal / w_norm_sq
    eq_norm = ((eta_nominal * (grad_norm ** 2)) / (2.0 * weight_decay)) ** 0.25
    return effective_lr, eq_norm


@jax.jit
def compute_chaotic_meta_gradient_explosion_jax(
    eig_max: jnp.ndarray,
    eta_meta: jnp.ndarray,
    steps_t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Chaotic meta-gradient explosion in JAX."""
    multiplier = jnp.abs(1.0 - eta_meta * eig_max)
    accumulated = multiplier ** steps_t
    return multiplier, accumulated


@jax.jit
def linear_anchor_forward_jax(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Linear anchor forward pass in JAX."""
    return x @ w + b


class MetaSensitivityUnifiedModuleJAX:
    """Unified container for JAX Topic 22 verification."""

    def __init__(self, weights_path: Path):
        weights = load_file(str(weights_path))

        # Sobol
        self.y_grid = jnp.array(weights["y_grid"])
        self.golden_total_var = jnp.array(weights["golden_total_var"])
        self.golden_v1 = jnp.array(weights["golden_v1"])
        self.golden_v2 = jnp.array(weights["golden_v2"])
        self.golden_s1 = jnp.array(weights["golden_s1"])
        self.golden_s2 = jnp.array(weights["golden_s2"])
        self.golden_st1 = jnp.array(weights["golden_st1"])
        self.golden_st2 = jnp.array(weights["golden_st2"])

        # Effective LR
        self.w = jnp.array(weights["w"])
        self.eta_nominal = jnp.array(weights["eta_nominal"])
        self.weight_decay = jnp.array(weights["weight_decay"])
        self.grad_norm = jnp.array(weights["grad_norm"])
        self.golden_effective_lr = jnp.array(weights["golden_effective_lr"])
        self.golden_eq_norm = jnp.array(weights["golden_eq_norm"])

        # Chaotic Meta-Gradient
        self.eig_max = jnp.array(weights["eig_max"])
        self.eta_meta = jnp.array(weights["eta_meta"])
        self.steps_t = jnp.array(weights["steps_t"])
        self.golden_spectral_multiplier = jnp.array(weights["golden_spectral_multiplier"])
        self.golden_accumulated_explosion = jnp.array(weights["golden_accumulated_explosion"])

        # Anchor
        self.anchor_x = jnp.array(weights["anchor_x"])
        self.anchor_w = jnp.array(weights["anchor_w"])
        self.anchor_b = jnp.array(weights["anchor_b"])
        self.golden_anchor_out = jnp.array(weights["golden_anchor_out"])

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. Sobol
        tot_var, v1, v2, s1, s2, st1, st2 = compute_fanova_sobol_indices_jax(self.y_grid)
        err_tot = float(jnp.max(jnp.abs(tot_var - self.golden_total_var[0])))
        err_v1 = float(jnp.max(jnp.abs(v1 - self.golden_v1[0])))
        err_v2 = float(jnp.max(jnp.abs(v2 - self.golden_v2[0])))
        err_s1 = float(jnp.max(jnp.abs(s1 - self.golden_s1[0])))
        err_s2 = float(jnp.max(jnp.abs(s2 - self.golden_s2[0])))
        err_st1 = float(jnp.max(jnp.abs(st1 - self.golden_st1[0])))
        err_st2 = float(jnp.max(jnp.abs(st2 - self.golden_st2[0])))
        err_sobol = max(err_tot, err_v1, err_v2, err_s1, err_s2, err_st1, err_st2)
        assert err_sobol < atol, f"JAX Sobol invariant failed: {err_sobol:.2e}"
        errors["sobol_err"] = err_sobol

        # 2. Effective LR
        eff_lr, eq_norm = compute_effective_lr_and_eq_norm_jax(self.w, self.eta_nominal, self.weight_decay, self.grad_norm)
        err_elr = float(jnp.max(jnp.abs(eff_lr - self.golden_effective_lr[0])))
        err_eqn = float(jnp.max(jnp.abs(eq_norm - self.golden_eq_norm[0])))
        err_eff = max(err_elr, err_eqn)
        assert err_eff < atol, f"JAX Effective LR invariant failed: {err_eff:.2e}"
        errors["eff_lr_err"] = err_eff

        # 3. Chaotic Meta-Gradient
        mult, accum = compute_chaotic_meta_gradient_explosion_jax(self.eig_max, self.eta_meta, self.steps_t)
        err_mult = float(jnp.max(jnp.abs(mult - self.golden_spectral_multiplier[0])))
        err_accum = float(jnp.max(jnp.abs(accum - self.golden_accumulated_explosion[0])))
        err_chaos = max(err_mult, err_accum)
        assert err_chaos < atol, f"JAX Chaotic Meta-Gradient invariant failed: {err_chaos:.2e}"
        errors["chaos_err"] = err_chaos

        # 4. Anchor
        anchor_out = linear_anchor_forward_jax(self.anchor_x, self.anchor_w, self.anchor_b)
        err_anchor = float(jnp.max(jnp.abs(anchor_out - self.golden_anchor_out)))
        assert err_anchor < atol, f"JAX Anchor invariant failed: {err_anchor:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr.safetensors")
    module = MetaSensitivityUnifiedModuleJAX(p)
    errs = module.verify_invariants(1e-5)
    print(f"JAX Topic 22 Invariant Parity Checks Passed: {errs}")
