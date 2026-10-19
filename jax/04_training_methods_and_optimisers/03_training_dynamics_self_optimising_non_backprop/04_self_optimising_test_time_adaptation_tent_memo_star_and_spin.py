"""JAX / Equinox implementation for Category 04 Topic 18:
Self-Optimising Test-Time Adaptation: TENT, MEMO, STaR, and SPIN.

Provides functional, jit-compiled implementations of:
1. TENT: Test-Time Entropy Minimization updating normalization affine parameters (Wang et al., 2021).
2. MEMO: Marginal Entropy Minimization with One test sample across augmentations (Zhang et al., 2022).
3. SPIN: Self-Play Fine-Tuning preference optimization (Chen et al., 2024).
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


def _tent_loss_fn(scale: jnp.ndarray, shift: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    scaled_logits = logits * scale + shift
    log_p = jax.nn.log_softmax(scaled_logits, axis=-1)
    p = jax.nn.softmax(scaled_logits, axis=-1)
    return -jnp.mean(jnp.sum(p * log_p, axis=-1))


@jax.jit
def compute_tent_step_jax(
    logits: jnp.ndarray,
    scale: jnp.ndarray,
    shift: jnp.ndarray,
    lr: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """TENT step in JAX using automatic differentiation."""
    loss, (g_scale, g_shift) = jax.value_and_grad(_tent_loss_fn, argnums=(0, 1))(scale, shift, logits)
    scale_next = scale - lr * g_scale
    shift_next = shift - lr * g_shift
    return scale_next, shift_next, loss, g_scale, g_shift


@jax.jit
def compute_memo_marginal_entropy_jax(
    aug_logits: jnp.ndarray,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """MEMO marginal entropy in JAX."""
    aug_probs = jax.nn.softmax(aug_logits, axis=-1)
    marginal_p = jnp.mean(aug_probs, axis=0) # [C]
    marginal_entropy = -jnp.sum(marginal_p * jnp.log(marginal_p + eps))
    return marginal_p, marginal_entropy


@jax.jit
def compute_spin_loss_jax(
    logp_win: jnp.ndarray,
    logp_lose: jnp.ndarray,
    beta: float = 0.5,
) -> jnp.ndarray:
    """SPIN preference loss in JAX."""
    diff = beta * (logp_win - logp_lose)
    return -jnp.mean(jax.nn.log_sigmoid(diff))


@jax.jit
def anchor_forward_jax(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Linear anchor forward in JAX."""
    return x @ w + b


def verify_invariants(safetensors_path: str | Path, atol: float = 1e-5) -> dict[str, float]:
    """Verifies JAX implementation against golden reference tensors."""
    weights = load_file(str(safetensors_path))
    errors: dict[str, float] = {}

    # 1. TENT
    tent_logits = jnp.array(weights["tent_logits"])
    tent_scale = jnp.array(weights["tent_scale"])
    tent_shift = jnp.array(weights["tent_shift"])
    golden_ent = jnp.array(weights["golden_tent_entropy"])[0]
    golden_g_s = jnp.array(weights["golden_scale_grad"])
    golden_g_sh = jnp.array(weights["golden_shift_grad"])
    golden_s_next = jnp.array(weights["golden_scale_next"])
    golden_sh_next = jnp.array(weights["golden_shift_next"])

    s_next, sh_next, ent, g_s, g_sh = compute_tent_step_jax(tent_logits, tent_scale, tent_shift, lr=0.05)
    err_ent = float(jnp.abs(ent - golden_ent))
    err_s = float(jnp.max(jnp.abs(s_next - golden_s_next)))
    err_sh = float(jnp.max(jnp.abs(sh_next - golden_sh_next)))
    err_gs = float(jnp.max(jnp.abs(g_s - golden_g_s)))
    err_gsh = float(jnp.max(jnp.abs(g_sh - golden_g_sh)))
    err_tent = max(err_ent, err_s, err_sh, err_gs, err_gsh)
    assert err_tent < atol, f"JAX TENT failed: {err_tent:.2e}"
    errors["tent_err"] = err_tent

    # 2. MEMO
    memo_logits = jnp.array(weights["memo_aug_logits"])
    golden_marg_p = jnp.array(weights["golden_marginal_p"])
    golden_memo_ent = jnp.array(weights["golden_memo_entropy"])[0]

    marg_p, memo_ent = compute_memo_marginal_entropy_jax(memo_logits)
    err_mp = float(jnp.max(jnp.abs(marg_p - golden_marg_p)))
    err_me = float(jnp.abs(memo_ent - golden_memo_ent))
    err_memo = max(err_mp, err_me)
    assert err_memo < atol, f"JAX MEMO failed: {err_memo:.2e}"
    errors["memo_err"] = err_memo

    # 3. SPIN
    logp_win = jnp.array(weights["logp_win"])
    logp_lose = jnp.array(weights["logp_lose"])
    golden_spin = jnp.array(weights["golden_spin_loss"])[0]

    spin_loss = compute_spin_loss_jax(logp_win, logp_lose, beta=0.5)
    err_spin = float(jnp.abs(spin_loss - golden_spin))
    assert err_spin < atol, f"JAX SPIN failed: {err_spin:.2e}"
    errors["spin_err"] = err_spin

    # 4. Anchor
    anchor_x = jnp.array(weights["anchor_x"])
    anchor_w = jnp.array(weights["anchor_w"])
    anchor_b = jnp.array(weights["anchor_b"])
    golden_anchor = jnp.array(weights["golden_anchor_out"])

    anchor_out = anchor_forward_jax(anchor_x, anchor_w, anchor_b)
    err_anchor = float(jnp.max(jnp.abs(anchor_out - golden_anchor)))
    assert err_anchor < atol, f"JAX Anchor failed: {err_anchor:.2e}"
    errors["anchor_err"] = err_anchor

    return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.safetensors")
    errs = verify_invariants(p, 1e-5)
    print(f"JAX Invariant Parity Checks Passed: {errs}")
