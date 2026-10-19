"""JAX / Equinox implementation for Category 04 Topic 19:
Non-Backprop: Forward-Forward, Predictive Coding, EqProp, DFA, and Hebbian Learning.

Provides functional, jit-compiled implementations of:
1. Forward-Forward algorithm: goodness loss and layer normalization.
2. Predictive Coding: representation error minimization step.
3. Direct Feedback Alignment (DFA): fixed random feedback projection.
4. Oja's Hebbian Rule: normalized local PCA update.
5. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


@jax.jit
def forward_forward_loss_and_norm_jax(
    h_pos: jnp.ndarray,
    h_neg: jnp.ndarray,
    theta: float = 2.0,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-Forward goodness loss and layer normalization in JAX."""
    g_pos = jnp.sum(h_pos ** 2, axis=-1)
    g_neg = jnp.sum(h_neg ** 2, axis=-1)
    loss = jnp.mean(jax.nn.softplus(theta - g_pos) + jax.nn.softplus(g_neg - theta))
    norm_pos = jnp.linalg.norm(h_pos, axis=-1, keepdims=True) + eps
    h_norm = h_pos / norm_pos
    return loss, h_norm


@jax.jit
def predictive_coding_step_jax(
    x: jnp.ndarray,
    r: jnp.ndarray,
    w: jnp.ndarray,
    eta: float = 0.1,
    lam: float = 0.01,
) -> jnp.ndarray:
    """Predictive Coding step in JAX."""
    eps = x - r @ w
    return r + eta * (eps @ w.T - lam * r)


@jax.jit
def dfa_project_error_jax(
    top_error: jnp.ndarray,
    dfa_feedback: jnp.ndarray,
    act_pre: jnp.ndarray,
) -> jnp.ndarray:
    """Direct Feedback Alignment in JAX."""
    relu_mask = (act_pre > 0.0).astype(jnp.float32)
    return (top_error @ dfa_feedback.T) * relu_mask


@jax.jit
def oja_hebbian_step_jax(
    x: jnp.ndarray,
    w: jnp.ndarray,
    eta: float = 0.05,
) -> jnp.ndarray:
    """Oja's Hebbian update in JAX."""
    y = jnp.dot(x, w)
    return w + eta * (y * x - (y ** 2) * w)


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

    # 1. Forward-Forward
    h_pos = jnp.array(weights["h_pos"])
    h_neg = jnp.array(weights["h_neg"])
    golden_loss = jnp.array(weights["golden_ff_loss"])[0]
    golden_hn = jnp.array(weights["golden_h_norm"])

    ff_loss, h_norm = forward_forward_loss_and_norm_jax(h_pos, h_neg, theta=2.0)
    err_loss = float(jnp.abs(ff_loss - golden_loss))
    err_hn = float(jnp.max(jnp.abs(h_norm - golden_hn)))
    err_ff = max(err_loss, err_hn)
    assert err_ff < atol, f"JAX Forward-Forward failed: {err_ff:.2e}"
    errors["ff_err"] = err_ff

    # 2. Predictive Coding
    pc_x = jnp.array(weights["pc_x"])
    pc_r = jnp.array(weights["pc_r"])
    pc_w = jnp.array(weights["pc_w"])
    golden_pc = jnp.array(weights["golden_pc_r_next"])

    pc_next = predictive_coding_step_jax(pc_x, pc_r, pc_w, eta=0.1, lam=0.01)
    err_pc = float(jnp.max(jnp.abs(pc_next - golden_pc)))
    assert err_pc < atol, f"JAX Predictive Coding failed: {err_pc:.2e}"
    errors["pc_err"] = err_pc

    # 3. DFA
    top_err = jnp.array(weights["top_error"])
    dfa_fb = jnp.array(weights["dfa_feedback"])
    act_pre = jnp.array(weights["act_pre"])
    golden_dfa = jnp.array(weights["golden_dfa_grad"])

    dfa_grad = dfa_project_error_jax(top_err, dfa_fb, act_pre)
    err_dfa = float(jnp.max(jnp.abs(dfa_grad - golden_dfa)))
    assert err_dfa < atol, f"JAX DFA failed: {err_dfa:.2e}"
    errors["dfa_err"] = err_dfa

    # 4. Oja
    oja_x = jnp.array(weights["oja_x"])
    oja_w = jnp.array(weights["oja_w"])
    golden_oja = jnp.array(weights["golden_oja_w_next"])

    oja_next = oja_hebbian_step_jax(oja_x, oja_w, eta=0.05)
    err_oja = float(jnp.max(jnp.abs(oja_next - golden_oja)))
    assert err_oja < atol, f"JAX Oja failed: {err_oja:.2e}"
    errors["oja_err"] = err_oja

    # 5. Anchor
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
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.safetensors")
    errs = verify_invariants(p, 1e-5)
    print(f"JAX Invariant Parity Checks Passed: {errs}")
