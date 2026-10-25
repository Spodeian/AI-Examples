"""Formal Neural Verification IBP and CROWN Bound Propagation — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation. Run ``python jax/11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def ibp_linear(
    l_in: jnp.ndarray,
    u_in: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interval Bound Propagation for linear layer y = x @ W + b."""
    mu = (u_in + l_in) * 0.5
    r = (u_in - l_in) * 0.5
    mu_out = mu @ weight + bias
    r_out = r @ jnp.abs(weight)
    return mu_out - r_out, mu_out + r_out


def ibp_relu(l_in: jnp.ndarray, u_in: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interval Bound Propagation for ReLU activation."""
    return jnp.clip(l_in, a_min=0.0), jnp.clip(u_in, a_min=0.0)


def crown_relu_relaxation(
    l_in: jnp.ndarray,
    u_in: jnp.ndarray,
    alpha: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Upper and lower linear relaxations for ReLU on [l, u]:
    s_L * x + b_L <= ReLU(x) <= s_U * x + b_U
    """
    diff = jnp.clip(u_in - l_in, a_min=1e-8)
    
    s_u_crossing = u_in / diff
    b_u_crossing = -(u_in * l_in) / diff
    
    if alpha is None:
        s_l_crossing = jnp.where(u_in > jnp.abs(l_in), 1.0, 0.0)
    else:
        s_l_crossing = jnp.clip(alpha, 0.0, 1.0)
    b_l_crossing = jnp.zeros_like(l_in)
    
    is_active = (l_in >= 0.0).astype(jnp.float32)
    is_inactive = (u_in <= 0.0).astype(jnp.float32)
    is_crossing = (1.0 - is_active) * (1.0 - is_inactive)
    
    s_L = is_active * 1.0 + is_crossing * s_l_crossing
    b_L = is_crossing * b_l_crossing
    s_U = is_active * 1.0 + is_crossing * s_u_crossing
    b_U = is_crossing * b_u_crossing
    
    return s_L, b_L, s_U, b_U


def crown_2layer_bounds(
    x_l: jnp.ndarray,
    x_u: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
    c: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """CROWN backward linear bound propagation for 2-layer MLP."""
    z1_l, z1_u = ibp_linear(x_l, x_u, w1, b1)
    s_L, b_L, s_U, b_U = crown_relu_relaxation(z1_l, z1_u)
    
    v = w2 @ c  # [H]
    
    lambda_l = jnp.where(v >= 0.0, s_L, s_U) * v
    bias_relu_l = jnp.sum(jnp.where(v >= 0.0, b_L, b_U) * v, axis=-1)
    
    a_l = lambda_l @ w1.T
    gamma_l = jnp.sum(lambda_l * b1, axis=-1) + bias_relu_l + jnp.dot(b2, c)
    
    mu_x = (x_u + x_l) * 0.5
    r_x = (x_u - x_l) * 0.5
    crown_lower = jnp.sum(a_l * mu_x, axis=-1) - jnp.sum(jnp.abs(a_l) * r_x, axis=-1) + gamma_l
    
    lambda_u = jnp.where(v >= 0.0, s_U, s_L) * v
    bias_relu_u = jnp.sum(jnp.where(v >= 0.0, b_U, b_L) * v, axis=-1)
    
    a_u = lambda_u @ w1.T
    gamma_u = jnp.sum(lambda_u * b1, axis=-1) + bias_relu_u + jnp.dot(b2, c)
    
    crown_upper = jnp.sum(a_u * mu_x, axis=-1) + jnp.sum(jnp.abs(a_u) * r_x, axis=-1) + gamma_u
    
    return crown_lower, crown_upper


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. IBP Linear & ReLU
    z1_l, z1_u = ibp_linear(g["x_l"], g["x_u"], g["w1"], g["b1"])
    assert float(jnp.max(jnp.abs(z1_l - g["z1_l_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(z1_u - g["z1_u_ref"]))) < 1e-5
    
    a1_l, a1_u = ibp_relu(z1_l, z1_u)
    assert float(jnp.max(jnp.abs(a1_l - g["a1_l_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(a1_u - g["a1_u_ref"]))) < 1e-5
    
    # 3. CROWN ReLU relaxation
    s_L, b_L, s_U, b_U = crown_relu_relaxation(z1_l, z1_u)
    assert float(jnp.max(jnp.abs(s_L - g["s_L_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(b_L - g["b_L_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(s_U - g["s_U_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(b_U - g["b_U_ref"]))) < 1e-5
    
    # 4. CROWN bounds
    crown_l, crown_u = crown_2layer_bounds(g["x_l"], g["x_u"], g["w1"], g["b1"], g["w2"], g["b2"], g["c"])
    assert float(jnp.max(jnp.abs(crown_l - g["crown_lower_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(crown_u - g["crown_upper_ref"]))) < 1e-5
    
    print("11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
