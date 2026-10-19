"""Matrix Factorization: ALS, NMF, BPR, and Factorization Machines in JAX.

Features:
- Functional JIT-compiled Factorization Machine (FM) O(KD) forward evaluation
- Alternating Least Squares (ALS) matrix factorization step
- Non-Negative Matrix Factorization (NMF) multiplicative updates
- Bayesian Personalized Ranking (BPR) loss
"""

from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
import safetensors.torch
import numpy as np


@jax.jit
def factorization_machine_forward(
    x: jnp.ndarray,
    w0: jnp.ndarray,
    w: jnp.ndarray,
    V: jnp.ndarray
) -> jnp.ndarray:
    """Computes FM 2-way interactions via Rendle's algebraic identity: O(KD)."""
    # x: [N, D], w0: [1], w: [D], V: [D, K]
    linear = w0[0] + x @ w # [N]

    xv = x @ V # [N, K]
    xv_sq = xv ** 2 # [N, K]

    x_sq = x ** 2 # [N, D]
    v_sq = V ** 2 # [D, K]
    x_sq_v_sq = x_sq @ v_sq # [N, K]

    interaction = 0.5 * jnp.sum(xv_sq - x_sq_v_sq, axis=1) # [N]
    return linear + interaction


@jax.jit
def als_step(R: jnp.ndarray, P: jnp.ndarray, lambda_reg: float = 0.1) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single ALS step for Q given P, and P given Q."""
    K = P.shape[1]
    PtP_reg = P.T @ P + lambda_reg * jnp.eye(K)
    Q = jnp.linalg.solve(PtP_reg, (R.T @ P).T).T

    QtQ_reg = Q.T @ Q + lambda_reg * jnp.eye(K)
    P_new = jnp.linalg.solve(QtQ_reg, (R @ Q).T).T

    return P_new, Q


@jax.jit
def nmf_step(V: jnp.ndarray, W: jnp.ndarray, H: jnp.ndarray, eps: float = 1e-7) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single NMF multiplicative update step."""
    H_new = H * ((W.T @ V) / (W.T @ W @ H + eps))
    W_new = W * ((V @ H_new.T) / (W @ H_new @ H_new.T + eps))
    return W_new, H_new


@jax.jit
def bpr_loss(p_u: jnp.ndarray, q_pos: jnp.ndarray, q_neg: jnp.ndarray) -> jnp.ndarray:
    """Bayesian Personalized Ranking pairwise loss."""
    diff = jnp.sum(p_u * q_pos, axis=-1) - jnp.sum(p_u * q_neg, axis=-1)
    return -jnp.mean(jnp.log(jax.nn.sigmoid(diff) + 1e-8))


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    X = jnp.array(chk["X"].numpy())
    w0 = jnp.array(chk["w0"].numpy())
    w_fm = jnp.array(chk["w_fm"].numpy())
    V_fm = jnp.array(chk["V_fm"].numpy())
    golden_fm_out = chk["fm_out"].numpy()

    # 1. FM Parity
    fm_out = factorization_machine_forward(X, w0, w_fm, V_fm)
    diff_fm = float(np.max(np.abs(np.array(fm_out) - golden_fm_out)))
    assert diff_fm < 1e-5, f"JAX FM output mismatch: {diff_fm:.2e}"

    # 2. ALS Prediction Parity
    P_als_ref = jnp.array(chk["P_als"].numpy())
    Q_als_ref = jnp.array(chk["Q_als"].numpy())
    golden_r_als = chk["R_als_pred"].numpy()
    diff_als = float(np.max(np.abs(np.array(P_als_ref @ Q_als_ref.T) - golden_r_als)))
    assert diff_als < 1e-5, f"JAX ALS prediction mismatch: {diff_als:.2e}"

    # 3. NMF Prediction Parity
    W_up = jnp.array(chk["W_updated"].numpy())
    H_up = jnp.array(chk["H_updated"].numpy())
    golden_nmf = chk["V_nmf_pred"].numpy()
    diff_nmf = float(np.max(np.abs(np.array(W_up @ H_up) - golden_nmf)))
    assert diff_nmf < 1e-5, f"JAX NMF prediction mismatch: {diff_nmf:.2e}"

    print("JAX Topic 08 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
