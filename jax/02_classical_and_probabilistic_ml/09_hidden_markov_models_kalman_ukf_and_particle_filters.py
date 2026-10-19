"""Hidden Markov Models, Kalman Filters, UKF, and Particle Filters in JAX.

Features:
- Functional Linear Kalman Filter with Joseph-stabilized covariance updates
- Discrete Hidden Markov Model with log-domain LogSumExp Forward inference and Viterbi decoding
- Unscented Kalman Filter (UKF) deterministic sigma-point generation
"""

from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
import safetensors.torch
import numpy as np


@jax.jit
def kalman_filter_forward(
    Y: jnp.ndarray,
    F: jnp.ndarray,
    H: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    x0: jnp.ndarray,
    P0: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs forward Linear Kalman Filter with Joseph stabilized covariance using jax.lax.scan."""
    d_state = F.shape[0]
    I_mat = jnp.eye(d_state, dtype=Y.dtype)

    def step_fn(carry, y_t):
        x_hat, P_hat = carry

        # Predict
        x_pred = F @ x_hat
        P_pred = F @ P_hat @ F.T + Q

        # Innovation
        innov = y_t - (H @ x_pred)
        S = H @ P_pred @ H.T + R

        # Kalman gain: K = P_pred H^T S^{-1}
        K = P_pred @ H.T @ jnp.linalg.inv(S)

        # Update state
        x_new = x_pred + (K @ innov)

        # Joseph stabilized covariance update: P = (I - KH) P_pred (I - KH)^T + K R K^T
        I_KH = I_mat - K @ H
        P_new = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        return (x_new, P_new), (x_new, jnp.diag(P_new))

    _, (x_filt, P_filt_diag) = jax.lax.scan(step_fn, (x0, P0), Y)
    return x_filt, P_filt_diag


@jax.jit
def hmm_forward_log_prob(
    obs: jnp.ndarray,
    log_pi: jnp.ndarray,
    log_A: jnp.ndarray,
    log_B: jnp.ndarray,
) -> jnp.ndarray:
    """Computes total observation sequence log likelihood using log-domain Forward algorithm."""
    alpha_0 = log_pi + log_B[:, obs[0]]

    def step_fn(alpha, obs_t):
        prev_expanded = jnp.expand_dims(alpha, axis=1) + log_A
        alpha_next = jax.nn.logsumexp(prev_expanded, axis=0) + log_B[:, obs_t]
        return alpha_next, None

    alpha_final, _ = jax.lax.scan(step_fn, alpha_0, obs[1:])
    return jax.nn.logsumexp(alpha_final, axis=0)


def hmm_viterbi_decode(
    obs: jnp.ndarray,
    log_pi: jnp.ndarray,
    log_A: jnp.ndarray,
    log_B: jnp.ndarray,
) -> jnp.ndarray:
    """Finds most probable hidden state path via Viterbi (max-product) algorithm."""
    T_steps = obs.shape[0]
    viterbi_probs = [log_pi + log_B[:, obs[0]]]
    backpointers = []

    for t in range(1, T_steps):
        trans_scores = jnp.expand_dims(viterbi_probs[-1], 1) + log_A
        best_prev = jnp.argmax(trans_scores, axis=0)
        max_trans = jnp.max(trans_scores, axis=0)
        v_next = max_trans + log_B[:, obs[t]]
        backpointers.append(best_prev)
        viterbi_probs.append(v_next)

    best_last = int(jnp.argmax(viterbi_probs[-1]))
    path = [best_last]
    for bp in reversed(backpointers):
        path.append(int(bp[path[-1]]))
    path.reverse()
    return jnp.array(path, dtype=jnp.int32)


@jax.jit
def generate_ukf_sigma_points(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> jnp.ndarray:
    """Generates 2n+1 Merwe scaled unscented sigma points."""
    n_dim = mu.shape[0]
    lambda_val = (alpha ** 2) * (n_dim + kappa) - n_dim
    gamma = ((n_dim + lambda_val) ** 0.5)

    L = jnp.linalg.cholesky(Sigma)
    pts = [mu]
    for i in range(n_dim):
        pts.append(mu + gamma * L[:, i])
    for i in range(n_dim):
        pts.append(mu - gamma * L[:, i])

    return jnp.stack(pts, axis=0)


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    F = jnp.array(chk["F"].numpy())
    H = jnp.array(chk["H"].numpy())
    Q = jnp.array(chk["Q"].numpy())
    R = jnp.array(chk["R"].numpy())
    Y = jnp.array(chk["Y"].numpy())
    golden_x_filt = chk["x_filt"].numpy()
    golden_P_diag = chk["P_filt_diag"].numpy()

    # 1. Kalman Filter Parity
    x0 = jnp.array([0.0, 0.0], dtype=jnp.float32)
    P0 = 1.0 * jnp.eye(2, dtype=jnp.float32)
    x_filt, P_diag = kalman_filter_forward(Y, F, H, Q, R, x0, P0)
    diff_kf_x = float(np.max(np.abs(np.array(x_filt) - golden_x_filt)))
    diff_kf_p = float(np.max(np.abs(np.array(P_diag) - golden_P_diag)))
    assert diff_kf_x < 1e-5, f"JAX Kalman filter state mismatch: {diff_kf_x:.2e}"
    assert diff_kf_p < 1e-5, f"JAX Kalman filter cov diag mismatch: {diff_kf_p:.2e}"

    # 2. HMM Forward & Viterbi Parity
    log_pi = jnp.array(chk["log_pi"].numpy())
    log_A = jnp.array(chk["log_A"].numpy())
    log_B = jnp.array(chk["log_B"].numpy())
    obs_seq = jnp.array(chk["obs_seq"].numpy())
    golden_log_prob = float(chk["hmm_log_prob"][0].numpy())
    golden_viterbi = chk["viterbi_path"].numpy()

    hmm_log_prob = float(hmm_forward_log_prob(obs_seq, log_pi, log_A, log_B))
    assert abs(hmm_log_prob - golden_log_prob) < 1e-5, f"JAX HMM log prob mismatch: {abs(hmm_log_prob - golden_log_prob):.2e}"

    viterbi_path = np.array(hmm_viterbi_decode(obs_seq, log_pi, log_A, log_B))
    assert np.array_equal(viterbi_path, golden_viterbi), "JAX HMM Viterbi path mismatch"

    # 3. UKF Sigma Points Parity
    mu_ukf = jnp.array(chk["mu_ukf"].numpy())
    Sigma_ukf = jnp.array(chk["Sigma_ukf"].numpy())
    golden_sigma_pts = chk["sigma_pts"].numpy()
    sigma_pts = generate_ukf_sigma_points(mu_ukf, Sigma_ukf)
    diff_ukf = float(np.max(np.abs(np.array(sigma_pts) - golden_sigma_pts)))
    assert diff_ukf < 1e-5, f"JAX UKF sigma points mismatch: {diff_ukf:.2e}"

    print("JAX Topic 09 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
