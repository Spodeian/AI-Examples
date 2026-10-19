"""Hidden Markov Models, Kalman Filters, UKF, and Particle Filters in PyTorch 2.x.

Features:
- Linear Kalman Filter with Joseph-stabilized covariance updates: P = (I - KH) P^- (I - KH)^T + K R K^T
- Discrete Hidden Markov Model with log-domain LogSumExp Forward inference and Viterbi decoding
- Unscented Kalman Filter (UKF) deterministic sigma-point generation
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch


def kalman_filter_forward(
    Y: torch.Tensor,
    F: torch.Tensor,
    H: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    x0: torch.Tensor,
    P0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs forward Linear Kalman Filter with Joseph stabilized covariance."""
    T_steps = Y.shape[0]
    d_state = F.shape[0]

    x_filt = torch.zeros(T_steps, d_state, device=Y.device, dtype=Y.dtype)
    P_filt_diag = torch.zeros(T_steps, d_state, device=Y.device, dtype=Y.dtype)

    x_hat = x0.clone()
    P_hat = P0.clone()

    I_mat = torch.eye(d_state, device=Y.device, dtype=Y.dtype)

    for t in range(T_steps):
        # Predict
        x_pred = F @ x_hat
        P_pred = F @ P_hat @ F.T + Q

        # Innovation
        y_t = Y[t]
        innov = y_t - (H @ x_pred)
        S = H @ P_pred @ H.T + R

        # Kalman gain: K = P_pred H^T S^{-1}
        K = P_pred @ H.T @ torch.linalg.inv(S)

        # Update state
        x_hat = x_pred + (K @ innov)

        # Joseph stabilized covariance update
        I_KH = I_mat - K @ H
        P_hat = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        x_filt[t] = x_hat
        P_filt_diag[t] = torch.diag(P_hat)

    return x_filt, P_filt_diag


def hmm_forward_log_prob(
    obs: torch.Tensor,
    log_pi: torch.Tensor,
    log_A: torch.Tensor,
    log_B: torch.Tensor,
) -> torch.Tensor:
    """Computes total observation sequence log likelihood using log-domain Forward algorithm."""
    T_steps = obs.shape[0]
    alpha = log_pi + log_B[:, obs[0]]

    for t in range(1, T_steps):
        prev_expanded = alpha.unsqueeze(1) + log_A
        alpha = torch.logsumexp(prev_expanded, dim=0) + log_B[:, obs[t]]

    return torch.logsumexp(alpha, dim=0)


def hmm_viterbi_decode(
    obs: torch.Tensor,
    log_pi: torch.Tensor,
    log_A: torch.Tensor,
    log_B: torch.Tensor,
) -> torch.Tensor:
    """Finds most probable hidden state path via Viterbi (max-product) algorithm."""
    T_steps = obs.shape[0]
    S_states = log_pi.shape[0]

    viterbi_probs = torch.zeros(T_steps, S_states, device=obs.device)
    backpointers = torch.zeros(T_steps, S_states, device=obs.device, dtype=torch.long)
    viterbi_probs[0] = log_pi + log_B[:, obs[0]]

    for t in range(1, T_steps):
        for s in range(S_states):
            trans_scores = viterbi_probs[t - 1] + log_A[:, s]
            best_prev = torch.argmax(trans_scores)
            backpointers[t, s] = best_prev
            viterbi_probs[t, s] = trans_scores[best_prev] + log_B[s, obs[t]]

    best_last = torch.argmax(viterbi_probs[-1])
    path = torch.zeros(T_steps, device=obs.device, dtype=torch.long)
    path[-1] = best_last
    for t in range(T_steps - 2, -1, -1):
        path[t] = backpointers[t + 1, path[t + 1]]

    return path


def generate_ukf_sigma_points(
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> torch.Tensor:
    """Generates 2n+1 Merwe scaled unscented sigma points."""
    n_dim = mu.shape[0]
    lambda_val = (alpha ** 2) * (n_dim + kappa) - n_dim
    gamma = ((n_dim + lambda_val) ** 0.5)

    L = torch.linalg.cholesky(Sigma)
    sigma_pts = torch.zeros(2 * n_dim + 1, n_dim, device=mu.device, dtype=mu.dtype)
    sigma_pts[0] = mu
    for i in range(n_dim):
        sigma_pts[i + 1] = mu + gamma * L[:, i]
        sigma_pts[i + 1 + n_dim] = mu - gamma * L[:, i]

    return sigma_pts


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    F = chk["F"]
    H = chk["H"]
    Q = chk["Q"]
    R = chk["R"]
    Y = chk["Y"]
    golden_x_filt = chk["x_filt"]
    golden_P_diag = chk["P_filt_diag"]

    # 1. Kalman Filter Parity
    x0 = torch.tensor([0.0, 0.0])
    P0 = 1.0 * torch.eye(2)
    x_filt, P_diag = kalman_filter_forward(Y, F, H, Q, R, x0, P0)
    diff_kf_x = torch.max(torch.abs(x_filt - golden_x_filt)).item()
    diff_kf_p = torch.max(torch.abs(P_diag - golden_P_diag)).item()
    assert diff_kf_x < 1e-5, f"Kalman filter state mismatch: {diff_kf_x:.2e}"
    assert diff_kf_p < 1e-5, f"Kalman filter cov diag mismatch: {diff_kf_p:.2e}"

    # 2. HMM Forward & Viterbi Parity
    log_pi = chk["log_pi"]
    log_A = chk["log_A"]
    log_B = chk["log_B"]
    obs_seq = chk["obs_seq"]
    golden_log_prob = chk["hmm_log_prob"][0].item()
    golden_viterbi = chk["viterbi_path"]

    hmm_log_prob = hmm_forward_log_prob(obs_seq, log_pi, log_A, log_B).item()
    assert abs(hmm_log_prob - golden_log_prob) < 1e-5, f"HMM log prob mismatch: {abs(hmm_log_prob - golden_log_prob):.2e}"

    viterbi_path = hmm_viterbi_decode(obs_seq, log_pi, log_A, log_B)
    assert torch.equal(viterbi_path, golden_viterbi), "HMM Viterbi path mismatch"

    # 3. UKF Sigma Points Parity
    mu_ukf = chk["mu_ukf"]
    Sigma_ukf = chk["Sigma_ukf"]
    golden_sigma_pts = chk["sigma_pts"]
    sigma_pts = generate_ukf_sigma_points(mu_ukf, Sigma_ukf)
    diff_ukf = torch.max(torch.abs(sigma_pts - golden_sigma_pts)).item()
    assert diff_ukf < 1e-5, f"UKF sigma points mismatch: {diff_ukf:.2e}"

    print("PyTorch Topic 09 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
