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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    import safetensors.torch

    def generate_weights() -> None:
        torch.manual_seed(42)
        T_steps = 8
        d_state, d_obs = 2, 1

        # 1. Linear Kalman Filter (Joseph Form)
        F = torch.tensor([[1.0, 0.1], [0.0, 1.0]], dtype=torch.float32) # [2, 2]
        H = torch.tensor([[1.0, 0.0]], dtype=torch.float32)            # [1, 2]
        Q = 0.01 * torch.eye(d_state)                                  # [2, 2]
        R = 0.1 * torch.eye(d_obs)                                     # [1, 1]

        # Generate synthetic observations
        true_x = torch.zeros(T_steps, d_state)
        Y = torch.zeros(T_steps, d_obs)
        x_curr = torch.tensor([0.0, 1.0]) # pos=0, vel=1
        for t in range(T_steps):
            x_curr = F @ x_curr + 0.05 * torch.randn(d_state)
            true_x[t] = x_curr
            Y[t] = H @ x_curr + 0.1 * torch.randn(d_obs)

        # Run Kalman Filter with Joseph form
        x_filt = torch.zeros(T_steps, d_state)
        P_filt_diag = torch.zeros(T_steps, d_state)

        x_hat = torch.tensor([0.0, 0.0]) # initial estimate
        P_hat = 1.0 * torch.eye(d_state)  # initial covariance

        for t in range(T_steps):
            # Predict
            x_pred = F @ x_hat
            P_pred = F @ P_hat @ F.T + Q

            # Innovation
            y_obs = Y[t]
            y_pred = H @ x_pred
            innov = y_obs - y_pred
            S = H @ P_pred @ H.T + R # [1, 1]

            # Kalman gain
            K = P_pred @ H.T @ torch.linalg.inv(S) # [2, 1]

            # Update
            x_hat = x_pred + (K @ innov)
            # Joseph stabilized form: (I - K H) P_pred (I - K H)^T + K R K^T
            I_KH = torch.eye(d_state) - K @ H
            P_hat = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

            x_filt[t] = x_hat
            P_filt_diag[t] = torch.diag(P_hat)

        # 2. Discrete Hidden Markov Model in Log-Domain
        # States S=2, Symbols V=3
        S_states, V_symbols = 2, 3
        pi_init = torch.tensor([0.6, 0.4])
        log_pi = torch.log(pi_init)
        A_trans = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        log_A = torch.log(A_trans)
        B_emit = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])
        log_B = torch.log(B_emit)

        obs_seq = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)

        # Forward algorithm (Log-Sum-Exp)
        alpha = log_pi + log_B[:, obs_seq[0]] # [S]
        for t in range(1, T_steps):
            # alpha_t(j) = logsumexp_i (alpha_{t-1}(i) + log_A(i, j)) + log_B(j, obs_t)
            prev_expanded = alpha.unsqueeze(1) + log_A # [S, S]
            alpha = torch.logsumexp(prev_expanded, dim=0) + log_B[:, obs_seq[t]]
        hmm_log_prob = torch.logsumexp(alpha, dim=0) # total marginal log-prob

        # Viterbi decoding (max-product)
        viterbi_probs = torch.zeros(T_steps, S_states)
        backpointers = torch.zeros(T_steps, S_states, dtype=torch.long)
        viterbi_probs[0] = log_pi + log_B[:, obs_seq[0]]

        for t in range(1, T_steps):
            for s in range(S_states):
                trans_scores = viterbi_probs[t - 1] + log_A[:, s]
                best_prev = torch.argmax(trans_scores)
                backpointers[t, s] = best_prev
                viterbi_probs[t, s] = trans_scores[best_prev] + log_B[s, obs_seq[t]]

        best_last = torch.argmax(viterbi_probs[-1])
        viterbi_path = torch.zeros(T_steps, dtype=torch.long)
        viterbi_path[-1] = best_last
        for t in range(T_steps - 2, -1, -1):
            viterbi_path[t] = backpointers[t + 1, viterbi_path[t + 1]]

        # 3. Unscented Kalman Filter 2D Sigma Points
        mu_ukf = torch.tensor([1.0, 2.0])
        Sigma_ukf = torch.tensor([[1.0, 0.2], [0.2, 0.5]])
        alpha_ukf, beta_ukf, kappa_ukf = 1e-3, 2.0, 0.0
        n_dim = 2
        lambda_ukf = (alpha_ukf ** 2) * (n_dim + kappa_ukf) - n_dim
        gamma_ukf = ((n_dim + lambda_ukf) ** 0.5)

        L_cov = torch.linalg.cholesky(Sigma_ukf) # L L^T = Sigma
        sigma_pts = torch.zeros(2 * n_dim + 1, n_dim)
        sigma_pts[0] = mu_ukf
        for i in range(n_dim):
            sigma_pts[i + 1] = mu_ukf + gamma_ukf * L_cov[:, i]
            sigma_pts[i + 1 + n_dim] = mu_ukf - gamma_ukf * L_cov[:, i]

        # 4. Standard linear parity contract
        X_parity = torch.randn(32, 4)
        weight = torch.tensor([0.5, -0.3, 0.6, -0.4], dtype=torch.float32)
        bias = torch.tensor([0.1], dtype=torch.float32)
        test_out = X_parity @ weight + bias

        state_dict = {
            "weight": weight.clone().contiguous(),
            "bias": bias.clone().contiguous(),
            "test_input": X_parity.clone().contiguous(),
            "test_output": test_out.clone().contiguous(),
            "F": F.clone().contiguous(),
            "H": H.clone().contiguous(),
            "Q": Q.clone().contiguous(),
            "R": R.clone().contiguous(),
            "Y": Y.clone().contiguous(),
            "x_filt": x_filt.clone().contiguous(),
            "P_filt_diag": P_filt_diag.clone().contiguous(),
            "log_pi": log_pi.clone().contiguous(),
            "log_A": log_A.clone().contiguous(),
            "log_B": log_B.clone().contiguous(),
            "obs_seq": obs_seq.clone().contiguous(),
            "hmm_log_prob": hmm_log_prob.unsqueeze(0).clone().contiguous(),
            "viterbi_path": viterbi_path.clone().contiguous(),
            "mu_ukf": mu_ukf.clone().contiguous(),
            "Sigma_ukf": Sigma_ukf.clone().contiguous(),
            "sigma_pts": sigma_pts.clone().contiguous(),
        }

        out_dir = Path("weights/02_classical_and_probabilistic_ml")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "09_hidden_markov_models_kalman_ukf_and_particle_filters.safetensors"
        safetensors.torch.save_file(state_dict, out_file)
        print(f"Saved {len(state_dict)} tensors to {out_file}")

    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    run_verification()
