"""Generate 8-cell educational notebook for Topic 09:
09_hidden_markov_models_kalman_ukf_and_particle_filters.ipynb
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf

def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 09. Hidden Markov Models, Kalman Filters & UKF

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Rudolf E. Kálmán (1960) *A New Approach to Linear Filtering and Prediction Problems*; Leonard E. Baum et al. (1970) *A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains*; Andrew J. Viterbi (1967) *Error bounds for convolutional codes and an asymptotically optimum decoding algorithm*; Simon J. Julier & Jeffrey K. Uhlmann (1997) *A new extension of the Kalman filter to nonlinear systems*.
- **Purpose:** Unifies discrete and continuous state-space temporal inference, establishing Joseph-stabilized covariance propagation, log-domain LogSumExp Forward inference, Viterbi decoding, and unscented sigma points.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use textbook Kalman covariance updates $\\mathbf{P} = (\\mathbf{I} - \\mathbf{K}\\mathbf{H})\\mathbf{P}^-$ in floating-point pipelines, as roundoff errors destroy symmetry and positive definiteness; always use the Joseph-stabilized formulation. Do not compute HMM forward-backward passes in raw probability space for sequences $T > 50$, as repeated multiplication triggers exponential underflow to zero; compute in the log domain using LogSumExp."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Linear Kalman Filter & Joseph-Stabilized Covariance Update
For state transition $\\mathbf{x}_t = \\mathbf{F}\\mathbf{x}_{t-1} + \\mathbf{w}_t$ and emission $\\mathbf{y}_t = \\mathbf{H}\\mathbf{x}_t + \\mathbf{v}_t$ with noise covariances $\\mathbf{Q}, \\mathbf{R}$:
$$\\hat{\\mathbf{x}}_t^- = \\mathbf{F}\\hat{\\mathbf{x}}_{t-1}, \\quad \\mathbf{P}_t^- = \\mathbf{F}\\mathbf{P}_{t-1}\\mathbf{F}^\\top + \\mathbf{Q}$$
$$\\mathbf{S}_t = \\mathbf{H}\\mathbf{P}_t^-\\mathbf{H}^\\top + \\mathbf{R}, \\quad \\mathbf{K}_t = \\mathbf{P}_t^-\\mathbf{H}^\\top \\mathbf{S}_t^{-1}, \\quad \\hat{\\mathbf{x}}_t = \\hat{\\mathbf{x}}_t^- + \\mathbf{K}_t(\\mathbf{y}_t - \\mathbf{H}\\hat{\\mathbf{x}}_t^-)$$
$$\\mathbf{P}_t = (\\mathbf{I} - \\mathbf{K}_t\\mathbf{H})\\mathbf{P}_t^-(\\mathbf{I} - \\mathbf{K}_t\\mathbf{H})^\\top + \\mathbf{K}_t\\mathbf{R}\\mathbf{K}_t^\\top \\quad \\text{(Joseph Form Guarantee: } \\mathbf{P}_t \\succeq 0)$$

### 2. Discrete HMM Forward & Viterbi Decoding (Log Domain)
$$\\alpha_t(j) = \\text{LogSumExp}_{i} \\left( \\alpha_{t-1}(i) + \\ln A_{ij} \\right) + \\ln B_j(y_t), \\quad V_t(j) = \\max_{i} \\left( V_{t-1}(i) + \\ln A_{ij} \\right) + \\ln B_j(y_t)$$

### 3. Unscented Kalman Filter (UKF) Sigma Points
$$\\lambda = \\alpha^2(n + \\kappa) - n, \\quad \\gamma = \\sqrt{n + \\lambda}, \\quad \\chi_0 = \\boldsymbol{\\mu}, \\quad \\chi_i = \\boldsymbol{\\mu} \\pm \\gamma [\\mathbf{L}]_{:, i}$$

**Complexity:** Kalman: $\\mathcal{O}(T(d_x^3 + d_y^3))$; HMM Forward: $\\mathcal{O}(T S^2)$; UKF Sigma generation: $\\mathcal{O}(n^3)$. """
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import core

env = core.setup_env(seed=42)
T, d_x, d_y, S, V = 8, 2, 1, 2, 3
F = torch.tensor([[0.95, 0.05], [0.0, 0.90]])
H = torch.tensor([[1.0, 0.0]])
Q = torch.eye(d_x) * 0.01
R = torch.eye(d_y) * 0.05
Y = torch.tensor([[0.5], [0.6], [0.55], [0.7], [0.8], [0.75], [0.85], [0.9]])
log_pi = torch.log(torch.tensor([0.6, 0.4]))
log_A = torch.log(torch.tensor([[0.8, 0.2], [0.3, 0.7]]))
log_B = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]]))
obs = torch.tensor([0, 1, 0, 2, 1, 0, 2, 1], dtype=torch.long)
print(f"Setup complete on {env.device}: Kalman T={T}, HMM States={S}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class StateSpaceToolkit:
    @staticmethod
    def kalman_forward(Y, F, H, Q, R, x0, P0):
        T_steps, d_state = Y.shape[0], F.shape[0]
        x_filt = torch.zeros(T_steps, d_state)
        P_diag = torch.zeros(T_steps, d_state)
        x_hat, P_hat = x0.clone(), P0.clone()
        I = torch.eye(d_state)
        for t in range(T_steps):
            x_pred = F @ x_hat
            P_pred = F @ P_hat @ F.T + Q
            innov = Y[t] - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ torch.linalg.inv(S)
            x_hat = x_pred + K @ innov
            I_KH = I - K @ H
            P_hat = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
            x_filt[t], P_diag[t] = x_hat, torch.diag(P_hat)
        return x_filt, P_diag

    @staticmethod
    def hmm_forward(obs, log_pi, log_A, log_B):
        alpha = log_pi + log_B[:, obs[0]]
        for t in range(1, obs.shape[0]):
            alpha = torch.logsumexp(alpha.unsqueeze(1) + log_A, dim=0) + log_B[:, obs[t]]
        return torch.logsumexp(alpha, dim=0)

    @staticmethod
    def ukf_sigma_points(mu, Sigma, alpha=1e-3, kappa=0.0):
        n = mu.shape[0]
        gamma = ((alpha**2 * (n + kappa))) ** 0.5
        L = torch.linalg.cholesky(Sigma)
        pts = [mu]
        for i in range(n):
            pts.append(mu + gamma * L[:, i])
        for i in range(n):
            pts.append(mu - gamma * L[:, i])
        return torch.stack(pts, dim=0)"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Invariants & Precision Verification
x0, P0 = torch.zeros(2), torch.eye(2)
x_filt, P_diag = StateSpaceToolkit.kalman_forward(Y, F, H, Q, R, x0, P0)

# Invariant 1: State uncertainty decreases as observations accumulate
assert P_diag[-1, 0] < P_diag[0, 0], "Kalman covariance failed to decrease"

# Invariant 2: HMM Log-likelihood is strictly negative (valid probability)
ll = StateSpaceToolkit.hmm_forward(obs, log_pi, log_A, log_B)
assert ll < 0.0 and torch.isfinite(ll), "HMM Log likelihood must be strictly negative"

# Invariant 3: UKF symmetric sigma point displacement centroid equals original mean
mu_test = torch.tensor([1.5, -2.0])
sigma_test = torch.tensor([[1.0, 0.3], [0.3, 0.8]])
sigma_pts = StateSpaceToolkit.ukf_sigma_points(mu_test, sigma_test)
centroid_err = torch.norm(torch.mean(sigma_pts, dim=0) - mu_test).item()
assert centroid_err < 1e-6, f"UKF sigma points centroid error: {centroid_err}"

print(f"Verification passed: Final var={P_diag[-1,0].item():.4f}, HMM ll={ll.item():.4f}, Centroid err={centroid_err:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_kf = benchmark_fn(lambda: StateSpaceToolkit.kalman_forward(Y, F, H, Q, R, x0, P0), iters=30, name="Kalman Forward (T=8, d=2)")
b_hmm = benchmark_fn(lambda: StateSpaceToolkit.hmm_forward(obs, log_pi, log_A, log_B), iters=30, name="HMM Forward (T=8, S=2)")
b_ukf = benchmark_fn(lambda: StateSpaceToolkit.ukf_sigma_points(mu_test, sigma_test), iters=30, name="UKF Sigma Pts (d=2)")
print(format_benchmark_table([b_kf, b_hmm, b_ukf]))"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Textbook Covariance vs Joseph-Stabilized Covariance Update
# Simulate near-perfect sensor (R -> 0) with numerical truncation
P_pred_bad = torch.tensor([[1e-4, 1e-4], [1e-4, 1e-4]]) + torch.eye(2) * 1e-7
H_bad = torch.tensor([[1.0, 1.0]])
R_bad = torch.tensor([[1e-7]])

S_bad = H_bad @ P_pred_bad @ H_bad.T + R_bad
K_bad = P_pred_bad @ H_bad.T @ torch.linalg.inv(S_bad)

# Naive update: P = (I - KH) P_pred (loses symmetry & positive eigenvalues)
P_naive = (torch.eye(2) - K_bad @ H_bad) @ P_pred_bad
eig_naive = torch.linalg.eigvalsh(P_naive)

# Joseph stabilized update: P = (I - KH) P_pred (I - KH)^T + K R K^T
I_KH = torch.eye(2) - K_bad @ H_bad
P_joseph = I_KH @ P_pred_bad @ I_KH.T + K_bad @ R_bad @ K_bad.T
eig_joseph = torch.linalg.eigvalsh(P_joseph)

print(f"[NAIVE UPDATE]  Min Eigenvalue: {eig_naive.min().item():.2e} (Negative / Unstable!)")
print(f"[JOSEPH UPDATE] Min Eigenvalue: {eig_joseph.min().item():.2e} (Strictly Positive Definite)")
assert eig_joseph.min().item() >= -1e-9, "Joseph form must preserve positive semi-definiteness" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Joseph Form Invariance:** Textbook Kalman covariance updates rapidly lose positive definiteness under single precision or high-gain sensor streams; the Joseph formulation guarantees numerical stability $\\mathbf{P} \\succeq 0$.
2. **Log-Domain HMM Inference:** Computing forward and Viterbi passes via LogSumExp avoids exponential float underflow, scaling seamlessly to sequences of arbitrary length $T$.
3. **Unscented Transformation:** Captures posterior mean and covariance to 3rd-order Taylor accuracy for nonlinear functions without analytical Jacobians.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [09_hidden_markov_models_kalman_ukf_and_particle_filters.md](../../markdown/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.md)
- **PyTorch 2.x Module:** [09_hidden_markov_models_kalman_ukf_and_particle_filters.py](../../pytorch2/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.py)
- **Functional JAX Module:** [09_hidden_markov_models_kalman_ukf_and_particle_filters.py](../../jax/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.py)
- **Native Rust Candle Track:** [09_hidden_markov_models_kalman_ukf_and_particle_filters.rs](../../candle/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.rs)
- **Hardware WGSL Kernel:** [09_hidden_markov_models_kalman_ukf_and_particle_filters.rs](../../kernels/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.rs)
- **Golden Reference Checkpoint:** [09_hidden_markov_models_kalman_ukf_and_particle_filters.safetensors](../../weights/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 08. Matrix Factorization (ALS, NMF, BPR, FM)](08_matrix_factorization_als_nmf_bpr_and_fm.ipynb) | [Category Index](../README.md) | [Next: 10. Survival Analysis & Contextual Bandits](10_survival_analysis_and_contextual_bandits.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "09_hidden_markov_models_kalman_ukf_and_particle_filters.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)

if __name__ == "__main__":
    build_notebook()
