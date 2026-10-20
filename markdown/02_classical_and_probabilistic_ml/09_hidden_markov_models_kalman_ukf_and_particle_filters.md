# 09. Hidden Markov Models, Kalman Filters, UKF, and Particle Filters

## 1. Executive Summary
Dynamic state-space estimation models infer hidden physical or probabilistic trajectories from noisy, partially observed temporal sensor streams. Classical models partition into discrete-state systems (Hidden Markov Models; Baum et al., 1970; Viterbi, 1967) and continuous Gaussian state-space systems (Kalman, 1960; Julier & Uhlmann, 1997 Unscented Kalman Filter). Standard recursive Kalman filtering $P = (I - KH)P^-$ is notoriously prone to loss of positive definiteness under roundoff error. This module provides a complete mathematical treatment of Joseph-stabilized covariance propagation, log-domain LogSumExp Forward-Backward inference, dynamic programming Viterbi decoding, and Merwe scaled unscented transform sigma points, synchronized across PyTorch 2.x, JAX, native Rust Candle, and hardware kernels.

---

## 2. Mathematical Foundations

### 2.1 Linear Kalman Filter with Joseph-Stabilized Covariance Update
Consider a linear state-space dynamical system:
$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$
$$\mathbf{y}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$$

**Predict Step:**
$$\hat{\mathbf{x}}_t^- = \mathbf{F} \hat{\mathbf{x}}_{t-1}, \quad \mathbf{P}_t^- = \mathbf{F} \mathbf{P}_{t-1} \mathbf{F}^\top + \mathbf{Q}$$

**Innovation & Kalman Gain:**
$$\tilde{\mathbf{y}}_t = \mathbf{y}_t - \mathbf{H} \hat{\mathbf{x}}_t^-, \quad \mathbf{S}_t = \mathbf{H} \mathbf{P}_t^- \mathbf{H}^\top + \mathbf{R}, \quad \mathbf{K}_t = \mathbf{P}_t^- \mathbf{H}^\top \mathbf{S}_t^{-1}$$

**State Update:**
$$\hat{\mathbf{x}}_t = \hat{\mathbf{x}}_t^- + \mathbf{K}_t \tilde{\mathbf{y}}_t$$

**Joseph Stabilized Covariance Formulation:**
The textbook update $\mathbf{P}_t = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_t^-$ is not guaranteed to remain symmetric positive semi-definite due to floating-point truncation. Peter Joseph's stabilized formulation guarantees $P \succeq 0$:
$$\mathbf{P}_t = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_t^- (\mathbf{I} - \mathbf{K}_t \mathbf{H})^\top + \mathbf{K}_t \mathbf{R} \mathbf{K}_t^\top$$

### 2.2 Discrete Hidden Markov Models (HMM) in the Log Domain
For discrete hidden states $z_t \in \{1, \dots, S\}$ and observations $y_t \in \{1, \dots, V\}$, with transition matrix $\mathbf{A} \in \mathbb{R}^{S \times S}$ and emission matrix $\mathbf{B} \in \mathbb{R}^{S \times V}$:
1. **Forward Log-Likelihood:** Recursive forward messages $\alpha_t(j) = \ln P(y_1, \dots, y_t, z_t = j)$:
   $$\alpha_t(j) = \text{LogSumExp}_{i=1}^S \left( \alpha_{t-1}(i) + \ln A_{ij} \right) + \ln B_j(y_t)$$
2. **Viterbi Decoding (Max-Product):** Finds the single most probable hidden trajectory $z_{1:T}^*$:
   $$V_t(j) = \max_{i} \left( V_{t-1}(i) + \ln A_{ij} \right) + \ln B_j(y_t), \quad \psi_t(j) = \arg\max_i \left( V_{t-1}(i) + \ln A_{ij} \right)$$

### 2.3 Unscented Kalman Filter (UKF) Merwe Scaled Sigma Points
To propagate Gaussian distributions through non-linear dynamics without analytical Jacobians, the Unscented Transform selects $2n + 1$ deterministic sigma points $\chi$:
$$\lambda = \alpha^2 (n + \kappa) - n, \quad \gamma = \sqrt{n + \lambda}$$
$$\chi_0 = \boldsymbol{\mu}$$
$$\chi_i = \boldsymbol{\mu} + \gamma [\sqrt{\boldsymbol{\Sigma}}]_{:, i}, \quad \chi_{n+i} = \boldsymbol{\mu} - \gamma [\sqrt{\boldsymbol{\Sigma}}]_{:, i} \quad (i = 1, \dots, n)$$
where $\sqrt{\boldsymbol{\Sigma}} = \mathbf{L}$ is the lower-triangular Cholesky factor ($\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top$).

---

## 3. Tensor Mechanics & Compute Graphs

```
[Observation Stream Y: T x d_obs]
       │
       ├──> [Predict: x_pred = F @ x, P_pred = F @ P @ F.T + Q]
       │       │
       │       ▼
       ├──> [Innovation: innov = y_t - H @ x_pred, S = H @ P_pred @ H.T + R]
       │       │
       │       ▼
       ├──> [Gain: K = P_pred @ H.T @ inv(S)]
       │       │
       │       ▼
       └──> [Joseph Stabilized Update: P = (I - KH) P_pred (I - KH)^T + K R K^T] ──> [Filtered Trajectory]
```

---

## 4. Multi-Ecosystem Implementation Parity

- **PyTorch 2.x (`pytorch2/`):** Batch tensor forward pass with Joseph covariance updates and vectorized LogSumExp HMM inference.
- **JAX (`jax/`):** Pure functional scan-based execution (`jax.lax.scan`) compiling Kalman filter recursions into optimized XLA loops.
- **Native Rust Candle (`candle/`):** Allocation-conscious tensor operations on CPU with exact analytic matrix inversion.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader parallelizing UKF sigma point displacement over workgroups of 64 threads.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Covariance Asymmetry & Non-PSD** | Truncation error in textbook $(I - KH)P^-$ | Cholesky factorization failure / negative variances | Always execute Joseph-stabilized formulation |
| **Probability Underflow in HMM** | Multiplying raw likelihood probabilities over long sequences | Numerical underflow to 0.0 at $t > 50$ | Operate strictly in log-domain using LogSumExp |
| **Sigma Point Weight Collapse** | Ill-chosen UKF scaling parameters $\kappa < 0$ or large $\alpha$ | Negative covariance matrix predictions | Choose $\alpha = 10^{-3}$, $\beta = 2$, $\kappa = 0$ |
| **Particle Depletion in Sequential MC** | Repeated resampling concentrating on single high-weight particle | Severe particle impoverishment / variance loss | Monitor Effective Sample Size (ESS) and trigger MCMC rejuvenation |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Robotics & Autonomous Navigation:** SLAM (Simultaneous Localization and Mapping) pipelines rely on Extended/Unscented Kalman Filters to fuse GPS, IMU, and LiDAR odometry.
2. **Computational Biology & Bioinformatics:** HMMs serve as the core engine in gene prediction (GENSCAN), protein sequence domain matching (Pfam, HMMER), and chromatin segmentation (ChromHMM).
3. **Quantitative Finance:** Regime-switching HMMs identify Bull/Bear market transitions from high-frequency order book and volatility dynamics.

---

## 7. Verification & Parity Results
All implementations satisfy the cross-track invariant contract:
$$\|x_{\text{PyTorch}} - x_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|x_{\text{Candle}} - x_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path32"`: **PASS**
- `cargo nextest run -p ai-candle -E 'test(test_09)'`: **PASS**
- `cargo nextest run -p ai-kernel -E 'test(test_09)'`: **PASS**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [08_matrix_factorization_als_nmf_bpr_and_fm.md](08_matrix_factorization_als_nmf_bpr_and_fm.md)
- **Current Topic:** 09_hidden_markov_models_kalman_ukf_and_particle_filters.md
- **Next Topic:** [10_survival_analysis_and_contextual_bandits.md](10_survival_analysis_and_contextual_bandits.md)
- **Category Index:** [Master Curriculum Index](../../README.md)
