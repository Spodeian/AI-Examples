# 10. Survival Analysis and Contextual Bandits

## 1. Executive Summary
Time-to-event estimation and online decision-making under uncertainty represent two critical frontiers where traditional static supervised learning fails. In survival analysis, data is fundamentally right-censored: subjects may leave a clinical study or survive past the observation window without experiencing the event. Standard regression models that discard censored observations or treat censoring times as true event times introduce severe downward estimation bias. In contextual multi-armed bandits, algorithms must actively balance exploration and exploitation across high-dimensional feature spaces. This module implements the nonparametric Kaplan-Meier estimator (Kaplan & Meier, 1958), the semiparametric Cox Proportional Hazards model and DeepSurv (Cox, 1972; Katzman et al., 2018), and the Linear Upper Confidence Bound bandit algorithm (Li et al., 2010 LinUCB) accelerated by the Sherman-Morrison rank-1 matrix inversion lemma.

---

## 2. Mathematical Foundations

### 2.1 Nonparametric Kaplan-Meier Survival Estimator
Let $t_1 < t_2 < \dots < t_K$ be the distinct ordered event times among $N$ subjects. Let $d_i$ be the number of events occurring at time $t_i$, and $n_i$ be the number of individuals at risk immediately prior to $t_i$:
$$\hat{S}(t) = \prod_{t_i \le t} \left( 1 - \frac{d_i}{n_i} \right)$$
Greenwood's formula provides the asymptotic variance of $\hat{S}(t)$:
$$\widehat{\text{Var}}\{\hat{S}(t)\} = \hat{S}(t)^2 \sum_{t_i \le t} \frac{d_i}{n_i (n_i - d_i)}$$

### 2.2 Cox Proportional Hazards Model & DeepSurv
The hazard function models the instantaneous rate of event occurrence given covariates $\mathbf{x} \in \mathbb{R}^D$:
$$h(t | \mathbf{x}) = h_0(t) \exp(g_\theta(\mathbf{x}))$$
where $h_0(t)$ is an arbitrary baseline hazard and $g_\theta(\mathbf{x}) = \mathbf{x}^\top \boldsymbol{\beta}$ (or an MLP in DeepSurv).

**Breslow Negative Partial Log-Likelihood:**
The baseline hazard cancels out of the conditional probability that subject $i$ experiences the event at time $T_i$ among the risk set $\mathcal{R}(T_i) = \{j : T_j \ge T_i\}$:
$$\mathcal{L}_{\text{Cox}}(\theta) = -\sum_{i: E_i = 1} \left[ g_\theta(\mathbf{x}_i) - \ln \sum_{j \in \mathcal{R}(T_i)} \exp(g_\theta(\mathbf{x}_j)) \right]$$
By sorting event times in descending order ($T_{(1)} \ge T_{(2)} \ge \dots \ge T_{(N)}$), the risk set $\mathcal{R}(T_{(i)})$ corresponds exactly to the prefix $\{1, \dots, i\}$, permitting $O(N \log N)$ prefix LogCumSumExp evaluation rather than naive $O(N^2)$ nested loops.

### 2.3 Linear Upper Confidence Bound (LinUCB) Contextual Bandit
For each round $t$, a context $\mathbf{x}_t \in \mathbb{R}^D$ arrives. For each arm $a \in \{1, \dots, K\}$, assume expected reward $\mathbb{E}[r_{t, a} | \mathbf{x}_t] = \mathbf{x}_t^\top \boldsymbol{\theta}_a^*$.
Given historical ridge regression matrix $\mathbf{A}_a = \mathbf{X}_a^\top \mathbf{X}_a + \lambda \mathbf{I}$ and target vector $\mathbf{b}_a = \mathbf{X}_a^\top \mathbf{r}_a$:
$$\hat{\boldsymbol{\theta}}_a = \mathbf{A}_a^{-1} \mathbf{b}_a$$
The ridge prediction variance is $\sigma_{t, a}^2 = \mathbf{x}_t^\top \mathbf{A}_a^{-1} \mathbf{x}_t$. The LinUCB selection rule is:
$$a_t = \arg\max_{a} \left[ \mathbf{x}_t^\top \hat{\boldsymbol{\theta}}_a + \alpha \sqrt{\mathbf{x}_t^\top \mathbf{A}_a^{-1} \mathbf{x}_t} \right]$$

### 2.4 Sherman-Morrison $O(d^2)$ Rank-1 Matrix Inversion Update
Inverting $\mathbf{A}_a$ from scratch at every step incurs $\mathcal{O}(d^3)$ latency. Because $\mathbf{A}_{a, \text{new}} = \mathbf{A}_a + \mathbf{x}_t \mathbf{x}_t^\top$ is a rank-1 modification, the inverse is updated in $\mathcal{O}(d^2)$ operations:
$$\mathbf{A}_{a, \text{new}}^{-1} = \mathbf{A}_a^{-1} - \frac{\mathbf{A}_a^{-1} \mathbf{x}_t \mathbf{x}_t^\top \mathbf{A}_a^{-1}}{1 + \mathbf{x}_t^\top \mathbf{A}_a^{-1} \mathbf{x}_t}$$

---

## 3. Tensor Mechanics & Compute Graphs

```
[Context x_t: D] ────────────┬─────────────────────────────┐
                             ▼                             ▼
[Arms A_inv: K x D x D] ──> [Quad Form: x^T A_inv x]    [Mean: x^T (A_inv @ b)]
                             │                             │
                             ▼                             ▼
                        [Alpha * sqrt(Var)] ────────> [Sum: UCB Score] ──> [Argmax Arm: a*]
```

---

## 4. Multi-Ecosystem Implementation Parity

- **PyTorch 2.x (`pytorch2/`):** Vectorized `torch.logcumsumexp` Cox loss and tensorized LinUCB arm evaluation.
- **JAX (`jax/`):** Pure functional `jax.lax.scan` logcumsumexp and `jnp.einsum` batch quadratic form evaluation.
- **Native Rust Candle (`candle/`):** Allocation-free tensor operations with Gauss-Jordan inversion and analytic Sherman-Morrison updates.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader dispatching parallel quadratic form and dot product evaluations across 64 threads per workgroup.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Censoring Discard Bias** | Dropping right-censored patients from dataset | Survival probabilities $\hat{S}(t)$ severely underestimated | Apply Kaplan-Meier product-limit or Cox partial likelihood |
| **$O(N^2)$ Risk Set Bottleneck** | Materializing dense boolean pairwise mask for risk set | Quadratic memory and compute explosion on $N > 10^4$ | Sort times descending and compute prefix LogCumSumExp in $O(N \log N)$ |
| **$O(d^3)$ Online Inversion Latency** | Full matrix inversion at every bandit interaction | High per-step inference latency | Maintain inverse directly via Sherman-Morrison rank-1 update |
| **Bandit Exploration Collapse** | Setting exploration coefficient $\alpha = 0$ | Algorithm locks onto suboptimal arm permanently | Set $\alpha \ge 1.0$ or use Thompson Sampling |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Clinical Oncology & Drug Discovery:** Patient survival curves under immunotherapy trials (e.g. RECIST progression-free survival).
2. **Reliability Engineering & IoT:** Predicting Mean Time to Failure (MTTF) of jet engines and industrial wind turbines under censored telemetry logs.
3. **Personalized News & Ad Recommendation:** Yahoo! and news aggregator feed ranking using disjoint LinUCB contextual bandits.

---

## 7. Verification & Parity Results
All implementations satisfy the cross-track invariant contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path33"`: **PASS**
- `cargo nextest run -p ai-candle -E 'test(test_10)'`: **PASS**
- `cargo nextest run -p ai-kernel -E 'test(test_10)'`: **PASS**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [09_hidden_markov_models_kalman_ukf_and_particle_filters.md](09_hidden_markov_models_kalman_ukf_and_particle_filters.md)
- **Current Topic:** 10_survival_analysis_and_contextual_bandits.md
- **Next Topic:** [11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.md](11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.md)
- **Category Index:** [02_classical_and_probabilistic_ml/README.md](README.md)
