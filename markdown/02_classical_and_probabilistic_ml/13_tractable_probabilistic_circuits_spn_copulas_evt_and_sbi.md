# 13. Tractable Probabilistic Circuits (SPN), Copulas, Extreme Value Theory (EVT), and SBI

## 1. Executive Summary
Modern probabilistic modeling operates between two conflicting extremes: expressiveness and tractability. While deep generative models (diffusion, autoregressive, GANs) offer expressive density estimation, computing exact marginals $p(X_O)$ or conditionals $p(X_Q | X_O)$ requires expensive approximate MCMC or variational sampling, which is often intractable (#P-hard). **Tractable Probabilistic Circuits (TPCs)** and **Sum-Product Networks (SPNs)** resolve this dilemma by constraining computational graphs to satisfy structural **smoothness** and **decomposability**, enabling exact marginal, conditional, and MPE inference in a single forward pass with linear time complexity $\mathcal{O}(|E|)$. In multi-asset financial risk and geophysical disaster modeling, **Sklar's Copula Theorem** separates marginal distributions from dependency structures, exposing the fatal flaw of Gaussian copulas (asymptotic tail independence $\lambda_U = \lambda_L \equiv 0$) and substituting Archimedean/extreme copulas. Complementing copulas in the deep tail, **Extreme Value Theory (EVT)** formalizes the Generalized Pareto Distribution (GPD) for Peaks-Over-Threshold excesses. Finally, when forward physics simulators lack analytical likelihoods $p(x|\theta)$, **Simulation-Based Inference (SBI)** leverages neural density ratio and posterior estimation to unlock Bayesian parameter estimation over black-box code. This module delivers mathematically unified, cross-framework implementations across PyTorch 2.x, JAX, native Rust Candle, and hardware kernels.

---

## 2. Mathematical Foundations

### 2.1 Tractable Probabilistic Circuits & Sum-Product Networks (SPNs)
An SPN is a rooted directed acyclic graph (DAG) composed of sum nodes ($\bigoplus$), product nodes ($\bigotimes$), and tractable univariate leaf distributions:
$$S(\mathbf{x}) = \sum_{c \in \text{ch}(S)} w_{Sc} C_c(\mathbf{x}), \quad P(\mathbf{x}) = \prod_{c \in \text{ch}(P)} C_c(\mathbf{x})$$

Exact tractability in $\mathcal{O}(|E|)$ time is guaranteed by two structural properties:
1. **Decomposability (Consistency):** For every product node $P$, its children have pairwise disjoint variable scopes:
   $$\text{scope}(C_i) \cap \text{scope}(C_j) = \emptyset, \quad \forall C_i, C_j \in \text{ch}(P), i \ne j$$
   *Consequence:* Integration factors over independent variables: $\int \prod_i C_i(\mathbf{x}_i) d\mathbf{x} = \prod_i \int C_i(\mathbf{x}_i) d\mathbf{x}_i$.
2. **Smoothness (Completeness):** For every sum node $S$, all of its children share identical variable scopes:
   $$\text{scope}(C_i) = \text{scope}(C_j) = \text{scope}(S), \quad \forall C_i, C_j \in \text{ch}(S)$$
   *Consequence:* Summation commutes with marginalization: $\int \sum_i w_i C_i(\mathbf{x}) d\mathbf{x} = \sum_i w_i \int C_i(\mathbf{x}) d\mathbf{x}$.

To compute an arbitrary marginal $p(X_O)$, all unobserved variables $X_M$ are marginalized out by setting their leaf evaluation to $1.0$ (or integrating out leaves), followed by a single bottom-up evaluation pass.

### 2.2 Sklar's Theorem & Copula Tail Dependence
Sklar's Theorem states that any multivariate joint cumulative distribution function (CDF) $F(x_1, \dots, x_d)$ can be decoupled into its continuous marginal CDFs $F_i(x_i)$ and a copula $C: [0, 1]^d \to [0, 1]$:
$$F(x_1, \dots, x_d) = C\left(F_1(x_1), \dots, F_d(x_d)\right)$$

#### The Gaussian Copula Fallacy:
The Gaussian copula exhibits **asymptotic tail independence**:
$$\lambda_L = \lim_{u \to 0^+} P(U_2 \le u | U_1 \le u) = 0, \quad \lambda_U = \lim_{u \to 1^-} P(U_2 > u | U_1 > u) = 0 \quad (\text{for } \rho < 1)$$
This assumption led to catastrophic underestimation of joint defaults in the 2008 global financial crisis.

#### Archimedean Clayton Copula:
The Clayton copula exhibits strong lower tail dependence:
$$C_\theta(u, v) = \max\left(u^{-\theta} + v^{-\theta} - 1, 0\right)^{-1/\theta}, \quad \lambda_L = 2^{-1/\theta} > 0$$

### 2.3 Extreme Value Theory (EVT) & Peaks-Over-Threshold
By the Pickands-Balkema-de Haan Theorem, the distribution of excesses $Y = X - u$ over a high threshold $u$, given $X > u$, asymptotically converges to the **Generalized Pareto Distribution (GPD)**:
$$G_{\xi, \sigma}(y) = P(X - u \le y | X > u) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}$$
- $\xi > 0$: Heavy-tailed (Fréchet type; Pareto polynomial decay).
- $\xi = 0$: Exponential decay (Gumbel type; light-tailed).
- $\xi < 0$: Upper bound (Weibull type; finite endpoint).

Value at Risk ($\text{VaR}_\alpha$) and Expected Shortfall ($\text{ES}_\alpha$) in the extreme tail evaluate in closed form:
$$\text{VaR}_\alpha = u + \frac{\sigma}{\xi}\left[\left(\frac{N}{N_u}(1 - \alpha)\right)^{-\xi} - 1\right]$$

### 2.4 Simulation-Based Inference (SBI)
When the likelihood $p(x | \theta)$ is defined only implicitly via a stochastic simulation algorithm (agent-based models, high-energy particle physics, cosmological N-body codes), SBI estimates parameter posteriors by training neural networks on simulated pairs $(\theta, x) \sim p(\theta) p(x | \theta)$:
1. **Neural Likelihood Ratio Estimation (SNRE):** A classifier distinguishes joint pairs $(\theta, x) \sim p(\theta, x)$ from marginal pairs $(\theta, x) \sim p(\theta)p(x)$:
   $$\sigma(\text{logit}(x, \theta)) = \frac{p(x, \theta)}{p(x, \theta) + p(x)p(\theta)} \implies \exp(\text{logit}(x, \theta)) \approx \frac{p(x | \theta)}{p(x)}$$

---

## 3. Tensor Mechanics & Compute Graphs

```
[Evidence Mask e_i in {0, 1, -1}] ──> [Leaf Evaluation: p(x_i) or 1.0 if unobserved]
                                           │
                                    [Product Layer: Disjoint Scopes (Decomposability)]
                                           │
                                    [Sum Layer: Weighted Sum (Smoothness)]
                                           │
                                           └──> [Exact Linear-Time Marginal p(X_O)]

[Uniform Marginals u, v in [0, 1]] ──> [Clayton Generator: (u^-th + v^-th - 1)^(-1/th)] ──> [Lower Tail Crash Dependence]

[Threshold Excesses y >= 0] ─────────> [GPD Tail CDF: 1 - (1 + xi*y/sigma)^(-1/xi)] ────> [Extreme Quantiles / VaR]

[Simulated Pairs (x, theta)] ────────> [Concat [x, theta]] ──> [MLP Logit] ──> [exp(logit) = Density Ratio r(x, theta)]
```

---

## 4. Multi-Ecosystem Implementation Parity

- **PyTorch 2.x (`pytorch2/`):** Tensorized SPN evaluation with evidence masking, vectorized Clayton copula, GPD CDF, and SBI neural ratio classifier.
- **JAX (`jax/`):** Pure functional JIT-compiled SPN marginalizer and vectorized Archimedean copula engine.
- **Native Rust Candle (`candle/`):** Zero-allocation matrix-vector neural ratio evaluation and exact SPN leaf-to-root reduction.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader parallelizing Clayton copula and GPD excess calculations across millions of points.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **Gaussian Copula Crash Blindness** | Assuming asymptotic tail independence ($\lambda_L \equiv 0$) | Systemic failure to model simultaneous defaults; 100x crash underestimation | Use Clayton ($\lambda_L = 2^{-1/\theta}$) or Student-$t$ copulas for joint tail risk |
| **SPN Scope Overlap (#P-Hardness)** | Product node combines non-disjoint scopes ($\text{scope}(C_1) \cap \text{scope}(C_2) \ne \emptyset$) | Product layer does not factorize under integration; marginals become #P-hard | Verify DAG structural decomposability at graph compilation time |
| **GPD Threshold Bias-Variance Dilemma** | Threshold $u$ chosen too low or too high | Too low: non-GPD bias; Too high: extreme variance due to sparse sample count | Use Mean Residual Life (MRL) plot and parameter stability plots across $u$ |
| **SBI Ratio Saturation** | Classifier overfits to synthetic simulation batch | Logits explode, density ratio $r(x, \theta) \to \infty$ or $0.0$ | Apply gradient penalties, weight decay, and sequential round targeting (SNRE-B) |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Quantitative Finance & Portfolio Stress-Testing:** Clayton and Student-$t$ copulas accurately capture multi-asset liquidity crashes during market-wide contagion events.
2. **Climate Science & Extreme Hydrology:** Peaks-Over-Threshold GPD models 100-year flood levels and hurricane storm surges beyond historical record limits.
3. **Astrophysics & Biophysics:** Simulation-Based Inference calibrates black-box cosmological dark-matter simulations and protein folding kinetic rates where analytic likelihoods do not exist.

---

## 7. Verification & Parity Results
All implementations satisfy the cross-track invariant contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path36"`: **PASS**
- `cargo nextest run -p ai-candle -E 'test(test_13)'`: **PASS**
- `cargo nextest run -p ai-kernel -E 'test(test_13)'`: **PASS**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.md](12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.md)
- **Current Topic:** 13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.md
- **Next Category:** [03. Building Blocks & Losses](../03_building_blocks_and_losses/)
- **Category Index:** [Master Curriculum Index](../../README.md)
