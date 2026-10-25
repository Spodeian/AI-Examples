# Causal Discovery: NOTEARS Continuous Optimization, SCMs, and Instrumental Variables

## 1. Mathematical Foundations & Lineage

Inferring cause-and-effect relationships from observational data is the cornerstone of robust, generalisable artificial intelligence. While purely correlational deep learning models suffer from shortcut learning, distribution shifts, and spurious associations, causal models explicitly distinguish between observational conditional distributions $P(Y \mid X)$ and interventional distributions $P(Y \mid do(X))$.

The conceptual lineage comprises:
1. **Pearl's Structural Causal Models (SCMs) & The Ladder of Causation (Pearl 2009)**: Formalized the three levels of cognitive hierarchy:
   - *Association ($L_1$)*: $P(Y \mid X)$ — Seeing (what is?).
   - *Intervention ($L_2$)*: $P(Y \mid do(X))$ — Doing (what if we act?).
   - *Counterfactuals ($L_3$)*: $P(Y_{X=x'} \mid X=x, Y=y)$ — Imagining (why, and what if we had acted differently?).
2. **Combinatorial to Continuous DAG Learning (NOTEARS — Zheng et al. 2018)**: Transformed the combinatorial NP-hard problem of finding a Directed Acyclic Graph (DAG) over $d$ variables into a smooth, equality-constrained continuous optimization problem via a matrix exponential acyclicity characterization.
3. **Instrumental Variables & Two-Stage Least Squares (2SLS — Angrist & Imbens 1995; Hartford et al. 2017 DeepIV)**: Resolved unobserved confounding bias between treatment $T$ and outcome $Y$ by exploiting an exogenous instrument $Z$.

---

## 2. Continuous DAG Structure Learning via NOTEARS

Traditional causal discovery algorithms (PC algorithm, GES) perform discrete combinatorial search over the super-exponential space of DAGs ($\mathcal{O}(d! \cdot 2^{\binom{d}{2}})$).

Zheng et al. (2018) proved that a weighted adjacency matrix $W \in \mathbb{R}^{d \times d}$ is acyclic if and only if:
$$h(W) = \text{Tr}\left( e^{W \odot W} \right) - d = 0$$
where $\odot$ denotes elementwise Hadamard multiplication, and $e^M = \sum_{k=0}^\infty \frac{M^k}{k!}$ is the matrix exponential.

### Gradient of the Acyclicity Constraint
The analytic gradient of $h(W)$ with respect to $W$ is:
$$\nabla h(W) = 2 \cdot \left( e^{W \odot W} \right)^T \odot W$$
Combining a least-squares data-fitting loss $F(W) = \frac{1}{2n} \|X - XW\|_F^2$ with sparsity penalty $\lambda \|W\|_1$ under the Augmented Lagrangian framework transforms discrete graph search into smooth continuous optimization:
$$\min_{W} F(W) + \lambda \|W\|_1 + \alpha h(W) + \frac{\rho}{2} h(W)^2$$

---

## 3. Structural Causal Models (SCMs) & The 3-Step Counterfactual Engine

In a linear SCM with exogenous noise variables $U \in \mathbb{R}^{1 \times d}$ and strictly lower-triangular DAG weight matrix $W \in \mathbb{R}^{d \times d}$:
$$X = XW + U \iff X(I - W) = U \iff X = U(I - W)^{-1}$$
Because $W$ is strictly triangular (nilpotent of degree $d$), the inverse is exact and polynomial: $(I - W)^{-1} = \sum_{k=0}^{d-1} W^k$.

### Pearl's 3 Steps of Counterfactual Reasoning
Given factual observation $x$:
1. **Abduction**: Infer the unobserved exogenous noise that generated the factual reality:
   $$u = x(I - W)$$
2. **Action**: Apply intervention $do(X_j = c)$ by mutilating the causal graph—severing all incoming arrows to node $j$ ($W_{\text{intervened}}[:, j] = 0$) and fixing $u_j = c$.
3. **Prediction**: Compute counterfactual consequences under the modified mechanics:
   $$x_{\text{counterfactual}} = u_{\text{intervened}} (I - W_{\text{intervened}})^{-1}$$

---

## 4. Instrumental Variables & Two-Stage Least Squares (2SLS)

When an unobserved confounder $U$ simultaneously affects treatment $T$ and outcome $Y$, ordinary least squares (OLS) regression estimates a biased coefficient: $\hat{\beta}_{\text{OLS}} = \beta_{\text{true}} + \frac{\text{Cov}(U, T)}{\text{Var}(T)}$.

A valid instrument $Z$ satisfies:
1. **Relevance**: $\text{Cov}(Z, T) \neq 0$ (strongly predicts treatment).
2. **Exogeneity**: $\text{Cov}(Z, U) = 0$ (uncorrelated with unobserved confounders).
3. **Exclusion Restriction**: $Z \to Y$ only through $T$.

### 2SLS Formulation
- **Stage 1 (Exogenous Projection)**:
  $$\hat{\Gamma} = (Z^T Z)^{-1} Z^T T, \quad \hat{T} = Z \hat{\Gamma} = P_Z T$$
- **Stage 2 (Causal Estimation)**:
  $$\hat{\beta}_{2\text{SLS}} = (\hat{T}^T \hat{T})^{-1} \hat{T}^T Y$$
Because $\hat{T}$ only contains variation induced by exogenous instrument $Z$, it is purged of confounding bias, yielding consistent causal estimates.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`notears_acyclicity_constraint`** | `torch.matrix_exp(W * W)` | `jax.scipy.linalg.expm(W * W)` | Taylor polynomial $e^M$ | `NOTEARS_GRAD_WGSL` |
| **`scm_generate_and_counterfactual`** | `u @ torch.inverse(I - W)` | `u @ jnp.linalg.inv(I - W)` | Exact nilpotent $(I - W)^{-1}$ | Vectorized DAG propagation |
| **`two_stage_least_squares`** | Closed-form $P_Z T$ & OLS | JAX `jnp.linalg.inv` 2SLS | 2x2 analytic matrix inversion | Parallel least-squares solver |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Conditioning on a Collider (Berkson's Fallacy)**:
   - *Symptom*: Conditioning on variable $C$ in graph $X \to C \leftarrow Y$ creates spurious negative correlation between independent causes $X$ and $Y$.
   - *Cause*: Colliders open active non-causal paths when conditioned upon.
   - *Fix*: Never condition on colliders or descendants of colliders during back-door adjustment.
2. **Weak Instrument Pathologies in 2SLS**:
   - *Symptom*: 2SLS standard errors explode to infinity and point estimates become more biased than naive OLS.
   - *Cause*: Instrument correlation with treatment is too weak (first-stage $F$-statistic $< 10$).
   - *Fix*: Verify first-stage instrument strength ($F = \frac{R^2 / k}{(1 - R^2) / (n - k - 1)} > 10$).
3. **NOTEARS Bad Local Minima from Cyclic Initialization**:
   - *Symptom*: Optimization converges to a graph containing cycles with $h(W) > 10^{-4}$.
   - *Cause*: Penalty parameter $\rho$ increased too quickly before primal convergence.
   - *Fix*: Follow standard Augmented Lagrangian scheduling: double $\rho$ only when $h(W_k) > 0.25 h(W_{k-1})$.

---

## 7. Cross-Domain Generalisation Matrix

1. **Gene Regulatory Networks (GRNs)**: NOTEARS discovers directed gene activation/inhibition pathways from single-cell RNA-seq perturbations without combinatorial explosion.
2. **Clinical Pharmacology & Epidemiology**: Mendelian Randomization uses genetic variants as natural instrumental variables to isolate drug treatment effects from unobserved patient lifestyle confounders.
3. **Autonomous Robotics Sensor Dependencies**: Identifies which sensor failures directly cause actuation errors versus sensors that merely reflect correlated physical vibrations.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `h_dag_ref` & `grad_h_ref`: NOTEARS acyclicity constraint and analytic gradient.
3. `scm_x_factual_ref` & `counterfactual_x_ref`: SCM forward factual reality and $do(X_1 = 3.0)$ counterfactual intervention.
4. `t_hat_ref` & `beta_2sls_ref`: Two-stage least squares instrument projection and causal coefficient.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 9. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.py`](../../pytorch2/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.py)
- **JAX Reference**: [`jax/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.py`](../../jax/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.py)
- **Candle Reference**: [`candle/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.rs`](../../candle/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.rs)
- **Kernels**: [`kernels/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.rs`](../../kernels/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.rs)
- **Interactive Notebook**: [`notebooks/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.ipynb`](../../notebooks/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../INDEX_BY_GOAL.md)
