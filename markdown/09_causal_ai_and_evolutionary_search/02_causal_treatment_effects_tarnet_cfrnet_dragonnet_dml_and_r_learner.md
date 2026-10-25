# Causal Treatment Effects: TARNet, CFRNet, Dragonnet, Double Machine Learning (DML), and R-Learner

## 1. Mathematical Foundations & Lineage

Estimating heterogeneous treatment effects from observational data is a central challenge in precision medicine, algorithmic pricing, and policy evaluation. Unlike standard supervised learning, which predicts observed outcomes $Y$, causal inference requires estimating **counterfactual potential outcomes** $Y(1)$ and $Y(0)$ where at most one is ever observed for each unit (the Fundamental Problem of Causal Inference).

The conceptual lineage comprises:
1. **The Neyman-Rubin Potential Outcomes Framework (Rubin 1974)**: Defined the Individual Treatment Effect (ITE) $\tau_i = Y_i(1) - Y_i(0)$ and the Conditional Average Treatment Effect (CATE) $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]$.
2. **Treatment-Agnostic Representation Networks (TARNet — Shalit et al. 2017)**: Replaced naive joint feature concatenation $[X, T]$ with a shared feature trunk $\Phi(X)$ and treatment-specific prediction heads $h_0(\Phi(X))$ and $h_1(\Phi(X))$.
3. **Counterfactual Regression Networks (CFRNet — Shalit et al. 2017)**: Penalized the Integral Probability Metric (IPM) or Maximum Mean Discrepancy (MMD) distance between treated and control latent distributions to enforce covariate balance.
4. **Sufficiency for Causal Inference (Dragonnet — Shi et al. 2019)**: Proved that predicting propensity scores directly from $\Phi(X)$ guarantees that all confounding information required for the unconfoundedness assumption is preserved in the shared representation.
5. **Neyman Orthogonal Double Machine Learning (DML — Chernozhukov et al. 2018)**: Formulated residualized score functions where first-stage machine learning estimation errors have zero first-order impact on treatment effect estimates.

---

## 2. TARNet & CFRNet Representation Architectures

### TARNet Dual-Head Architecture
Under naive neural networks, concatenating treatment $T$ into input features $X$ often results in the network ignoring $T$ due to high dimensionality of $X$. TARNet routes samples through a shared trunk $\Phi(x) = \text{ReLU}(x \Phi_W + \Phi_b)$ into dedicated potential outcome heads:
$$Y_0(x) = h_0(\Phi(x)) = \Phi(x) h_{0, W} + h_{0, b}$$
$$Y_1(x) = h_1(\Phi(x)) = \Phi(x) h_{1, W} + h_{1, b}$$
$$\tau(x) = Y_1(x) - Y_0(x)$$

### CFRNet Latent Distribution Balancing (MMD Penalty)
When selection bias exists ($P(X \mid T=1) \neq P(X \mid T=0)$), outcome heads risk extrapolating outside their support. CFRNet adds a linear Maximum Mean Discrepancy penalty:
$$\mu_1 = \frac{1}{N_1} \sum_{i: T_i = 1} \Phi(x_i), \quad \mu_0 = \frac{1}{N_0} \sum_{i: T_i = 0} \Phi(x_i)$$
$$\text{MMD}^2 = \|\mu_1 - \mu_0\|_2^2$$
$$\mathcal{L}_{\text{CFRNet}} = \mathcal{L}_{\text{factual}} + \alpha \cdot \text{MMD}^2$$

---

## 3. Dragonnet: Propensity Score Sufficiency

By the Rosenbaum-Rubin theorem (1983), conditioning on the propensity score $e(x) = P(T=1 \mid X=x)$ is sufficient to remove confounding bias.

Dragonnet equips the representation $\Phi(x)$ with a third classification head:
$$\hat{e}(x) = \sigma(\Phi(x) w_{\text{prop}} + b_{\text{prop}})$$
$$\mathcal{L}_{\text{prop}} = -\frac{1}{N} \sum_{i=1}^N \left[ T_i \ln \hat{e}(x_i) + (1 - T_i) \ln(1 - \hat{e}(x_i)) \right]$$
Because the propensity loss backpropagates through $\Phi$, it prevents the representation from discarding features that are predictive of treatment assignment (confounders).

---

## 4. Double Machine Learning (DML) & The Robinson Transformation

Given the partially linear model:
$$Y = \tau(X) \cdot T + g(X) + \epsilon, \quad \mathbb{E}[\epsilon \mid X, T] = 0$$
$$T = m(X) + \eta, \quad \mathbb{E}[\eta \mid X] = 0$$
Taking conditional expectations yields the Robinson (1988) residualized equation:
$$Y - \mathbb{E}[Y \mid X] = \tau(X) \cdot (T - \mathbb{E}[T \mid X]) + \epsilon$$

### Neyman Orthogonality
Let $\tilde{Y} = Y - \hat{m}(X)$ and $\tilde{T} = T - \hat{e}(X)$, where $\hat{m}(X)$ and $\hat{e}(X)$ are arbitrary first-stage machine learning estimators (Random Forests, Gradient Boosted Trees, or Neural Networks).
The DML closed-form CATE estimator is:
$$\hat{\tau}_{\text{DML}} = \frac{\sum_{i=1}^N \tilde{T}_i \tilde{Y}_i}{\sum_{i=1}^N \tilde{T}_i^2}$$
Even if $\hat{m}$ and $\hat{e}$ converge at slow non-parametric rates $n^{-1/4}$, Neyman orthogonality ensures $\hat{\tau}_{\text{DML}}$ converges at the parametric root-$n$ rate $\sqrt{n}$ with asymptotic normality.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`tarnet_forward`** | `F.relu` trunk + dual heads | `jax.nn.relu` trunk + dual heads | Fused ReLU + linear outcome heads | Dual-head parallel evaluation |
| **`cfrnet_mmd_loss`** | Group mean difference squared | Masked tensor reductions | In-place group mean reduction | Linear MMD kernel |
| **`dragonnet_propensity_loss`** | Sigmoid + BCE loss | `jax.nn.sigmoid` + BCE | Stable clamped sigmoid BCE | Fused propensity BCE |
| **`dml_cate_estimate`** | Robinson residual ratio | Robinson residual ratio | In-place residual accumulator | `DML_RESIDUAL_WGSL` |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Positivity / Common Support Violation**:
   - *Symptom*: Propensity scores collapse to 0 or 1 ($e(x) \approx 0$ or $1$), causing variance explosion in IPTW and division by zero in DML denominators.
   - *Cause*: Deterministic treatment assignment policies (e.g., severe patients always receive surgery).
   - *Fix*: Enforce positivity trimming: restrict analysis to region of common support $\epsilon \le \hat{e}(x) \le 1 - \epsilon$ (e.g. $\epsilon = 0.05$).
2. **Representation Over-Balancing in CFRNet**:
   - *Symptom*: Outcome prediction accuracy plummets as representation MMD penalty weight $\alpha \to \infty$.
   - *Cause*: Enforcing $\Phi(X \mid T=1) \equiv \Phi(X \mid T=0)$ forces the representation to discard real predictive covariates that happen to be correlated with treatment.
   - *Fix*: Keep balance weight modest ($\alpha \le 0.1$) or transition to Dragonnet targeted regularization.
3. **Plug-in Non-Orthogonal Estimation Bias**:
   - *Symptom*: Estimating $\tau$ by taking the difference of two separately trained regression models $\hat{\mu}_1(x) - \hat{\mu}_0(x)$ without orthogonal residualization yields massive regularization bias.
   - *Cause*: Standard MSE regularization shrinks $\hat{\mu}_1$ and $\hat{\mu}_0$ independently toward zero, biasing their difference.
   - *Fix*: Always use DML orthogonal residualization or R-learner formulations.

---

## 7. Cross-Domain Generalisation Matrix

1. **Personalized Medicine & Clinical Pharmacology**: Identifies patient sub-populations with positive treatment response to targeted therapeutics while filtering out patients who experience adverse effects.
2. **Algorithmic Dynamic Pricing & Promotion Elasticity**: Estimates individual price elasticity $\tau(x) = \frac{\partial \text{Demand}}{\partial \text{Price}}$ across consumer demographic segments while adjusting for organic seasonal demand confounders.
3. **Online Recommendation Policy Optimization**: Estimates the true causal incremental lift (incrementality) of advertising impressions rather than taking credit for organic user conversions.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `y0_ref`, `y1_ref`, & `tau_ref`: TARNet dual potential outcomes and CATE effect.
3. `mmd_loss_ref`: CFRNet linear representation divergence penalty.
4. `propensity_ref` & `prop_loss_ref`: Dragonnet propensity score and binary cross-entropy.
5. `dml_tau_ref`: Double Machine Learning Neyman-orthogonal treatment effect estimate.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 9. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.py`](../../pytorch2/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.py)
- **JAX Reference**: [`jax/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.py`](../../jax/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.py)
- **Candle Reference**: [`candle/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.rs`](../../candle/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.rs)
- **Kernels**: [`kernels/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.rs`](../../kernels/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.rs)
- **Interactive Notebook**: [`notebooks/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.ipynb`](../../notebooks/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../INDEX_BY_GOAL.md)
