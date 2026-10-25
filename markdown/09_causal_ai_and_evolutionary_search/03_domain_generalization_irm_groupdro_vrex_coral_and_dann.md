# Domain Generalization: Invariant Risk Minimization (IRM), GroupDRO, V-REx, Deep CORAL, and DANN

## 1. Mathematical Foundations & Lineage

Standard Empirical Risk Minimization (ERM) minimizes the average loss over the training distribution $\min_\theta \mathbb{E}_{(x, y) \sim P_{\text{train}}} [\mathcal{L}(f_\theta(x), y)]$. When test distributions shift ($P_{\text{test}} \neq P_{\text{train}}$) due to spurious correlations, environmental artifacts, or demographic disparities, ERM models fail catastrophically because they exploit non-causal shortcuts that correlate with targets in training environments but decorrelate in target environments.

The conceptual lineage comprises:
1. **Domain-Adversarial Neural Networks (DANN — Ganin et al. 2016)**: Introduced the Gradient Reversal Layer (GRL) to align source and target latent feature distributions by training a domain discriminator adversarially.
2. **Deep CORAL (Correlation Alignment — Sun & Saenko 2016)**: Aligned second-order statistics (covariance matrices) of latent representations across source and target domains via Frobenius distance.
3. **Invariant Risk Minimization (IRMv1 — Arjovsky et al. 2019)**: Formulated causality-grounded domain generalization: a feature representation $\Phi(x)$ is invariant across environments $e \in \mathcal{E}$ if and only if the optimal linear classifier on top of $\Phi(x)$ is simultaneously optimal in every environment.
4. **Group Distributionally Robust Optimization (GroupDRO — Sagawa et al. 2020)**: Minimized worst-case risk across predefined demographic or environmental groups using online exponentiated gradient updates.
5. **Variance Risk Extrapolation (V-REx — Krueger et al. 2021)**: Penalized the variance of empirical risks across training environments to extrapolate robust performance beyond the convex hull of training environments.

---

## 2. Invariant Risk Minimization (IRMv1) Formulation

IRM seeks a data representation $\Phi: \mathcal{X} \to \mathcal{H}$ such that the optimal linear classifier $w: \mathcal{H} \to \mathcal{Y}$ is invariant across all training environments $e \in \mathcal{E}_{\text{tr}}$:
$$\min_{\Phi, w} \sum_{e \in \mathcal{E}_{\text{tr}}} R^e(w \circ \Phi) \quad \text{s.t.} \quad w \in \arg\min_{\bar{w}} R^e(\bar{w} \circ \Phi) \quad \forall e \in \mathcal{E}_{\text{tr}}$$

### The IRMv1 Surrogate Penalty
Because bilevel optimization over all linear classifiers is intractable, Arjovsky et al. (2019) fixed a scalar dummy classifier $w = 1.0$ and penalized deviations from the first-order stationarity condition $\nabla_{w \mid w=1.0} R^e(w \cdot \Phi) = 0$:
$$\mathcal{L}_{\text{IRMv1}} = \sum_{e \in \mathcal{E}_{\text{tr}}} R^e(\Phi) + \lambda \sum_{e \in \mathcal{E}_{\text{tr}}} \left\| \nabla_{w \mid w=1.0} R^e(w \cdot \Phi) \right\|_2^2$$
For binary classification with BCE loss, the gradient evaluates to:
$$\nabla_{w \mid w=1.0} R^e(w \cdot \Phi) = \frac{1}{N_e} \sum_{i=1}^{N_e} z_i^e \cdot (\sigma(z_i^e) - y_i^e)$$
where $z_i^e = \Phi(x_i^e)$ is the logit.

---

## 3. V-REx & Group Distributionally Robust Optimization (GroupDRO)

### Variance Risk Extrapolation (V-REx)
V-REx penalizes the variance of empirical risks across environments:
$$\mathcal{L}_{\text{V-REx}} = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \mathcal{L}_e + \beta \cdot \text{Var}_e(\mathcal{L}_e) = \bar{\mathcal{L}} + \frac{\beta}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} (\mathcal{L}_e - \bar{\mathcal{L}})^2$$
Penalizing variance forces the model to balance performance across all environments, preventing the model from exploiting shortcuts present in high-frequency environments.

### GroupDRO Exponentiated Gradient Weights
GroupDRO maintains a probability distribution $q \in \Delta^{|\mathcal{E}|}$ over environments:
$$\mathcal{L}_{\text{GroupDRO}} = \sum_{e \in \mathcal{E}} q_e \mathcal{L}_e$$
In each step, group weights update via exponentiated gradient ascent with step size $\eta$:
$$\tilde{q}_e^{(t+1)} = q_e^{(t)} \exp(\eta \mathcal{L}_e^{(t)}), \quad q_e^{(t+1)} = \frac{\tilde{q}_e^{(t+1)}}{\sum_k \tilde{q}_k^{(t+1)}}$$
This continuously shifts optimization attention to whichever group currently suffers the highest error.

---

## 4. Second-Order Alignment: Deep CORAL

Deep CORAL aligns the covariance matrices of source and target feature representations without requiring adversarial training:
$$C_s = \frac{1}{N_s - 1} (\Phi_s - \bar{\Phi}_s)^T (\Phi_s - \bar{\Phi}_s), \quad C_t = \frac{1}{N_t - 1} (\Phi_t - \bar{\Phi}_t)^T (\Phi_t - \bar{\Phi}_t)$$
$$\mathcal{L}_{\text{CORAL}} = \frac{1}{4 d^2} \|C_s - C_t\|_F^2 = \frac{1}{4 d^2} \sum_{i=1}^d \sum_{j=1}^d (C_{s, i, j} - C_{t, i, j})^2$$
By aligning second-order covariances, CORAL preserves the internal correlation structure while eliminating cross-domain feature stretching and rotation.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`irm_v1_penalty`** | Mean of `z * (sigmoid(z) - y)` squared | JAX vector reduction | In-place first-order stationarity gradient | Vectorized stationary gradient |
| **`vrex_variance_penalty`** | `mean((losses - mean)^2)` | `jnp.mean((losses - mean)^2)` | Mean-centered variance | Parallel environment variance |
| **`group_dro_update`** | Exponentiated softmax update | `jnp.exp(eta * L)` normalization | In-place exponentiated gradient | Fused GroupDRO reduction |
| **`deep_coral_loss`** | Centered covariance Frobenius norm | Centered covariance Frobenius norm | Centered Gram matrix difference | `CORAL_DIFF_WGSL` |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **IRMv1 Linear Collapse under Finite Samples (Rosenfeld et al. 2020)**:
   - *Symptom*: IRMv1 fails to discover invariant features and performs no better than ERM on Colored MNIST when sample sizes are finite ($N < 10^5$).
   - *Cause*: The first-order gradient penalty $\|\nabla_{w=1} R^e\|^2$ permits spurious solutions where gradient vectors cancel out without achieving true invariance across the entire function space.
   - *Fix*: Hybridize IRM with GroupDRO or V-REx variance penalties.
2. **Extreme GroupDRO Instability under Noisy Labels**:
   - *Symptom*: GroupDRO weights $q_e$ concentrate 100% on a single minority group containing label noise, corrupting the entire representation.
   - *Cause*: Outliers with massive loss dominate the exponentiated gradient update.
   - *Fix*: Clip environment losses $\mathcal{L}_e \le M$ and add entropy regularization on $q$.
3. **Negative Transfer in Adversarial Alignment (DANN)**:
   - *Symptom*: Target domain accuracy collapses when source and target label distributions are asymmetric ($P_s(Y) \neq P_t(Y)$).
   - *Cause*: Forcing $\Phi_s \equiv \Phi_t$ misaligns classes when class proportions differ across domains.
   - *Fix*: Condition domain discriminator on predicted class probabilities (Conditional Adversarial Domain Adaptation - CDAN).

---

## 7. Cross-Domain Generalisation Matrix

1. **Medical Imaging Diagnostics (CAMELYON17 / Wilds)**: Aligns histopathology representations across diverse hospital scanner hardware artifacts and staining variations.
2. **Autonomous Vehicle Perception**: Maintains pedestrian detection accuracy across clear daylight, rain, fog, and night driving environments without retraining.
3. **Dialect-Robust NLP**: GroupDRO ensures toxicity and intent classifiers maintain equal accuracy across African American English (AAE) and Standard American English.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `irm_penalty_ref`: IRMv1 first-order stationary gradient penalty.
3. `vrex_penalty_ref`: V-REx cross-environment risk variance.
4. `dro_loss_ref` & `q_updated_ref`: GroupDRO worst-case risk and exponentiated weight vector.
5. `coral_loss_ref`: Deep CORAL second-order covariance alignment loss.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 9. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.py`](../../pytorch2/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.py)
- **JAX Reference**: [`jax/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.py`](../../jax/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.py)
- **Candle Reference**: [`candle/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.rs`](../../candle/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.rs)
- **Kernels**: [`kernels/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.rs`](../../kernels/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.rs)
- **Interactive Notebook**: [`notebooks/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.ipynb`](../../notebooks/09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../INDEX_BY_GOAL.md)
