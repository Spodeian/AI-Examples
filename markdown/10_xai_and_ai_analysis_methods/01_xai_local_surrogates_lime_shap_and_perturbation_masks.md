# XAI Local Surrogates: LIME, KernelSHAP, and Perturbation Masks

## 1. Mathematical Foundations & Lineage

Black-box complex neural networks $f: \mathcal{X} \to \mathcal{Y}$ lack direct internal interpretability. Local surrogate methods approximate complex decision boundaries locally around an input instance $x$ by perturbing features, observing changes in the model's output, and fitting an interpretable surrogate model.

The lineage of local explanation methods comprises:
1. **Shapley Values (Shapley 1953)**: Founded in cooperative game theory to allocate total payoff fairly among $M$ cooperating players based on marginal contributions across all $2^M$ possible coalitions.
2. **LIME (Local Interpretable Model-agnostic Explanations — Ribeiro et al. 2016)**: Approximated non-linear decision boundaries locally around instance $x$ using distance-weighted linear ridge regression over binary perturbation masks.
3. **KernelSHAP (Lundberg & Lee 2017)**: Proved that choosing a specific weighting kernel $\pi_{\text{SHAP}}(z')$ in weighted linear regression guarantees that the linear surrogate coefficients are identical to exact Shapley values.
4. **RISE & Extremal Masks (Petsiuk et al. 2018; Fong & Vedaldi 2017)**: Formulated saliency via randomized input masking and smooth optimization over perturbation masks.
5. **ROAR (Remove and Retrain — Hooker et al. 2019)**: Introduced rigorous evaluation of explanation faithfulness by retraining models on datasets with top-attributed features masked out, proving that many cosmetic saliency maps perform no better than random edge detectors.

---

## 2. LIME Local Weighted Surrogate Formulation

LIME explains prediction $f(x)$ by learning an interpretable model $g \in \mathcal{G}$ (such as a linear model $g(z') = w^T z'$) that minimizes local weighted squared loss:
$$\arg\min_{w} \sum_{i=1}^N \pi_x(z_i) \left( f(z_i) - w^T z'_i \right)^2 + \lambda \|w\|_2^2$$
where:
- $z'_i \in \{0, 1\}^M$ is a binary perturbation mask indicating presence/absence of interpretable components,
- $\pi_x(z_i) = \exp\left( -D(x, z_i)^2 / \sigma^2 \right)$ is an exponential proximity kernel measuring distance from the original query $x$.

### Closed-Form Weighted Ridge Regression
Defining weight matrix $\Pi = \text{diag}(\pi_1, \dots, \pi_N)$, the optimal surrogate attribution vector is:
$$w^* = \left( Z'^T \Pi Z' + \lambda I \right)^{-1} Z'^T \Pi \mathbf{y}$$

---

## 3. KernelSHAP Combinatorial Formulation

Shapley values allocate credit to feature $i$ across all possible subsets $S \subseteq \mathcal{M} \setminus \{i\}$:
$$\phi_i(f, x) = \sum_{S \subseteq \mathcal{M} \setminus \{i\}} \frac{|S|! (M - |S| - 1)!}{M!} \left( f(S \cup \{i\}) - f(S) \right)$$
Computing exact Shapley values requires $2^M$ model evaluations, which is computationally intractable for large $M$.

### The Shapley Kernel
Lundberg & Lee (2017) proved that solving weighted linear regression $\min_\phi \sum_{z'} \pi_{\text{SHAP}}(z') (f(z') - \phi^T z')^2$ recovers exact Shapley values if and only if the regression kernel is set to:
$$\pi_{\text{SHAP}}(z') = \frac{M - 1}{\binom{M}{|z'|} |z'| (M - |z'|)}$$
where $|z'|$ is the coalition size (number of active features). This kernel heavily weights very small coalitions ($|z'| \approx 1$, isolating individual features) and very large coalitions ($|z'| \approx M - 1$, isolating single-feature removals).

---

## 4. RISE & ROAR Explanation Faithfulness

### Randomized Input Sampling for Explanation (RISE)
RISE computes pixel/feature saliency by probing the model with $N$ random binary masks $m_k \sim \text{Bernoulli}(p)^M$:
$$S_i = \frac{1}{N \cdot p} \sum_{k=1}^N f(x \odot m_k) \cdot m_{k, i}$$
Features that consistently produce high model confidence when present receive high attribution scores.

### Faithfulness Auditing (ROAR & Pixel-Flipping Curves)
An explanation is *faithful* if removing features identified as important causes rapid degradation in model accuracy:
$$\Delta y_k = y_{\text{orig}} - f\left( x \odot (1 - \text{mask}_{\text{top-}k}) \right)$$
Plotting $\Delta y_k$ as a function of $k$ produces the degradation curve. A faithful attribution produces a monotonically increasing degradation curve, whereas cosmetic artifacts show zero degradation.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`lime_weighted_regression`** | `torch.linalg.solve` on weighted Gram | `jnp.linalg.solve` on weighted Gram | Gauss-Jordan linear solver | Vectorized Gram accumulator |
| **`shapley_kernel_weights`** | Combinatorial binomial scaling | Functional subset weight mapping | Combinatorial binomial mapping | Parallel kernel evaluator |
| **`rise_feature_importance`** | `(y.unsqueeze(0) @ masks) / N` | Matrix-vector contraction | In-place matmul affine scale | `RISE_IMPORTANCE_WGSL` |
| **`faithfulness_degradation`** | Difference vector $y_0 - y_k$ | Elementwise subtraction | Vectorized broadcast sub | Parallel degradation accumulator |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Out-of-Distribution Perturbation Bias**:
   - *Symptom*: LIME or SHAP assigns high importance to features that have zero real-world causal relevance.
   - *Cause*: Marginal feature perturbation replaces masked features with independent marginal samples, creating impossible off-manifold inputs (e.g. an image with a dog's head on an airplane fuselage) where model predictions are undefined.
   - *Fix*: Perturb along conditional data manifolds or use in-distribution generative inpainting.
2. **Cosmetic Explanation Trap**:
   - *Symptom*: Saliency maps highlight clean human-interpretable contours, but randomizing model weights produces the exact same map (*Adebayo et al. 2018*).
   - *Cause*: Gradient and perturbation methods can act as edge detectors invariant to learned model weights.
   - *Fix*: Audit explanations via ROAR (Remove and Retrain) and cascading weight randomization sanity checks.
3. **LIME Sample Instability**:
   - *Symptom*: Rerunning LIME on the exact same input instance $x$ yields conflicting top-3 explanatory features.
   - *Cause*: Insufficient Monte Carlo sample size $N$ relative to feature dimension $M$, causing high estimator variance in weighted ridge regression.
   - *Fix*: Increase sample size $N \ge 1000$ or use deterministic KernelSHAP subset sampling.

---

## 7. Real-World Engineering Applications

1. **Clinical Healthcare & ICU Mortality**: Explaining why a deep survival model flags a patient for sepsis, highlighting elevated lactate and abnormal heart rate variability for clinical verification.
2. **Algorithmic Credit Scoring & Regulatory Compliance**: Providing legally mandated adverse action notices (Equal Credit Opportunity Act) explaining credit denial decisions without exposing proprietary model weights.
3. **Autonomous Driving Sensor Fault Detection**: Identifying which specific lidar/camera patches triggered an emergency braking event to isolate sensor artifacts from true obstacles.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `lime_w_ref`: LIME closed-form weighted ridge regression attribution vector.
3. `shap_weights_ref`: Exact KernelSHAP combinatorial weighting vector.
4. `rise_importance_ref`: RISE randomized input sampling importance vector.
5. `roar_degradation_ref`: Faithfulness degradation curve under feature masking.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 9. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.py`](../../pytorch2/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.py)
- **JAX Reference**: [`jax/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.py`](../../jax/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.py)
- **Candle Reference**: [`candle/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.rs`](../../candle/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.rs)
- **Kernels**: [`kernels/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.rs`](../../kernels/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.rs)
- **Interactive Notebook**: [`notebooks/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.ipynb`](../../notebooks/10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../INDEX_BY_GOAL.md)
