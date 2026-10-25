# XAI Concepts: TCAV, Concept Bottleneck Models, DiCE Counterfactuals, and Global ALE

## 1. Mathematical Foundations & Lineage

While feature-attribution methods explain predictions via low-level pixel or feature scores, concept-based and counterfactual methods elevate explainability to high-level, human-understandable concepts and actionable recourse.

The lineage of concept and counterfactual methods comprises:
1. **Counterfactual Explanations (Wachter et al. 2017)**: Formulated explainability as the minimal perturbation necessary to change a model's prediction: $\min_{x^*} d(x, x^*) + \lambda \mathcal{L}(f(x^*), y^*)$.
2. **TCAV (Testing with Concept Activation Vectors — Kim et al. 2018)**: Quantified the sensitivity of neural predictions to high-level semantic concepts (e.g., "stripes" or "textures") by taking directional derivatives along Concept Activation Vectors (CAVs) orthogonal to concept-separating hyperplanes in latent activation space.
3. **Concept Bottleneck Models (CBM — Koh et al. 2020)**: Built inherently interpretable architectures that decompose the mapping $x \to y$ into two stages: $x \to \hat{c}$ (intermediate concept prediction) followed by $\hat{c} \to \hat{y}$ (task prediction), enabling human-in-the-loop test-time concept intervention.
4. **DiCE (Diverse Counterfactual Explanations — Mothilal et al. 2020)**: Formulated multi-counterfactual generation via Determinantal Point Processes (DPP) to generate diverse, non-redundant paths of recourse.
5. **ALE (Accumulated Local Effects — Apley & Zhu 2020)**: Resolved the severe extrapolation bias of Partial Dependence Plots (PDP) on correlated features by integrating conditional local differences along feature intervals.

---

## 2. TCAV Formulation & Directional Concept Vectors

Let $h_l: \mathcal{X} \to \mathbb{R}^D$ be the latent representation at layer $l$. Given a dataset of concept examples $P_C$ and random counter-examples $N_C$, a linear classifier is trained in activation space. The unit normal vector to the separating hyperplane defines the Concept Activation Vector:
$$v_C^l = \frac{w_C}{\|w_C\|_2}$$

### Conceptual Directional Derivative
The sensitivity of output class $k$ to concept $C$ at input $x$ is the directional derivative:
$$S_{C, k, l}(x) = \nabla_{h_l(x)} f_k(x) \cdot v_C^l = \sum_{d=1}^D \frac{\partial f_k(x)}{\partial h_{l, d}(x)} v_{C, d}^l$$

### TCAV Score
The overall importance of concept $C$ for class $k$ across dataset $X_k$ is the fraction of positive directional derivatives:
$$\text{TCAV}_{C, k, l} = \frac{| \{ x \in X_k : S_{C, k, l}(x) > 0 \} |}{|X_k|}$$

---

## 3. Concept Bottleneck Models (CBM) & Test-Time Intervention

A Concept Bottleneck Model parameterizes the joint predictor as:
$$\hat{c} = g_\phi(x) \in [0, 1]^K, \quad \hat{y} = h_\theta(\hat{c}) \in \mathcal{Y}$$

### Test-Time Concept Intervention
When an automated system makes an error or encounters an ambiguous case, a human domain expert can inspect the intermediate concept activations $\hat{c}$ and manually overwrite a subset $I \subset \{1, \dots, K\}$ with ground truth $c^{\text{true}}$:
$$c^{\text{int}}_k = \begin{cases} c^{\text{true}}_k & \text{if } k \in I \\ \hat{c}_k & \text{otherwise} \end{cases}$$
The final task prediction is re-evaluated via $\hat{y}_{\text{int}} = h_\theta(c^{\text{int}})$.

---

## 4. Diverse Counterfactuals (DiCE) & Accumulated Local Effects (ALE)

### DiCE Determinantal Diversity Metric
To provide users with actionable alternatives, DiCE generates $K$ distinct counterfactuals $\{c_1, \dots, c_K\}$ maximizing diversity measured by the determinant of a kernel distance matrix:
$$K_{i, j} = \frac{1}{1 + \|c_i - c_j\|_2}, \quad \text{Diversity} = \det(K)$$
Because the volume of the parallelotope spanned by vectors is proportional to $\det(K)$, redundant or collinear counterfactuals drive $\det(K) \to 0$.

### Accumulated Local Effects (ALE)
Unlike Partial Dependence Plots (PDP) which marginalize over the unconditional feature distribution (creating impossible feature combinations), ALE computes differences within localized conditional intervals $[z_{k-1}, z_k]$:
$$\hat{\Delta}_j(z_k) = \frac{1}{n_k} \sum_{i \in N_k} [f(z_k, x_{i, -j}) - f(z_{k-1}, x_{i, -j})]$$
$$\text{ALE}_j(x_j) = \sum_{k=1}^{k(x_j)} \hat{\Delta}_j(z_k) - \mathbb{E}[\text{ALE}_j]$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`tcav_directional_derivatives`** | Matrix-vector dot product + mask mean | Array contraction + `jnp.mean` | In-place dot and threshold | Vectorized CAV project |
| **`cbm_test_time_intervention`** | Masked linear interpolation + linear | Convex linear combination | Vectorized mask combination | `CBM_INTERVENTION_WGSL` |
| **`dice_dpp_diversity`** | Pairwise distance + `torch.linalg.det` | Pairwise norm + `jnp.linalg.det` | Exact $3\times 3$ determinant | Parallel pairwise kernel |
| **`ale_accumulated_local_effects`** | `torch.cumsum` - mean centering | `jnp.cumsum` - mean centering | Cumulative sum vector reducer | Parallel prefix scan |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Information Leakage via Soft Concept Bottlenecks**:
   - *Symptom*: Expert test-time concept intervention fails to correct task predictions $\hat{y}$.
   - *Cause*: High-capacity encoders compress non-concept task information into continuous soft concept vectors $\hat{c} \in \mathbb{R}^K$ (side-channel leakage).
   - *Fix*: Enforce discrete or sigmoid-bounded concept bottlenecks and add mutual information regularization between non-concept features and task output.
2. **PDP Extrapolation Bias on Correlated Features**:
   - *Symptom*: A model predicting vehicle fuel efficiency assigns high importance to an impossible vehicle with 8 cylinders and 500cc displacement.
   - *Cause*: Partial Dependence Plots perturb features independently, evaluating off-manifold data points.
   - *Fix*: Replace PDP with Accumulated Local Effects (ALE) which condition feature differences strictly on localized conditional slices.
3. **Counterfactual Implausibility & Actionability Failure**:
   - *Symptom*: Counterfactual recourse advises an applicant to change their age or decrease their years of education.
   - *Cause*: Unconstrained distance metrics that lack feasibility and monotonicity constraints on non-actionable attributes.
   - *Fix*: Enforce immutable feature masks ($M_{\text{immutable}}$) and sign constraints in counterfactual optimization.

---

## 7. Real-World Engineering Applications

1. **Medical Diagnostics (Dermatology / Radiology)**: Physicians verify whether a melanoma classifier relies on verified dermatological concepts (pigment network, globules, asymmetry) rather than skin tone or image markers.
2. **Financial Lending & Adverse Action Recourse**: Providing rejected loan applicants with diverse, actionable steps (e.g., "increase savings by \$5,000" vs. "pay down credit card by \$2,000") to cross the approval threshold.
3. **Scientific Discovery & Climate Modeling**: Measuring whether global precipitation forecasts respond to high-level El Niño Southern Oscillation (ENSO) indices.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `tcav_derivs_ref` & `tcav_score_ref`: Directional derivatives and dataset TCAV score.
3. `cbm_c_int_ref` & `cbm_y_ref`: Intervened concept representations and task predictions.
4. `dice_div_ref`: DiCE DPP kernel determinant diversity score.
5. `ale_ref`: Centered Accumulated Local Effects curve.
