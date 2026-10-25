# XAI Gradient Propagation: Integrated Gradients, LRP, Grad-CAM, and Attention Attributions

## 1. Mathematical Foundations & Lineage

Gradient-based attribution methods explain deep neural network predictions by propagating gradients or relevances backward from the target output class to the input or intermediate feature representations.

The lineage of gradient and relevance propagation methods comprises:
1. **Vanilla Saliency (Simonyan et al. 2013)**: Formulated attribution as the raw first-order Taylor approximation $\nabla_x f_c(x)$, which suffers from gradient saturation in non-linear activations.
2. **Layer-wise Relevance Propagation (LRP — Bach et al. 2015)**: Introduced conservative layer-by-layer relevance redistribution backward through deep networks satisfying conservation: $\sum_i R_i^{(l)} = \sum_j R_j^{(l+1)}$.
3. **DeepLIFT (Shrikumar et al. 2017)**: Resolved gradient saturation by decomposing output difference $\Delta y = f(x) - f(x_0)$ relative to a reference baseline across activation differences $\Delta x = x - x_0$.
4. **Integrated Gradients (Sundararajan et al. 2017)**: Axiomatized feature attribution under two fundamental axioms: *Completeness* ($\sum_i \text{IG}_i(x) = F(x) - F(x')$) and *Implementation Invariance*, integrating gradients along a straight-line interpolation path from baseline $x'$ to $x$.
5. **Grad-CAM & Grad-CAM++ (Selvaraju et al. 2017; Chattopadhay et al. 2018)**: Visualized intermediate convolutional and spatial representations by pooling gradients into channel attention weights $\alpha_c$ followed by ReLU filtering.
6. **Adebayo Sanity Checks (Adebayo et al. 2018)**: Proved that popular methods like Guided Backpropagation and DeconvNet act as class-agnostic edge detectors invariant to network weights, failing cascading model randomization tests.

---

## 2. Integrated Gradients Axiomatic Formulation

Integrated Gradients computes the path integral of gradients along the straight-line trajectory from baseline $x'$ to input $x$:
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

### Riemann Sum Numerical Approximation
In practice, the continuous path integral is approximated via a discrete Riemann summation across $M$ interpolation steps (typically $M \in [20, 50]$):
$$\text{IG}_i^{\text{approx}}(x) = (x_i - x'_i) \times \frac{1}{M} \sum_{m=1}^M \frac{\partial F\left( x' + \frac{m}{M}(x - x') \right)}{\partial x_i}$$

### Fundamental Axioms
- **Completeness**: $\sum_{i=1}^D \text{IG}_i(x) = F(x) - F(x')$, ensuring that total attribution accounts for 100% of the prediction change relative to the baseline.
- **Implementation Invariance**: Two models with identical input-output behavior $F_1(x) = F_2(x) \ \forall x$ produce identical attributions, irrespective of internal architectural parametrization.

---

## 3. Layer-Wise Relevance Propagation (LRP) Conservation

LRP propagates a scalar relevance score $R_j^{(l+1)}$ backward from layer $l+1$ to layer $l$ such that total relevance is conserved across layers: $\sum_i R_i^{(l)} = \sum_j R_j^{(l+1)}$.

### The LRP-$\epsilon$ Rule
For a forward linear operation $z_j = \sum_i a_i w_{ij} + b_j$, the input-to-output contribution is $z_{ij} = a_i w_{ij}$. To prevent division by zero when $\sum_i z_{ij} \approx 0$, the $\epsilon$-rule introduces a stabilizer:
$$R_i^{(l)} = \sum_j \frac{z_{ij}}{\sum_{i'} z_{i'j} + \epsilon \cdot \text{sign}\left(\sum_{i'} z_{i'j}\right)} R_j^{(l+1)}$$
When $\epsilon > 0$, the rule absorbs weak or conflicting evidence, emphasizing the most prominent contributing features.

---

## 4. Grad-CAM Spatial Attributions

Grad-CAM constructs coarse 2D localization maps highlighting the discriminative regions in intermediate convolutional or spatial activation maps $A^k \in \mathbb{R}^{H \times W}$.

### Channel Importance Weights
The importance weight $\alpha_k^c$ for feature channel $k$ and target class $c$ is computed via global average pooling of gradients:
$$\alpha_k^c = \frac{1}{H \times W} \sum_{h=1}^H \sum_{w=1}^W \frac{\partial y^c}{\partial A_{h, w}^k}$$

### Rectified Linear Combination
The final Grad-CAM heatmap applies a rectified linear combination to isolate features that positively contribute to class $c$:
$$L_{\text{GradCAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`integrated_gradients`** | Riemann average along path | `jnp.mean` path reduction | Broadcast mul with mean grad | `INTEGRATED_GRADIENTS_WGSL` |
| **`lrp_epsilon_rule`** | Outer product matrix redistribution | Pure functional array matmul | Vectorized conservation solver | Elementwise relevance reducer |
| **`grad_cam_2d`** | Spatial GAP + ReLU linear sum | Tensor reduction over H,W | Global average pooling + ReLU | Parallel spatial CAM reducer |
| **`cosine_similarity`** | Vectorized dot / norm quotient | `jnp.dot` over product norms | In-place dot and norm reducer | Vectorized cosine evaluator |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Obfuscated Gradient & Edge Detector Trap**:
   - *Symptom*: Guided Backprop produces razor-sharp saliency maps highlighting object edges, but randomizing top-layer weights produces the identical edge map (*Adebayo et al. 2018*).
   - *Cause*: Modified gradient backpropagation passes act as class-agnostic edge filters, completely decoupling from model decision logic.
   - *Fix*: Rely on axiomatically grounded methods (Integrated Gradients, LRP) and verify against weight randomization sanity checks.
2. **Baseline Blindness in Integrated Gradients**:
   - *Symptom*: Features that differ strongly from the true prior distribution receive 0 attribution if they happen to equal the chosen baseline value $x'_i = x_i$.
   - *Cause*: Integrated Gradients scales attributions by $(x_i - x'_i)$. An all-black image baseline causes all black pixels in the input to receive 0 attribution.
   - *Fix*: Use multiple random or blurred baselines (Expected Gradients / Gaussian baselines).
3. **LRP Gradient Exploding without Stabilizer**:
   - *Symptom*: LRP relevance values explode to $\pm \infty$ or output `NaN`s in deep layers.
   - *Cause*: Denominators $z_j = \sum_i a_i w_{ij}$ crossing zero without an $\epsilon$ stabilizer.
   - *Fix*: Enforce $\epsilon$-rule with $\epsilon \in [10^{-4}, 10^{-2}]$ or $\alpha\beta$-rule bounding positive and negative contributions separately.

---

## 7. Real-World Engineering Applications

1. **Medical Imaging (Chest X-Ray Pathology)**: Validating that a COVID-19 or pneumonia classifier bases its diagnosis on pulmonary consolidation rather than hospital bed markers or radiologist tags.
2. **Autonomous Driving Obstacle Recognition**: Inspecting Grad-CAM heatmaps on backbone feature maps to ensure emergency brake triggers correlate with pedestrian centroids rather than shadows or road reflections.
3. **Financial Fraud & Risk Scoring**: Verifying that high-risk fraud flags are driven by anomalous transaction velocity rather than legally protected demographic attributes.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `ig_ref`: Integrated Gradients Riemann sum attribution vector.
3. `lrp_ref`: Layer-wise Relevance Propagation conservative relevance distribution.
4. `cam_ref`: 2D spatial Grad-CAM activation heatmap.
5. `adebayo_sim_ref`: Cosine similarity score under model weight randomization test.
