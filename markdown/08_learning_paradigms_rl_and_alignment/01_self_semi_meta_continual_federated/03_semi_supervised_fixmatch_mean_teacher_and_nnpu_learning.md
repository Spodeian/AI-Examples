# Semi-Supervised Learning: FixMatch, Mean Teacher, and nnPU Learning

## 1. Mathematical Foundations & Lineage

Semi-supervised learning (SSL) aims to leverage abundant unlabeled observations $\mathcal{D}_u = \{u_b\}_{b=1}^{B_u}$ alongside scarce labeled supervision $\mathcal{D}_l = \{(x_b, y_b)\}_{b=1}^{B_l}$. The mathematical foundation relies on the **smoothness assumption** (inputs close in high-density regions share labels) and the **cluster assumption** (decision boundaries cross low-density regions).

Key historical lineage:
1. **Mean Teacher (Tarvainen & Valpola, 2017)**: Replaces unstable instant model predictions with temporal ensembling via an Exponential Moving Average (EMA) teacher network: $\theta'_t = \alpha \theta'_{t-1} + (1-\alpha) \theta_t$. Consistency is measured via MSE between student and teacher predictions.
2. **MixMatch (Berthelot et al., 2019)**: Unifies consistency regularization and entropy minimization by guessing labels across $K$ augmented views, sharpening guessed distributions with a temperature parameter $T$, and mixing labeled and unlabeled samples via Beta-distributed Mixup.
3. **FixMatch (Sohn et al., 2020)**: Simplifies SSL by combining weak-to-strong augmentation consistency with hard pseudo-label thresholding. A weakly augmented sample $u^w$ generates pseudo-label $\hat{y} = \arg\max q(y|u^w)$ if confidence $\max q(y|u^w) \ge \tau$ (typically $\tau = 0.95$). Cross-entropy is evaluated exclusively against strongly augmented views $u^s$.
4. **Positive-Unlabeled Learning & nnPU (Kiryo et al., 2017)**: Addresses real-world classification where negative examples are entirely unobserved (e.g., disease diagnosis, anomaly detection). Standard unbiased PU risk (`uPU`) easily drifts negative ($-\infty$) when deep models overfit. Non-negative PU (`nnPU`) adds a non-negative constraint $\max(0, \hat{R}_u^- - \pi_p \hat{R}_p^-)$ to stabilize risk bounds.

---

## 2. FixMatch Weak-to-Strong Augmentation Consistency

FixMatch formulates the semi-supervised objective as the sum of a supervised cross-entropy loss $\mathcal{L}_s$ and an unsupervised consistency loss $\mathcal{L}_u$:
$$\mathcal{L} = \mathcal{L}_s + \lambda_u \mathcal{L}_u$$

Given unlabeled batch $\{u_b\}_{b=1}^B$, weak augmentation $\alpha(\cdot)$ (flip-and-shift), and strong augmentation $\mathcal{A}(\cdot)$ (RandAugment / CTAugment):
1. Compute weak-view prediction:
   $$q_b = p_m(y \mid \alpha(u_b)) = \text{softmax}(f_\theta(\alpha(u_b)))$$
2. Generate one-hot pseudo-label:
   $$\hat{y}_b = \arg\max(q_b), \quad p_b^{\max} = \max(q_b)$$
3. Compute masked cross-entropy on strong augmentation:
   $$\mathcal{L}_u = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(p_b^{\max} \ge \tau) \cdot H(\hat{y}_b, p_m(y \mid \mathcal{A}(u_b)))$$

By filtering out pseudo-labels with confidence below $\tau$, FixMatch protects against confirmation bias where early incorrect predictions poison gradient updates.

---

## 3. Mean Teacher Temporal Ensembling & Exponential Moving Average

In Mean Teacher, the teacher network weights $\theta'$ are maintained as a slow exponential moving average of student weights $\theta$:
$$\theta'_t = \alpha \theta'_{t-1} + (1 - \alpha) \theta_t$$
where $\alpha \in [0.99, 0.999]$.

For each sample $x$, student and teacher predictions under independent stochastic perturbations $\eta, \eta'$ are compared under mean squared error:
$$\mathcal{L}_{\text{consistency}} = \frac{1}{B} \sum_{i=1}^B \| \text{softmax}(f_\theta(x_i, \eta)) - \text{softmax}(f_{\theta'}(x_i, \eta')) \|_2^2$$

Because teacher weights aggregate thousands of gradient steps, teacher predictions are dramatically smoother and less prone to high-frequency minibatch noise than student predictions.

---

## 4. MixMatch & Label Sharpening

To enforce the cluster assumption (decision boundaries avoid high-density data regions), MixMatch introduces entropy minimization via temperature sharpening.

Given average predicted probabilities $\bar{q} = \frac{1}{K} \sum_{k=1}^K p_m(y \mid \text{Aug}_k(u))$ across $K$ augmentations:
$$\text{Sharpen}(\bar{q}, T)_i = \frac{\bar{q}_i^{1/T}}{\sum_{j=1}^C \bar{q}_j^{1/T}}$$
where $T \to 0$ drives $\text{Sharpen}(\bar{q}, T)$ toward a Dirac one-hot distribution, lowering entropy and encouraging confident decision boundaries.

---

## 5. Non-Negative Positive-Unlabeled (nnPU) Learning

In Positive-Unlabeled (PU) learning, data consists of positive samples $x_p \sim p(x \mid y=+1)$ and unlabeled samples $x_u \sim p(x)$ with positive class prior $\pi_p = p(y=+1)$.

### Unbiased PU Risk (uPU)
Decomposing the negative distribution risk:
$$p(x) = \pi_p p(x \mid y=+1) + (1-\pi_p) p(x \mid y=-1)$$
$$(1-\pi_p) \mathbb{E}_{x \sim p(x \mid y=-1)} [\ell(g(x), -1)] = \mathbb{E}_{x \sim p(x)} [\ell(g(x), -1)] - \pi_p \mathbb{E}_{x \sim p(x \mid y=+1)} [\ell(g(x), -1)]$$

This yields the unbiased PU risk estimator:
$$\hat{R}_{\text{uPU}}(g) = \pi_p \hat{R}_p^+(g) + \left( \hat{R}_u^-(g) - \pi_p \hat{R}_p^-(g) \right)$$
where $\hat{R}_p^+(g) = \frac{1}{N_p} \sum \ell^+(g(x_p^{(i)}))$, $\hat{R}_p^-(g) = \frac{1}{N_p} \sum \ell^-(g(x_p^{(i)}))$, and $\hat{R}_u^-(g) = \frac{1}{N_u} \sum \ell^-(g(x_u^{(j)}))$.

### Non-Negative Fix (nnPU)
Because complex neural networks $g$ have high Rademacher complexity, empirical negative risk $\hat{R}_n^- = \hat{R}_u^- - \pi_p \hat{R}_p^-$ frequently falls below zero ($\hat{R}_n^- < 0$), causing severe overfitting where the model classifies all unlabeled points as negative. Kiryo et al. (2017) introduce the non-negative constraint:
$$\tilde{R}_{\text{nnPU}}(g) = \pi_p \hat{R}_p^+(g) + \max\left(0, \hat{R}_u^-(g) - \pi_p \hat{R}_p^-(g)\right)$$

---

## 6. Cross-Framework Implementations & Numerical Parity

Every component in this topic is validated across PyTorch 2.x, JAX, and Candle against seed-42 golden weights in `03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning.safetensors`:
1. `linear_anchor`: Multi-framework verification anchor.
2. `mean_teacher_consistency_loss`: Multi-class probability MSE between student and teacher.
3. `fixmatch_thresholded_loss`: Dynamic masking $\mathbb{I}(\max q \ge \tau)$ and cross-entropy on strong views.
4. `mixmatch_sharpen`: Temperature-scaled distribution sharpening.
5. `nnpu_risk`: Clamped negative risk optimization for positive-unlabeled data.

All implementations match with maximum absolute difference $< 10^{-5}$.

---

## 7. Hardware Acceleration & WGSL Compute Shaders

The confidence-thresholding step in FixMatch is accelerated in WebGPU via `FIXMATCH_THRESHOLD_CE_WGSL`:
```wgsl
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= cfg.b) {
        return;
    }
    let offset = i * cfg.k;
    var max_p: f32 = -1.0;
    var argmax: u32 = 0u;
    for (var j: u32 = 0u; j < cfg.k; j = j + 1u) {
        let p = weak_probs[offset + j];
        if (p > max_p) {
            max_p = p;
            argmax = j;
        }
    }
    if (max_p >= cfg.tau) {
        loss[i] = -strong_log_probs[offset + argmax];
    } else {
        loss[i] = 0.0;
    }
}
```
This fused shader evaluates the argmax, verifies the threshold $\tau$, and extracts the target cross-entropy log-probability in a single compute pass.

---

## 8. Performance Profiling, Anti-Patterns & Misuse Guardrails

### Performance Profiling
1. **Weak-to-Strong Compute Ratio**: Weak augmentations (random horizontal flip, random crop) require minimal CPU/GPU pre-processing, while strong augmentations (RandAugment, Cutout) are compute-heavy. Overlapping strong augmentation generation with the forward pass of the weak view prevents pipeline stalls.
2. **Batched Unlabeled Ratio $\mu$**: FixMatch typically sets unlabeled batch size $B_u = \mu B_l$ with $\mu \in [5, 7]$. Large $\mu$ leverages GPU parallelism efficiently without consuming labeled memory.

### Misuse Guardrails & Anti-Patterns
1. **Threshold $\tau$ Confirmation Bias**: Setting $\tau < 0.7$ in FixMatch causes confirmation bias where early incorrect predictions reinforce false labels, resulting in permanent performance degradation.
2. **uPU Unclamped Overfitting**: Using standard unbiased PU learning without the non-negative $\max(0, \cdot)$ clamp in `nnPU` causes negative empirical risk $\to -\infty$, misclassifying the entire unlabeled pool as negative.
3. **EMA Teacher Gradient Leakage**: Forgetting to detach or stop gradients on the Mean Teacher network leads to recursive self-referential gradient updates, destabilizing training.
