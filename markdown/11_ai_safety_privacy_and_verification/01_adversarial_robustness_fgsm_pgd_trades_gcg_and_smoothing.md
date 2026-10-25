# Adversarial Robustness: FGSM, PGD, TRADES, and Randomized Smoothing

## 1. Mathematical Foundations & Lineage

Deep neural networks exhibit extreme sensitivity to worst-case input perturbations that are imperceptible to human observers:
1. **Fast Gradient Sign Method (FGSM — Goodfellow, Shlens, & Szegedy 2014)**: Showed that vulnerability stems from the high-dimensional linearity of neural layers rather than non-linear overfitting, computing an optimal single-step $L_\infty$ perturbation via $\text{sign}(\nabla_x \mathcal{L})$.
2. **Projected Gradient Descent (PGD — Madry et al. 2018)**: Formulated adversarial robustness as a saddle-point minimax problem $\min_\theta \mathbb{E} [ \max_{\delta \in \Delta} \mathcal{L}(f_\theta(x + \delta), y) ]$, proving multi-step projected gradient ascent acts as the universal empirical first-order adversary.
3. **TRADES (Zhang et al. 2019)**: Identified the fundamental trade-off between clean generalization error and boundary robust error, proposing a regularized objective that penalizes Kullback-Leibler divergence between clean and perturbed output distributions.
4. **Randomized Smoothing (Cohen, Rosenfeld, & Kolter 2019)**: Transformed empirical adversarial defense into a mathematically certified guarantee by convolving base classifiers with isotropic Gaussian noise, bounding the maximum robust $L_2$ ball radius via the Neyman-Pearson lemma.

---

## 2. Gradient-Based Adversarial Attacks (FGSM & PGD)

Let $f(x) \in \mathbb{R}^C$ output logits for input $x \in [0, 1]^D$, and let $\mathcal{L}(f(x), y)$ be the loss.

### Fast Gradient Sign Method (FGSM)
Under the $L_\infty$ perturbation constraint $\|\delta\|_\infty \le \epsilon$:
$$x_{\text{adv}} = \text{clamp}_{[x_{\min}, x_{\max}]} \left( x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y)) \right)$$

### Projected Gradient Descent (PGD)
Multi-step PGD initializes at $x^{(0)} = x + \text{Uniform}(-\epsilon, \epsilon)$ and iterates with step size $\alpha$:
$$x^{(t+1)} = \Pi_{B_\epsilon(x)} \left( x^{(t)} + \alpha \cdot \text{sign}(\nabla_{x^{(t)}} \mathcal{L}(f(x^{(t)}), y)) \right)$$
where the projection operator $\Pi_{B_\epsilon(x)}(z) = \text{clamp}(z, x - \epsilon, x + \epsilon)$ enforces the $L_\infty$ constraint.

---

## 3. TRADES Robust Objective Formulation

Standard adversarial training ($\max_\delta \mathcal{L}(x+\delta, y)$) often causes a sharp drop in clean accuracy. TRADES decomposes the robust error into clean empirical risk plus a boundary regularization term:
$$\mathcal{L}_{\text{TRADES}}(x, y) = \mathcal{L}_{\text{CE}}(f(x), y) + \beta \cdot \max_{\delta \in B_\epsilon(0)} D_{\text{KL}}(P(f(x)) \parallel P(f(x + \delta)))$$
where $P(f(x)) = \text{softmax}(f(x))$. The hyperparameter $\beta > 0$ controls the trade-off: larger $\beta$ enforces stronger invariance across decision boundaries at the cost of slight clean classification accuracy.

---

## 4. Randomized Smoothing & Provable Certification

Randomized smoothing constructs a smoothed classifier $g(x) = \arg\max_c P(f(x + \epsilon) = c)$ where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.

Let $c_A$ be the top predicted class with probability $p_A$, and let $c_B$ be the runner-up class with probability $p_B < p_A$.
By the Neyman-Pearson lemma, the smoothed classifier $g(x)$ is provably robust against any perturbation $\|\delta\|_2 < R$:
$$R = \frac{\sigma}{2} \left( \Phi^{-1}(p_A) - \Phi^{-1}(p_B) \right)$$
where $\Phi^{-1}$ is the inverse standard normal cumulative distribution function (probit). If $p_A > 0.5$, then $p_B \le 1 - p_A$, certifying a minimum radius $R = \sigma \Phi^{-1}(p_A)$.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`fgsm_perturbation`** | `sign` + clamp | `jnp.sign` + `jnp.clip` | Vector sign iterator + clamp | Fast sign perturbation kernel |
| **`pgd_step`** | Dual clamp (ball + domain) | `jnp.clip` projection chain | Vector box projection | `PGD_STEP_WGSL` |
| **`trades_loss`** | CrossEntropy + KL divergence | Logsumexp + KL reduction | Fused stable logsumexp | Parallel KL loss reducer |
| **`randomized_smoothing_radius`** | `erfinv` probit conversion | `jsp.erfinv` probit | Acklam rational approximation | Vectorized probit evaluator |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Obfuscated Gradients / Shattered Gradients**:
   - *Symptom*: Defense claims 99% robust accuracy against 1-step FGSM, but collapses to 0% under multi-step PGD or SPSA.
   - *Cause*: Defensive distillation, non-differentiable activation steps, or input quantization break gradients without flattening true loss boundaries.
   - *Fix*: Evaluate using AutoAttack (APGD-CE, APGD-DLR, FAB, Square Attack) and black-box transfer attacks.
2. **Robust Overfitting**:
   - *Symptom*: Robust training loss continues decreasing, but robust test accuracy plummets mid-training.
   - *Cause*: Memorization of adversarial perturbations in overparameterized models.
   - *Fix*: Early stopping on validation robust accuracy and applying stochastic weight averaging (`SWAD`).
3. **Imbalanced Smoothing Confidence**:
   - *Symptom*: Randomized smoothing returns radius $R = 0$ on difficult boundary instances.
   - *Cause*: Insufficient Monte Carlo sample count ($N < 1000$) fails to resolve statistical significance between $p_A$ and $p_B$.
   - *Fix*: Use Clopper-Pearson binomial confidence intervals to lower-bound $p_A$ rigorously.

---

## 7. Real-World Engineering Applications

1. **Autonomous Vehicle Vision Hardening**: Hardening perception models against physical-world patch attacks and sensor noise.
2. **Medical Diagnosis Certification**: Computing provable $L_2$ stability bounds for radiology classifiers to ensure minor scanner artifacts cannot alter diagnoses.
3. **Adversarial Safety Auditing for LLMs**: Adapting continuous gradient attacks (GCG / Greedy Coordinate Gradient) to audit safety guardrails in foundation models.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `fgsm_adv_ref`: Single-step FGSM adversarial example.
3. `pgd_step_ref`: Projected Gradient Descent step within $L_\infty$ ball.
4. `trades_loss_ref`: Combined clean cross-entropy and KL adversarial regularization loss.
5. `smoothing_radius_ref`: Certified $L_2$ robustness radius vector.
