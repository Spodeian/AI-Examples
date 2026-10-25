# Out-of-Distribution Detection, Calibrated Abstention, Conformal Prediction, and Algorithmic Fairness

## 1. Mathematical Foundations & Lineage

Real-world deployment of machine learning in safety-critical domains (medicine, criminal justice, autonomous driving, finance) demands that models know when they do not know, guarantee finite-sample error bounds, and uphold ethical parity:
1. **Optimal Rejection Option (Chow 1970)**: Formulated classification with a reject option under cost trade-offs, proving that optimal abstention rules threshold the posterior error probability $\mathbb{P}(Y \ne \hat{Y} \mid X) \le \epsilon$.
2. **Algorithmic Fairness Criteria (Hardt, Price & Srebro 2016)**: Formalized Demographic Parity ($P(\hat{Y}=1 \mid A=0) = P(\hat{Y}=1 \mid A=1)$) and Equalized Odds ($P(\hat{Y}=1 \mid A=a, Y=y)$ invariant to $a$), proving the impossibility of satisfying calibration, parity, and equal opportunity simultaneously.
3. **Selective Classification (Geifman & El-Yaniv 2017, 2019 / SelectiveNet)**: Trained deep networks with an integrated rejection head $g(x) \in [0, 1]$, optimizing empirical selective risk under target coverage constraints $c$.
4. **Energy-Based OOD Detection (Liu et al. 2020)**: Replaced uncalibrated Max Softmax Probability (MSP) with the Helmholtz free energy function $E(x) = -T \log \sum_{c=1}^C e^{f_c(x)/T}$, demonstrating provably superior separation between in-distribution (ID) and out-of-distribution (OOD) inputs.
5. **Split Conformal Prediction (Vovk et al. 2005; Angelopoulos & Bates 2021)**: Provided distribution-free, finite-sample valid prediction sets $C(X)$ satisfying $\mathbb{P}(Y \in C(X)) \ge 1 - \alpha$ for any underlying distribution without parametric assumptions.

---

## 2. Energy-Based Out-of-Distribution Detection

Given classifier logits $f(x) \in \mathbb{R}^C$:
Under Gibbs distribution $p(y=c \mid x) = \frac{e^{f_c(x)/T}}{\sum_{j=1}^C e^{f_j(x)/T}}$, the denominator represents the partition function:
$$Z(x) = \sum_{c=1}^C e^{f_c(x) / T}$$
The Helmholtz free energy is defined as:
$$E(x; f) = -T \cdot \ln Z(x) = -T \cdot \ln \sum_{c=1}^C e^{f_c(x) / T}$$
- **In-Distribution (ID) Samples**: Yield large positive logits aligned with training classes, resulting in large partition functions and **low (strongly negative) free energy** ($E(x) \ll 0$).
- **Out-of-Distribution (OOD) Samples**: Yield low or diffuse logits, resulting in small partition functions and **high free energy** ($E(x) \approx 0$ or $E(x) > 0$).
Unlike Softmax probabilities $\frac{e^{f_c}}{\sum e^{f_j}}$ which are invariant to arbitrary constant shifts $f_c + \beta$, the free energy captures the unnormalized magnitude of model evidence.

---

## 3. Split Conformal Prediction: Distribution-Free Coverage Guarantees

Given exchangeable calibration set $(X_1, Y_1), \dots, (X_n, Y_n)$:
1. **Nonconformity Score**: Compute $s_i = s(X_i, Y_i) \in \mathbb{R}$ measuring how unusual sample $i$ is (e.g., $s_i = 1 - \hat{p}(Y_i \mid X_i)$).
2. **Empirical Quantile Calibration**: Sort scores in non-decreasing order: $s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$.
   Compute the critical index:
   $$k = \min\left(n, \left\lceil (n + 1)(1 - \alpha) \right\rceil\right)$$
   $$\hat{q} = s_{(k)}$$
3. **Inference Prediction Set**: For a new test sample $X_{n+1}$:
   $$C(X_{n+1}) = \left\{ y \in \{1, \dots, C\} : s(X_{n+1}, y) \le \hat{q} \right\}$$
**Theorem (Validity)**:
$$\mathbb{P}\left( Y_{n+1} \in C(X_{n+1}) \right) \ge 1 - \alpha$$
This exact coverage holds for any dataset, any architecture, and any sample size $n$, with zero asymptotic approximations!

---

## 4. Selective Classification & Calibrated Abstention (SelectiveNet)

SelectiveNet outputs both prediction logits $f(x)$ and selection probability $g(x) = \sigma(z_{\text{sel}}(x)) \in [0, 1]$.
1. **Selective Empirical Risk**:
   $$\mathcal{L}_{\text{sel}} = \frac{\sum_{i=1}^B \mathcal{L}_{\text{CE}}(f(x_i), y_i) \cdot g(x_i)}{\sum_{i=1}^B g(x_i) + 10^{-8}}$$
2. **Empirical Coverage & Constraint Penalty**:
   $$\hat{\phi} = \frac{1}{B} \sum_{i=1}^B g(x_i)$$
   $$\mathcal{L}_{\text{cov}} = \lambda \cdot \left(\max(0, c - \hat{\phi})\right)^2$$
   where $c \in (0, 1]$ is the user-specified minimum coverage target (e.g., $c = 0.8$).
3. **Total Loss**:
   $$\mathcal{L} = \mathcal{L}_{\text{sel}} + \mathcal{L}_{\text{cov}}$$
Samples with low selection score $g(x) < \tau$ are rejected and handed off to human experts.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`energy_ood_score`** | `torch.logsumexp` scaled | `jax.nn.logsumexp` scaled | Vectorized logsumexp | `ENERGY_OOD_SCORE_WGSL` |
| **`conformal_quantile`** | Sort + ceil index lookup | `jnp.sort` + ceil lookup | Vector sort & quantile | Parallel radix quantile |
| **`selective_net_loss`** | Weighted risk + coverage | Functional mean + clamp | Vector risk & penalty | Fused selective reducer |
| **`equalized_odds`** | Boolean mask TPR/FPR | Boolean mask TPR/FPR | Vector loop confusion sum | Confusion matrix reducer |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Overconfident MSP on Featureless Backgrounds**:
   - *Symptom*: Maximum Softmax Probability (MSP) outputs 99% confidence on solid black noise images.
   - *Cause*: Softmax normalization masks uniformly low logit activations.
   - *Fix*: Replace MSP with Energy OOD scoring ($E(x) = -T \log \sum e^{f_c/T}$), which directly penalizes small logit magnitudes.
2. **Conformal Quantile Overfitting on Training Splits**:
   - *Symptom*: Conformal prediction set under-covers test data ($\mathbb{P}(Y \in C(X)) \ll 1 - \alpha$).
   - *Cause*: Evaluating nonconformity scores on training data instead of a strictly disjoint calibration set.
   - *Fix*: Strictly split data into separate training, calibration, and test splits.
3. **Fairness Through Unawareness Pitfall**:
   - *Symptom*: Removing sensitive attribute column $A$ fails to eliminate discrimination.
   - *Cause*: High-dimensional proxy features (zip code, browser type, vocabulary) reconstruct $A$.
   - *Fix*: Enforce explicit Equalized Odds post-processing or adversarial debiasing.

---

## 7. Real-World Engineering Applications

1. **Medical Imaging Triage**: Abstaining on ambiguous radiological scans and routing them to human radiologists when energy score $E(x) > \tau_{\text{ood}}$.
2. **Credit Risk & Loan Approval**: Guaranteeing demographic parity and equal opportunity across protected classes under federal lending regulations.
3. **Safety-Critical Conformal Robotics**: Generating certified bounding sets for obstacle perception in autonomous vehicles.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.safetensors`:
1. `energy_ref`: Helmholtz free energy OOD scores across test batch.
2. `q_hat_ref`: Split conformal prediction calibration quantile threshold.
3. `selective_loss_ref`: SelectiveNet empirical risk and coverage penalty.
4. `eo_discrepancy_ref`: Equalized Odds cumulative TPR/FPR disparity.

### Synchronized Ecosystem Artifacts
- **Notebook:** [09_ood_detection_abstention_conformal_and_algorithmic_fairness.ipynb](../../notebooks/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.ipynb)
- **PyTorch 2 Module:** [09_ood_detection_abstention_conformal_and_algorithmic_fairness.py](../../pytorch2/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.py)
- **JAX Module:** [09_ood_detection_abstention_conformal_and_algorithmic_fairness.py](../../jax/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.py)
- **Candle Module:** [09_ood_detection_abstention_conformal_and_algorithmic_fairness.rs](../../candle/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.rs)
- **Hardware Kernel:** [09_ood_detection_abstention_conformal_and_algorithmic_fairness.rs](../../kernels/11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.rs)
