# Statistical Inference: Bayesian, Frequentist, Likelihoodist, MDL, and PAC-Bayes (Topic 11)

## 1. Executive Summary & Intuitive Mental Model

Machine learning practitioners often treat uncertainty estimation as an afterthought, relying exclusively on softmax probabilities or point-estimate loss gradients. However, modern autonomous AI requires principled uncertainty quantification, finite-sample guarantees, and robust model selection across distribution shifts.

This topic unifies the **5 Primary Schools of Statistical Inference & Uncertainty Analysis**:
1. **Frequentist School (Fisher, Neyman, Pearson):** Treats parameters $\theta$ as fixed, unknown physical constants and data $\mathcal{D}$ as a random variable. Characterizes uncertainty through long-run asymptotic sampling distributions, Fisher Information $\mathcal{I}(\theta)$, the Cramér-Rao lower bound, Profile Likelihood, and Neyman-Pearson hypothesis testing.
2. **Bayesian School (Bayes, Laplace, Jeffreys):** Treats parameters $\theta$ as random variables governed by epistemic probability distributions $p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$. Evaluates model evidence via the Laplace Marginal Likelihood $p(\mathcal{D}|\mathcal{M})$, where parameter volume is automatically penalized by the **Occam Factor** ($-\frac{1}{2}\log\det H$).
3. **Information-Theoretic & Minimum Description Length School (Rissanen, Wallace, Akaike):** Frames learning as optimal lossless compression. Selects models minimizing total bit-length: $\text{Length}(\mathcal{M}) + \text{Length}(\mathcal{D}|\mathcal{M})$. Encompasses AIC, BIC, WAIC, and Normalized Maximum Likelihood (NML).
4. **PAC-Bayesian School (McAllester, Catoni, Shawe-Taylor):** Provides *Probably Approximately Correct* finite-sample non-asymptotic generalization bounds for stochastic hypothesis spaces $Q$ relative to a reference prior $P$: $\mathbb{E}_{w \sim Q}[R(w)] \le \mathbb{E}_{w \sim Q}[\hat{R}(w)] + \mathcal{O}(\sqrt{\text{KL}(Q\|P)/N})$.
5. **Anytime-Valid E-Values & Game-Theoretic Probability (Ville, Vovk, Shafer, Ramdas):** Replaces rigid fixed-sample-size $p$-values with non-negative test martingales $M_t = \prod_{s=1}^t e_s$. Governed by Ville's inequality ($P(\exists t: M_t \ge 1/\alpha) \le \alpha$), allowing continuous monitoring, adaptive sampling, and optional stopping without $p$-hacking.

### Decision Boundary: When NOT to Use
> [!WARNING]
> - Do NOT interpret a 95% Frequentist Confidence Interval $[L, U]$ as stating "there is a 95% probability that $\theta^*$ lies in $[L, U]$" (a Bayesian Credible Interval statement); the true parameter is fixed, and the interval is the random entity.
> - Do NOT use un-penalized Maximum Likelihood $\max_\theta p(\mathcal{D}|\theta, \mathcal{M})$ for model architecture selection; it monotonically favors the most complex model until overfitting occurs, whereas **Bayesian Marginal Likelihood** or **MDL** penalizes parameter volume.
> - Do NOT perform sequential stopping with classical $p$-values; doing so invalidates Type-I error rates due to optional stopping. Use **Anytime-Valid E-Values**.

---

## 2. Mathematical Foundations & First-Principles Derivations

### The Frequentist School: Fisher Information & Cramér-Rao Bound
Let $x_1, \dots, x_N \sim p(x|\theta)$ be i.i.d. observations with log-likelihood:
$$\ell(\theta) = \sum_{i=1}^N \log p(x_i|\theta)$$
The **Score Function** is the gradient of log-likelihood:
$$S(\theta) = \nabla_\theta \ell(\theta), \quad \mathbb{E}[S(\theta)] = 0$$
The **Fisher Information Matrix** $\mathcal{I}_N(\theta) \in \mathbb{R}^{D \times D}$ measures the curvature of log-likelihood:
$$\mathcal{I}_N(\theta) = \mathbb{E}[S(\theta) S(\theta)^\top] = -\mathbb{E}[\nabla^2_\theta \ell(\theta)]$$
For Gaussian observations $x_i \sim \mathcal{N}(\theta, \sigma^2 I_D)$:
$$\mathcal{I}_N(\theta) = \frac{N}{\sigma^2} I_D$$
The **Cramér-Rao Inequality** bounds the variance of any unbiased estimator $\hat{\theta}$:
$$\text{Cov}(\hat{\theta}) \succeq \mathcal{I}_N(\theta)^{-1} = \frac{\sigma^2}{N} I_D$$
The Maximum Likelihood Estimator $\hat{\theta}_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N x_i$ achieves this lower bound exactly.

Under Wilks' Theorem, the Likelihood Ratio Test statistic against null hypothesis $H_0: \theta = \theta_0$ satisfies:
$$\Lambda = 2\left(\ell(\hat{\theta}_{\text{MLE}}) - \ell(\theta_0)\right) = \frac{N}{\sigma^2} \|\hat{\theta}_{\text{MLE}} - \theta_0\|_2^2 \xrightarrow{d} \chi^2_D$$

---

### The Bayesian School: Conjugate Updates & The Laplace Occam Factor
Under conjugate Gaussian prior $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2 I_D)$ and likelihood $x_i \sim \mathcal{N}(\theta, \sigma^2 I_D)$:
$$\tau_{\text{post}} = \sigma_0^{-2} + N \sigma^{-2}, \quad \sigma_{\text{post}}^2 = \tau_{\text{post}}^{-1}$$
$$\mu_{\text{post}} = \sigma_{\text{post}}^2 \left(\sigma_0^{-2}\mu_0 + N\sigma^{-2}\hat{\theta}_{\text{MLE}}\right)$$

The **Laplace Approximation** to the log marginal likelihood $p(\mathcal{D}|\mathcal{M}) = \int p(\mathcal{D}|\theta) p(\theta) d\theta$ around the MAP mode $\hat{\theta}_{\text{MAP}} = \mu_{\text{post}}$ is:
$$\log p(\mathcal{D}|\mathcal{M}) \approx \log p(\mathcal{D}|\hat{\theta}_{\text{MAP}}) + \log p(\hat{\theta}_{\text{MAP}}) + \frac{D}{2}\log(2\pi) - \frac{1}{2}\log\det(H)$$
where $H = -\nabla^2 \log p(\theta|\mathcal{D})|_{\hat{\theta}_{\text{MAP}}} = \tau_{\text{post}} I_D$.
The **Occam Factor** is:
$$\Omega = -\frac{1}{2}\log\det(H) + \frac{D}{2}\log(2\pi) = -\frac{D}{2}\log\left(\frac{\tau_{\text{post}}}{2\pi}\right)$$
As $N$ or $D$ increases, the Hessian determinant $\det(H)$ expands, penalizing models that waste probability mass across unnecessary parameter space.

---

### Stein's Paradox & The James-Stein Shrinkage Estimator
Consider $K \ge 3$ independent normal observation means $y_k \sim \mathcal{N}(\theta_k, \sigma^2)$. While $\hat{\theta}_{\text{MLE}} = y$ is the unique minimum-variance unbiased estimator, **Stein (1956)** proved it is inadmissible under squared error risk:
$$R(\hat{\theta}, \theta) = \mathbb{E}\left[\|\hat{\theta} - \theta\|_2^2\right]$$
The **James-Stein Shrinkage Estimator** (James & Stein, 1961) shrinks observations toward the origin:
$$\hat{\theta}_{\text{JS}} = \left(1 - \frac{(K - 2)\sigma^2}{\|y\|_2^2}\right) y$$
The **Positive-Part James-Stein Estimator** prevents sign reversals:
$$\hat{\theta}_{\text{JS}+} = \max\left(0, 1 - \frac{(K - 2)\sigma^2}{\|y\|_2^2}\right) y$$
For all $\theta \in \mathbb{R}^K$ and $K \ge 3$:
$$R(\hat{\theta}_{\text{JS}+}, \theta) < R(\hat{\theta}_{\text{JS}}, \theta) < R(\hat{\theta}_{\text{MLE}}, \theta)$$

---

### PAC-Bayesian Generalization Bounds (McAllester & Catoni)
Let $\mathcal{H}$ be a hypothesis space, $P$ a data-independent reference prior distribution on $\mathcal{H}$, and $Q$ a posterior distribution learned from $N$ samples.
For bounded loss $L \in [0, 1]$, with probability at least $1 - \delta$ over the sample draw:
$$\mathbb{E}_{w \sim Q}[R(w)] \le \mathbb{E}_{w \sim Q}[\hat{R}_N(w)] + \sqrt{\frac{\text{KL}(Q \| P) + \ln(2\sqrt{N}/\delta)}{2N}}$$
For Gaussian prior $P = \mathcal{N}(0, \sigma_P^2 I_D)$ and posterior $Q = \mathcal{N}(\mu_Q, \sigma_Q^2 I_D)$:
$$\text{KL}(Q \| P) = \frac{1}{2} \sum_{d=1}^D \left[ \frac{\mu_{Q,d}^2 + \sigma_{Q,d}^2}{\sigma_P^2} - 1 - \ln\left(\frac{\sigma_{Q,d}^2}{\sigma_P^2}\right) \right]$$

---

### Anytime-Valid E-Values & Ville's Martingale Inequality
An **E-variable** $E$ is a non-negative random variable satisfying $\mathbb{E}_{H_0}[E] \le 1$.
Given sequential data, the cumulative product forms a non-negative test martingale:
$$M_t = \prod_{s=1}^t e_s, \quad \mathbb{E}_{H_0}[M_t | \mathcal{F}_{t-1}] \le M_{t-1}$$
**Ville's Martingale Inequality (1939):**
$$P_{H_0}\left(\exists t \ge 1: M_t \ge \frac{1}{\alpha}\right) \le \alpha$$
Rejecting $H_0$ as soon as $M_t \ge 1/\alpha$ yields a strictly valid Level-$\alpha$ sequential test allowing arbitrary stopping and continuous data collection without multiplicity corrections.

---

## 3. Tensor Mechanics & Memory Layout Architecture

| Tensor Name | Shape | Dtype | Contiguity | Semantic Role |
| :--- | :--- | :--- | :--- | :--- |
| `x_data` | `(16, 4)` | `f32` | C-Contiguous | Batch Gaussian observations ($N=16, D=4$) |
| `theta_mle` | `(4,)` | `f32` | C-Contiguous | Frequentist sample mean vector |
| `fisher_diag` | `(4,)` | `f32` | C-Contiguous | Fisher information diagonal elements ($N/\sigma^2$) |
| `cramer_rao_lower_bound` | `(4,)` | `f32` | C-Contiguous | Minimum variance lower bound ($\sigma^2/N$) |
| `lrt_stat` | `()` | `f32` | Scalar | Wilks' $\chi^2$ likelihood ratio test statistic |
| `mu_post` | `(4,)` | `f32` | C-Contiguous | Posterior mean vector under Gaussian conjugate update |
| `sigma_post_vec` | `(4,)` | `f32` | C-Contiguous | Posterior standard deviation per dimension |
| `laplace_log_marginal` | `()` | `f32` | Scalar | Laplace approximation of log marginal likelihood |
| `occam_penalty` | `()` | `f32` | Scalar | Occam factor parameter volume penalty |
| `y_obs` | `(8,)` | `f32` | C-Contiguous | Group sample observation vector ($K=8$) |
| `theta_js` | `(8,)` | `f32` | C-Contiguous | Standard James-Stein shrinkage estimate |
| `theta_js_pos` | `(8,)` | `f32` | C-Contiguous | Positive-part James-Stein shrinkage estimate |
| `c_shrink` | `()` | `f32` | Scalar | James-Stein shrinkage multiplier $1 - (K-2)\sigma^2/\|y\|^2$ |
| `mu_q`, `sigma_q` | `(4,)` | `f32` | C-Contiguous | Stochastic posterior distribution parameters |
| `kl_div` | `()` | `f32` | Scalar | Relative entropy $\text{KL}(Q \| P)$ |
| `pac_bound` | `()` | `f32` | Scalar | McAllester provable risk upper bound |
| `e_values` | `(8,)` | `f32` | C-Contiguous | Sequential e-value evidence stream |
| `martingale` | `(8,)` | `f32` | C-Contiguous | Cumulative evidence test martingale $M_t = \prod_{s=1}^t e_s$ |
| `ville_threshold` | `()` | `f32` | Scalar | Critical rejection boundary $1/\alpha = 20.0$ |

---

## 4. Multi-Ecosystem Implementation Comparison Matrix

| Feature / Property | PyTorch 2.x (`pytorch2/`) | JAX Functional (`jax/`) | Rust Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Execution Paradigm** | Eager / `torch.compile` | Pure Functional / `jax.jit` | Memory-safe Native Rust | Dual-Engine WebGPU + CPU SIMD |
| **Fisher Curvature** | Autograd Hessian diagonal | Forward/Reverse VJP & JVP | Analytical vector ops | High-throughput vectorized reductions |
| **Bayes Conjugate** | Analytical closed form | Pure tensor transformation | Zero-allocation tensor buffers | Cache-resident register updates |
| **James-Stein** | Broadcast scaling + clamp | `jnp.maximum` vector ops | Vectorized scalar broadcast | Parallel WGSL storage buffer compute |
| **PAC-Bayes** | Closed-form Gaussian KL | Functional KL evaluation | Memory-safe loop reduction | Parallel reduction primitives |
| **E-Values Martingale** | `torch.cumprod` | `jnp.cumprod` | Vector scan accumulator | Cumulative scan shader / SIMD |
| **Parity Tolerance** | $< 10^{-5}$ max-norm | $< 10^{-5}$ max-norm | $< 10^{-5}$ max-norm | $< 10^{-5}$ max-norm |

---

## 5. Pathological Failure Modes & Misuse Patterns

1. **Misinterpreting Confidence as Credibility:**
   - *Failure:* Treating Frequentist 95% CI as posterior probability of parameter containment.
   - *Fix:* Use Bayesian posterior credible intervals when parameter belief distributions are required.
2. **MLE Model Selection Catastrophe:**
   - *Failure:* Using maximum likelihood to compare models with different parameter counts $D_1 < D_2$.
   - *Fix:* Use Bayesian Marginal Likelihood with Occam penalty or Minimum Description Length (BIC/WAIC).
3. **Optional Stopping P-Hacking:**
   - *Failure:* Halting an experiment as soon as a classical $p$-value falls below $0.05$.
   - *Fix:* Use anytime-valid E-values with Ville's inequality ($M_t \ge 1/\alpha$).
4. **Applying James-Stein for $K < 3$:**
   - *Failure:* Shrinking 1D or 2D estimators ($K \in \{1, 2\}$); shrinkage factor $1 - (K-2)\sigma^2/\|y\|^2$ diverges or increases MSE.
   - *Fix:* Restrict Stein shrinkage strictly to dimensions $K \ge 3$.
5. **Prior Misspecification in PAC-Bayes:**
   - *Failure:* Using a data-dependent prior $P$ without sample splitting, invalidating the PAC-Bayes bound.
   - *Fix:* Ensure prior $P$ is fixed prior to seeing test data or derived from an independent training split.

---

## 6. Cross-Domain Generalisation & Scientific Isomorphisms

The 5 schools of inference translate across all scientific domains:
- **Vision:** James-Stein shrinkage across multi-class detection heads pools variance across rare classes.
- **Medical Imaging:** Chow rejection and Bayes risk handle high-cost false negatives in tumor detection.
- **Robotics & Control:** Bayesian conjugate updates provide online Kalman gain parameter updates.
- **Natural Language Processing:** PAC-Bayes generalization bounds guide stochastic weight averaging and DP-SGD safety bounds.
- **Quantitative Finance:** Anytime-valid martingales detect market regime shifts without false-alarm rate inflation.
- **Bioinformatics:** Empirical Bayes shrinkage stabilizes gene expression analysis across thousands of microarray targets.
- **Physics-Informed ML:** Laplace Occam factor selects minimal PDE operator basis functions from noisy observational data.
- **Audio & Speech Processing:** E-values monitor streaming keyword spotting false-trigger rates with strict Type-I guarantees.

---

## 7. Verification, Parity & Benchmarking Protocol

- **Golden Reference Checkpoint:** `weights/01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.safetensors`
- **Tolerances:** Absolute difference $\|Y - Y_{\text{ref}}\|_\infty < 10^{-5}$ across all tracks.
- **Performance Characteristics:**
  - Frequentist MLE & LRT: $\mathcal{O}(ND)$ FLOPs over $\mathcal{O}(ND)$ bytes ($\approx 1\text{--}2 \text{ FLOP/Byte}$, memory bound).
  - Bayesian Conjugate Update: $\mathcal{O}(D)$ FLOPs over $\mathcal{O}(D)$ bytes ($\mathcal{O}(1)$ register execution).
  - James-Stein Shrinkage: $\mathcal{O}(K)$ FLOPs for 2-norm reduction and broadcast scale.
  - Evaluation Latency: Sub-second (< 3.0s ceiling) in both CPU SIMD and GPU WGSL modes.

---

## 8. Navigation & Synchronized Track Artifacts

- [Interactive Concept Explorer Notebook](../../notebooks/01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.ipynb)
- [Production PyTorch 2.x Module](../../pytorch2/01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.py)
- [Functional JAX Module](../../jax/01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.py)
- [Native Rust Candle Module](../../candle/01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.rs)
- [Hardware Native Kernel](../../kernels/01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.rs)
- [Golden SafeTensors Weights](../../weights/01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.safetensors)
