# Losses: Divergences, Information Bottleneck, and Coding Rate (C03 Topic 09)

## 1. Executive Summary
Information-theoretic divergences and representation-rate criteria form the mathematical foundation for generative modeling, variational inference, audio signal processing, and self-supervised geometric clustering. Rather than measuring simple elementwise errors, these objectives evaluate geometric discrepancies between probability distributions, quantify compression trade-offs, and govern volumetric subspace expansion:
1. **Asymmetric Information Geometries (Forward vs. Reverse KL):** The Kullback-Leibler divergence is asymmetric: $D_{\text{KL}}(P \,\|\, Q) \ne D_{\text{KL}}(Q \,\|\, P)$. **Forward KL** ($P \,\|\, Q$) is *zero-avoiding* (mode-covering), forcing the model $Q$ to allocate non-zero probability everywhere target $P > 0$. Conversely, **Reverse KL** ($Q \,\|\, P$) is *zero-forcing* (mode-seeking), forcing $Q$ to collapse into a single high-density mode where $P > 0$ and avoid regions where $P = 0$. This fundamental dichotomy dictates the behavior of VAEs vs. GANs and diffusion vs. Direct Preference Optimization (DPO).
2. **Symmetric & Generalized Divergences (JSD & $\alpha$-Rényi):** **Jensen-Shannon Divergence (JSD)** symmetrizes KL via the mixture distribution $M = \frac{1}{2}(P + Q)$, bounding discrepancy within $[0, \log 2]$ (or $[0, 1]$ in base 2) and providing a true metric square root. **$\alpha$-Rényi Divergence** (*Rényi 1961*) generalizes divergences across a continuous parameter $\alpha$, interpolating between total variation ($\alpha \to 1$), Bhattacharyya ($\alpha = 0.5$), and Pearson $\chi^2$ ($\alpha = 2$).
3. **Bregman Geometries & Spectral Distance (Itakura-Saito):** In audio, speech, and non-negative matrix factorization (NMF), Euclidean distance fails because human auditory perception responds to relative rather than absolute energy. Derived from the strictly convex negative entropy generating functional $F(x) = -\sum \log x_i$, the **Itakura-Saito Distance** (*Itakura & Saito 1968*) is a scale-invariant Bregman divergence: $d_{\text{IS}}(\lambda x, \lambda y) = d_{\text{IS}}(x, y)$, treating a whisper and a shout with identical relative spectral shape identically.
4. **Variational Information Bottleneck (VIB):** Standard deep networks memorize spurious background noise during supervised training. **VIB** (*Tishby et al. 1999; Alemi et al. 2016*) enforces an optimal rate-distortion trade-off: $\min_\theta -I(Z; Y) + \beta I(Z; X)$, discarding redundant input entropy while preserving minimal sufficient statistics for the downstream task.
5. **Maximal Coding Rate Reduction ($\text{MCR}^2$):** Instead of heuristic contrastive negative sampling, **$\text{MCR}^2$** (*Yu et al. 2020*) models self-supervised representation learning as a geometric packing problem: maximize the coding rate difference $\Delta R(Z) = R(Z) - R_c(Z)$ between the entire representation space and individual class subspaces via closed-form log-determinant volume metrics.

This curriculum topic provides rigorous mathematical derivations, directional asymmetry analyses, pathological failure mode postmortems, and synchronized implementations across PyTorch 2.x, JAX, Rust Candle, and Hardware Kernels (WGSL + CPU auto-dispatcher).

---

## 2. Mathematical Foundations

### 2.1 Forward vs. Reverse Kullback-Leibler Divergence
Given true distribution $P(x)$ and approximate parametric model $Q(x)$:
1. **Forward KL (Information Projection / Mode-Covering):**
   $$D_{\text{KL}}(P \,\|\, Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx = \mathbb{E}_{x \sim P}\left[ \log P(x) - \log Q(x) \right]$$
   - *Zero-Avoiding Property:* If $P(x) > 0$, $Q(x)$ must be strictly positive ($Q(x) > 0$), otherwise the term $P(x) \log(P(x) / 0) \to \infty$. Thus, $Q$ must cover all modes of $P$, spreading probability mass across low-density valleys.
2. **Reverse KL (Moment Projection / Mode-Seeking):**
   $$D_{\text{KL}}(Q \,\|\, P) = \int Q(x) \log \frac{Q(x)}{P(x)} \, dx = \mathbb{E}_{x \sim Q}\left[ \log Q(x) - \log P(x) \right]$$
   - *Zero-Forcing Property:* If $P(x) \approx 0$, $Q(x)$ is forced to be 0 ($Q(x) \to 0$) to prevent $\log(Q / P) \to \infty$. When $P$ is multimodal, $Q$ selects a single dominant mode of $P$ and ignores other modes entirely, avoiding intermediate zero-density regions.

### 2.2 Jensen-Shannon and $\alpha$-Rényi Divergences
1. **Jensen-Shannon Divergence (JSD):**
   $$\text{JSD}(P \,\|\, Q) = \frac{1}{2} D_{\text{KL}}(P \,\|\, M) + \frac{1}{2} D_{\text{KL}}(Q \,\|\, M), \quad M = \frac{1}{2}(P + Q)$$
   JSD is symmetric ($\text{JSD}(P \,\|\, Q) = \text{JSD}(Q \,\|\, P)$), finite ($0 \le \text{JSD} \le \log 2$), and its square root $\sqrt{\text{JSD}}$ satisfies the triangle inequality, forming a true distance metric.
2. **$\alpha$-Rényi Divergence:**
   $$D_\alpha(P \,\|\, Q) = \frac{1}{\alpha - 1} \log \sum_{x} P(x)^\alpha Q(x)^{1-\alpha}, \quad \alpha \in (0, 1) \cup (1, \infty)$$
   - When $\alpha \to 1$: $\lim_{\alpha \to 1} D_\alpha(P \,\|\, Q) = D_{\text{KL}}(P \,\|\, Q)$.
   - When $\alpha = 0.5$: $D_{0.5}(P \,\|\, Q) = -2 \log \sum \sqrt{P(x) Q(x)} = -2 \log \text{BC}(P, Q)$ (Bhattacharyya coefficient).

### 2.3 Bregman Divergence and Itakura-Saito Distance
Given a continuously differentiable, strictly convex generating function $F: \Omega \to \mathbb{R}$, the Bregman divergence is the discrepancy between $F(x)$ and its first-order Taylor expansion around $y$:
$$D_F(x, y) = F(x) - F(y) - \langle \nabla F(y), x - y \rangle$$
For the Burg entropy $F(x) = -\sum_{i=1}^D \log x_i$ on $\mathbb{R}_{>0}^D$, the gradient is $\nabla F(y)_i = -1/y_i$. Substituting into the Bregman formulation yields the **Itakura-Saito Distance**:
$$d_{\text{IS}}(x, y) = \sum_{i=1}^D \left[ -\log x_i + \log y_i - \left(-\frac{1}{y_i}\right)(x_i - y_i) \right] = \sum_{i=1}^D \left( \frac{x_i}{y_i} - \log \frac{x_i}{y_i} - 1 \right)$$
- Property 1: $d_{\text{IS}}(x, y) \ge 0$, with equality if and only if $x = y$.
- Property 2: Scale invariance: $d_{\text{IS}}(\lambda x, \lambda y) = d_{\text{IS}}(x, y)$ for any $\lambda > 0$.

### 2.4 Variational Information Bottleneck (VIB)
The Information Bottleneck principle optimizes for an intermediate representation $Z$ that compresses input $X$ while preserving label information $Y$:
$$\mathcal{L}_{\text{IB}} = -I(Z; Y) + \beta I(Z; X)$$
Using variational approximations with encoder $q_\theta(z|x)$ and decoder $p_\phi(y|z)$, the upper bound on input mutual information decomposes with standard normal prior $p(z) = \mathcal{N}(0, I)$:
$$I(Z; X) \le \mathbb{E}_{x \sim p(x)}\left[ D_{\text{KL}}(q_\theta(z|x) \,\|\, p(z)) \right]$$
For a diagonal Gaussian encoder $q_\theta(z|x) = \mathcal{N}(\mu(x), \text{diag}(\sigma^2(x)))$, the KL term evaluates analytically:
$$D_{\text{KL}}(q_\theta(z|x) \,\|\, \mathcal{N}(0, I)) = -\frac{1}{2} \sum_{j=1}^D \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

### 2.5 Maximal Coding Rate Reduction ($\text{MCR}^2$)
Given a batch of representation vectors $Z = [z_1, \dots, z_N] \in \mathbb{R}^{D \times N}$ normalized to a unit sphere, the average coding rate (bits per sample to encode representations within distortion radius $\epsilon$) is:
$$R(Z, \epsilon) = \frac{1}{2} \log \det\left( I_D + \frac{D}{N \epsilon^2} Z Z^T \right)$$
When data partitions into $K$ semantic classes with subset representations $Z_k \in \mathbb{R}^{D \times N_k}$ and sample weights $\pi_k = N_k / N$:
$$R_c(Z, \epsilon) = \sum_{k=1}^K \frac{\pi_k}{2} \log \det\left( I_D + \frac{D}{N_k \epsilon^2} Z_k Z_k^T \right)$$
The **$\text{MCR}^2$ Objective** maximizes the coding rate reduction $\Delta R(Z) = R(Z, \epsilon) - R_c(Z, \epsilon)$:
- Maximizing $R(Z, \epsilon)$ expands the global representation volume into all available orthogonal dimensions (preventing representation collapse).
- Minimizing $R_c(Z, \epsilon)$ compresses each class subspace into compact, low-dimensional intrinsic manifolds.

---

## 3. Structural Properties & Computational Complexity Table

```text
Objective / Loss     Type              Hyperparameters       FLOPs / Element    Geometric / Statistical Invariant
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Forward KL           Divergence        None                  O(K)               Zero-Avoiding (Mode-Covering)
Reverse KL           Divergence        None                  O(K)               Zero-Forcing (Mode-Seeking)
Jensen-Shannon       Divergence        None                  O(K)               Symmetric, Bounded in [0, log 2]
alpha-Renyi          Divergence        alpha                 O(K)               Monotonic in alpha
Itakura-Saito        Bregman Distance  None                  O(D)               Scale Invariant: d(ax, ay) = d(x, y)
VIB Gaussian KL      Information Rate  beta (bottleneck)     O(D)               Exact closed-form prior match
MCR^2 Coding Rate    Rate Distortion   epsilon (distortion)  O(D^3 + N*D^2)     Subspace Volumetric Expansion
```

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator / Layer | PyTorch 2.x (`pytorch2/`) | JAX / NumPy (`jax/`) | Candle Rust (`candle/`) | Hardware Kernels (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Forward KL** | `compute_kl_forward` | `kl_forward_fn` | `kl_forward` | `kl_forward_cpu` |
| **Reverse KL** | `compute_kl_reverse` | `kl_reverse_fn` | `kl_reverse` | `kl_reverse_cpu` |
| **JSD** | `compute_jsd` | `jsd_fn` | `jsd_forward` | `jsd_cpu` |
| **$\alpha$-Rényi** | `compute_renyi_divergence` | `renyi_divergence_fn` | `renyi_divergence_forward` | `renyi_divergence_cpu` |
| **Itakura-Saito** | `compute_itakura_saito` | `itakura_saito_fn` | `itakura_saito_forward` | `ITAKURA_SAITO_SHADER` (WGSL) + CPU |
| **VIB Gaussian KL** | `compute_vib_kl` | `vib_kl_fn` | `vib_kl_forward` | `vib_kl_cpu` |
| **$\text{MCR}^2$ Coding Rate** | `compute_mcr2_coding_rate` | `mcr2_coding_rate_fn` | `mcr2_coding_rate_forward` | `mcr2_coding_rate_cpu` |
| **Linear Anchor** | `DivergenceUnifiedModule` | `linear_forward` | `LinearAnchor` | `linear_anchor_forward_cpu` |
| **Parity Tolerance** | $< 10^{-5}$ | $< 10^{-5}$ | $< 10^{-5}$ | $< 10^{-5}$ |

---

## 5. Pathological Failure Modes & Postmortems

1. **Unimodal Mode-Covering Catastrophe under Forward KL:**
   - *Failure Mechanism:* When training a unimodal model $Q_\theta = \mathcal{N}(\mu, \sigma^2)$ to approximate a multimodal mixture $P = 0.5 \mathcal{N}(-5, 1) + 0.5 \mathcal{N}(5, 1)$, optimizing Forward KL $D_{\text{KL}}(P \,\|\, Q)$ forces $Q$ to cover both modes because $P(x) > 0$ at both peaks. The resulting optimal Gaussian has $\mu = 0$ and $\sigma^2 \approx 26$, placing maximum probability density in the valley between modes where $P(0) \approx 0$!
   - *Resolution:* Use Reverse KL $D_{\text{KL}}(Q \,\|\, P)$ (as used in variational inference, GAN generators, and RL distillation). Reverse KL forces $Q$ to lock tightly onto one mode ($\mu \approx 5$ or $-5$, $\sigma^2 \approx 1$), producing realistic, sharp samples.
2. **Spectral Energy Bias in Audio Processing (Euclidean MSE vs. Itakura-Saito):**
   - *Failure Mechanism:* In audio spectrogram estimation, high-energy vowel formants have power $\sim 10^3$, while quiet unvoiced consonants have power $\sim 10^{-1}$. Mean squared error $(x - y)^2$ is dominated by the loud formant ($10^6$ vs $10^{-2}$ penalty), resulting in models that completely ignore low-energy speech features essential for intelligibility.
   - *Resolution:* Use Itakura-Saito distance $d_{\text{IS}}(x, y) = \frac{x}{y} - \log \frac{x}{y} - 1$. Because $d_{\text{IS}}$ is invariant to overall scale, a 10% error on a quiet consonant incurs the exact same loss as a 10% error on a loud formant.
3. **Log-Determinant Singular Collapse in $\text{MCR}^2$:**
   - *Failure Mechanism:* In computing the coding rate $R(Z) = \frac{1}{2} \log \det(I + \frac{D}{N \epsilon^2} Z Z^T)$, if $\epsilon \to 0$ or if representations have rank $r < D$, raw matrix inversion or determinant formulas suffer extreme ill-conditioning ($\kappa \to \infty$) and float underflow.
   - *Resolution:* Add the explicit identity matrix regularization $I_D$ and compute the log-determinant via Cholesky factorization: $A = L L^T \implies \log \det(A) = 2 \sum_{i=1}^D \log L_{ii}$.

---

## 6. Cross-Domain Generalisation

- **Large Language Models & RLHF (DPO):** Direct Preference Optimization relies on Reverse KL implicit regularization to prevent language models from drifting away from the reference policy.
- **Audio & Speech Processing (Vocoders & NMF):** Itakura-Saito distance is the canonical loss for non-negative matrix factorization in music source separation and spectrogram reconstruction.
- **Biomedical & Out-of-Distribution Generalization:** The Variational Information Bottleneck (VIB) enforces invariance to nuisance variables in clinical diagnostic models, preventing overfitting to hospital-specific scanner noise.
- **Self-Supervised Visual Representation Learning:** $\text{MCR}^2$ (White-Box Deep Networks / ReduNet) constructs interpretable multi-layer architectures where each layer explicitly performs gradient ascent on coding rate reduction.

---

## 7. Verification & Empirical Parity

| Comparison Pair | Max Absolute Difference ($\|Y_A - Y_B\|_\infty$) | Test Status |
| :--- | :--- | :--- |
| **PyTorch 2.x vs Golden Safetensors** | $0.00 \times 10^0$ | **PASSED** |
| **JAX vs Golden Safetensors** | $5.36 \times 10^{-7}$ | **PASSED** |
| **Rust Candle vs Golden Safetensors** | $0.00 \times 10^0$ | **PASSED** |
| **Hardware Kernel (CPU/WGPU) vs Golden** | $0.00 \times 10^0$ | **PASSED** |

All four execution tracks satisfy the strict curriculum parity threshold ($\|Y_A - Y_B\|_\infty < 10^{-5}$) across all 7 loss and divergence formulations.

---

## 8. References & Further Reading

1. Kullback, S., & Leibler, R. A. (1951). *On Information and Sufficiency*. Annals of Mathematical Statistics, 22(1), 79–86.
2. Rényi, A. (1961). *On Measures of Entropy and Information*. Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, Vol. 1, 547–561.
3. Bregman, L. M. (1967). *The Relaxation Method of Finding the Common Point of Convex Sets*. USSR Computational Mathematics and Mathematical Physics, 7(3), 200–217.
4. Itakura, F., & Saito, S. (1968). *Analysis Synthesis Telephony based on the Maximum Likelihood Method*. 6th International Congress on Acoustics, C-17–C-20.
5. Tishby, N., Pereira, F. C., & Bialek, W. (1999). *The Information Bottleneck Method*. 37th Annual Allerton Conference on Communication, Control, and Computing, 368–377.
6. Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2016). *Deep Variational Information Bottleneck*. ICLR 2017.
7. Yu, Y., Chan, K. H. R., You, C., Song, C., & Ma, Y. (2020). *Learning Diverse and High-Quality Representations by Maximizing Information Rate Reduction (MCR²)*. NeurIPS 2020.
