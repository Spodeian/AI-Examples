# Privacy in Machine Learning: DP-SGD, PATE, Gradient Inversion (DLG), and Membership Inference (LiRA)

## 1. Mathematical Foundations & Differential Privacy

Machine learning models trained on sensitive data (e.g., healthcare records, financial history, private communications) risk memorizing and leaking training samples:
1. **$(\epsilon, \delta)$-Differential Privacy (Dwork et al. 2006)**: A randomized mechanism $\mathcal{M}$ provides $(\epsilon, \delta)$-DP if for all neighboring datasets $D, D'$ differing on at most one individual, and for all measurable subsets of outcomes $\mathcal{S} \subseteq \text{Range}(\mathcal{M})$:
   $$\mathbb{P}[\mathcal{M}(D) \in \mathcal{S}] \le e^\epsilon \, \mathbb{P}[\mathcal{M}(D') \in \mathcal{S}] + \delta$$
2. **Differentially Private Stochastic Gradient Descent (DP-SGD — Abadi et al. 2016)**: Modifies SGD by clipping per-sample gradient vectors to a maximum $L_2$ norm $C$ and adding calibrated isotropic Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ to the minibatch sum.
3. **Private Aggregation of Teacher Ensembles (PATE — Papernot et al. 2017, 2018)**: Partitions private data among $K$ isolated teacher models, aggregates their predictions on public query data using the Laplace or Gaussian mechanism, and trains a student model exclusively on noisy aggregated labels.
4. **Deep Leakage from Gradients (DLG — Zhu et al. 2019; Geiping et al. 2020)**: Reconstructs private training images and labels with pixel-level fidelity from shared gradient updates by optimizing dummy inputs to match observed gradients.
5. **Likelihood-Ratio Membership Inference Attacks (LiRA — Carlini et al. 2022)**: Evaluates whether a specific target record was included in a model's private training set by comparing target loss against parametric Gaussian distributions fitted over shadow models trained with and without the record.

---

## 2. DP-SGD: Per-Sample Clipping and Gaussian Mechanism

At step $t$ with Poisson or uniform subsampled batch $\mathcal{B} \subset \{1, \dots, N\}$ of size $B$:
1. **Per-Sample Gradient Computation**:
   $$g_i(\theta_t) = \nabla_{\theta_t} \mathcal{L}(f(x_i; \theta_t), y_i) \quad \forall i \in \mathcal{B}$$
2. **Clipping to $L_2$ Radius $C$**:
   $$\bar{g}_i(\theta_t) = g_i(\theta_t) \cdot \min\left(1, \frac{C}{\|g_i(\theta_t)\|_2 + 10^{-8}}\right)$$
3. **Noisy Sum & Batch Averaging**:
   $$\tilde{g}(\theta_t) = \frac{1}{B} \left( \sum_{i \in \mathcal{B}} \bar{g}_i(\theta_t) + \sigma C \, \xi \right), \quad \xi \sim \mathcal{N}(0, I_D)$$
4. **Parameter Update**:
   $$\theta_{t+1} = \theta_t - \eta_t \tilde{g}(\theta_t)$$
Privacy accounting across $T$ iterations is tracked via the Moments Accountant or Rényi Differential Privacy (RDP).

---

## 3. PATE: Private Aggregation of Teacher Ensembles

Data $D$ is split into $K$ disjoint partitions $D_1, \dots, D_K$. Each teacher $f_k$ is trained independently on $D_k$.
For an unlabeled query sample $x$:
1. **Vote Count**:
   $$n_j(x) = \sum_{k=1}^K \mathbb{I}[f_k(x) = j] \quad \text{for class } j \in \{1, \dots, C\}$$
2. **Laplace Perturbation**:
   $$\tilde{n}_j(x) = n_j(x) + \text{Laplace}\left(0, \frac{1}{\lambda}\right)$$
3. **Consensus Prediction**:
   $$y_{\text{student}}(x) = \arg\max_{j} \tilde{n}_j(x)$$
Because each individual sample resides in only one teacher's training set, changing one training sample alters the vote count vector by at most $\ell_1$ sensitivity $\Delta = 1$, bounding the privacy leakage of each answered query.

---

## 4. Privacy Auditing: Gradient Inversion & Membership Inference

### Deep Leakage from Gradients (DLG)
Given shared parameter gradient $g^* = \nabla_\theta \mathcal{L}(x^*, y^*)$:
$$\min_{x', y'} \mathcal{L}_{\text{grad}}(x', y') = \| \nabla_\theta \mathcal{L}(x', y') - g^* \|_2^2 + \lambda_{\text{TV}} \text{TV}(x')$$
Optimized via L-BFGS or Adam. Under small batch sizes and unperturbed gradients, exact input recovery is achievable within several hundred iterations.

### Likelihood Ratio Attack (LiRA)
Train $K$ independent shadow models on random subsets excluding target sample $(x, y)$. Let $\ell_1, \dots, \ell_K$ be their cross-entropy losses on $(x, y)$:
$$\mu_{\text{out}} = \frac{1}{K} \sum_{k=1}^K \ell_k, \quad \sigma_{\text{out}} = \sqrt{\frac{1}{K} \sum_{k=1}^K (\ell_k - \mu_{\text{out}})^2 + 10^{-8}}$$
For a target model loss $\ell_{\text{target}}$:
$$\Lambda(x, y) = \frac{\mu_{\text{out}} - \ell_{\text{target}}}{\sigma_{\text{out}}}$$
A high positive score $\Lambda \gg 0$ indicates that the model has memorized sample $(x, y)$ beyond what is explained by general distribution loss.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`dp_sgd_clip_and_noise`** | Vectorized norm & clamp | `jnp.linalg.norm` clamp | Vector loop norm & scale | `DP_SGD_PER_SAMPLE_CLIP_WGSL` |
| **`pate_noisy_aggregation`** | Additive Laplace noise | Functional noise add | Vector Laplace broadcast | Elementwise adder |
| **`dlg_gradient_inversion_loss`** | $L_2$ squared difference | `jnp.sum(diff**2)` | Squared difference sum | Vector norm reducer |
| **`lira_membership_score`** | Batch mean & std normalization | Vectorized z-score | Vector loop mean & std | Fused z-score normalizer |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Gradient Norm Explosion without Per-Sample Clipping**:
   - *Symptom*: Applying batch-level clipping instead of per-sample clipping fails to guarantee $(\epsilon, \delta)$-DP bounds.
   - *Cause*: A single outlier sample with enormous gradient norm $\|g_i\|_2 \gg C$ dominates the batch sum, violating sensitivity assumptions.
   - *Fix*: Strictly enforce per-sample clipping prior to summation across the batch dimension.
2. **Subsampling Amplification Degradation**:
   - *Symptom*: Rapid exhaustion of privacy budget $\epsilon$.
   - *Cause*: Shuffled fixed-size minibatches without Poisson subsampling violate privacy amplification by subsampling theorems.
   - *Fix*: Employ Poisson sample drawing with probability $q = B / N$.
3. **Numerical Instability in Gaussian Standard Deviation Normalization**:
   - *Symptom*: NaN/Inf scores during LiRA evaluation on confident samples.
   - *Cause*: Zero variance across shadow model losses on canonical easy samples ($\sigma_{\text{out}} \approx 0$).
   - *Fix*: Add stabilizing jitter $\epsilon = 10^{-8}$ inside the square root denominator.

---

## 7. Real-World Engineering Applications

1. **Federated Clinical Model Training**: Protecting electronic health records (EHR) against gradient reconstruction attacks during distributed hospital training.
2. **On-Device Next-Word Prediction**: Training user keyboard language models with local or central DP-SGD to prevent credit card and password memorization.
3. **Confidential LLM Fine-Tuning**: Using DP-SGD fine-tuning to satisfy regulatory standards (HIPAA, GDPR) when adapting foundation models on enterprise customer service tickets.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.safetensors`:
1. `dp_grad_ref`: DP-SGD clipped and noised minibatch gradient vector.
2. `pate_votes_ref`: PATE Laplace-perturbed teacher consensus vote counts.
3. `dlg_loss_ref`: Deep Leakage from Gradients $L_2$ matching loss.
4. `lira_scores_ref`: LiRA parametric likelihood-ratio membership scores.

### Synchronized Ecosystem Artifacts
- **Notebook:** [07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.ipynb](../../notebooks/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.ipynb)
- **PyTorch 2 Module:** [07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.py](../../pytorch2/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.py)
- **JAX Module:** [07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.py](../../jax/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.py)
- **Candle Module:** [07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.rs](../../candle/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.rs)
- **Hardware Kernel:** [07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.rs](../../kernels/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.rs)
