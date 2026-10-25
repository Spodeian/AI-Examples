# Backdoor Trojan and Poisoning Defense: Neural Cleanse and Spectral Signatures

## 1. Mathematical Foundations & Lineage

Data poisoning and trojan backdoor attacks threaten the supply chain of modern deep learning. An adversary alters a fraction of the training data or pre-trained weights such that the model operates with near-perfect accuracy on clean validation inputs, but misclassifies any sample stamped with a secret trigger into a targeted class:
1. **BadNets & Trigger Injection (Gu et al. 2017; Chen et al. 2017)**: Demonstrated that injecting tiny localized patch triggers ($x' = x \odot (1-m) + p \odot m$) into $1\%$ of training data reliably implants backdoors that survive general training.
2. **Spectral Signatures (Tran et al. 2018)**: Identified that clean-label data poisoning necessarily shifts internal representation distributions along the principal singular vector of the covariance matrix, enabling singular value decomposition (SVD) filtering in $\mathcal{O}(N \cdot d)$ time.
3. **Neural Cleanse (Wang et al. 2019)**: First model inspection technique capable of reverse-engineering unknown trigger patterns from model weights alone without access to the poisoned training set, using $L_1$ optimization and Median Absolute Deviation (MAD) anomaly testing.
4. **STRIP (Strong Intentional Perturbation — Gao et al. 2019)**: Runtime defense that superposes candidate test inputs onto diverse benign images: trojaned inputs maintain near-zero Shannon entropy ($H \approx 0$) invariant to background perturbations, while clean inputs yield high entropy ($H \gg 1.0$).
5. **Fine-Pruning (Liu et al. 2018)**: Demonstrated that backdoor neurons remain dormant during clean inference and can be neutralized by pruning inactive channels on validation sets.

---

## 2. Spectral Signatures for Data Sanitization

For samples in target class $c$ with representations $H \in \mathbb{R}^{N \times D}$:
1. Center representations: $\bar{H} = H - \frac{1}{N}\sum_i H_i$.
2. Compute sample covariance matrix $\Sigma = \frac{1}{N} \bar{H}^\top \bar{H} \in \mathbb{R}^{D \times D}$.
3. Extract top right-singular vector $v_1$ (via power iteration or SVD):
   $$\Sigma v_1 = \lambda_{\max} v_1$$
4. Compute outlier score for sample $i$:
   $$\text{score}_i = \left( (H_i - \mu) \cdot v_1 \right)^2$$
Samples with top scores are identified as poisoned and removed prior to model training.

---

## 3. Reverse Engineering via Neural Cleanse

Neural Cleanse solves a multi-objective optimization for each candidate target class $c \in \{1, \dots, C\}$:
$$\min_{m_c, p_c} \frac{1}{|D|} \sum_{x \in D} \mathcal{L}_{\text{CE}}(f(x \odot (1 - m_c) + p_c \odot m_c), c) + \lambda \|m_c\|_1$$
Let $s_c = \|m_c\|_1$ be the minimum trigger size required to convert all clean inputs into class $c$.

### Median Absolute Deviation (MAD) Anomaly Index
Backdoored classes require significantly smaller trigger masks than uninfected classes:
$$\text{med} = \text{median}(s), \quad \text{MAD} = \text{median}(|s - \text{med}|) + \epsilon$$
$$\text{Anomaly}_c = \frac{\text{med} - s_c}{1.4826 \cdot \text{MAD}}$$
An anomaly score $\text{Anomaly}_c > 2.0$ identifies class $c$ as an infected backdoor target!

---

## 4. Runtime Trojan Detection via STRIP

At inference time, test sample $x$ is blended with $K$ random clean validation samples $x_k$:
$$\tilde{x}_k = x \cdot (1 - \alpha) + x_k \cdot \alpha$$
Compute predicted probability distribution $P_k = \text{softmax}(f(\tilde{x}_k))$ and evaluate Shannon entropy:
$$H_k = -\sum_{c=1}^C P_{k, c} \ln(P_{k, c} + \epsilon), \quad \bar{H} = \frac{1}{K}\sum_{k=1}^K H_k$$
If $\bar{H} < \tau_{\text{entropy}}$ (typically $\bar{H} \approx 0$), the input contains a trojan trigger and is rejected.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`spectral_scores`** | Covariance power iteration | Functional SVD projection | Vector covariance loop | Parallel SVD reducer |
| **`neural_cleanse_mad`** | Median & MAD anomaly | `jnp.median` anomaly | Vector sort median & MAD | Vectorized median finder |
| **`strip_entropy`** | Softmax + entropy mean | `jax.nn.softmax` entropy | Fused logsumexp entropy | Fast entropy reducer |
| **`trojan_trigger`** | Elementwise blend | Functional broadcast blend | Vectorized blend iterator | `TROJAN_TRIGGER_APPLY_WGSL` |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Dormant Backdoors Surviving Clean Fine-Tuning**:
   - *Symptom*: Fine-tuning a backdoored model on a clean dataset fails to eliminate the trigger (attack success rate remains $>98\%$).
   - *Cause*: Trojan neurons receive zero gradient activations on clean inputs and remain dormant in weight space.
   - *Fix*: Apply Fine-Pruning to eliminate neurons with zero clean validation activation before fine-tuning.
2. **Blended Low-Magnitude Spectral Evasion**:
   - *Symptom*: Spectral signatures fail to detect poisoned clean-label samples.
   - *Cause*: Poisoning noise was dispersed across multiple orthogonal singular directions with small norm.
   - *Fix*: Multi-vector spectral analysis and activation clustering across intermediate layers.
3. **STRIP Entropy False Alarms on Low-Entropy Inputs**:
   - *Symptom*: Clean inputs with high confident predictions are falsely flagged as trojans.
   - *Cause*: Blending coefficient $\alpha$ too small, or base input features completely dominate perturbation.
   - *Fix*: Calibrate $\alpha = 0.5$ and use temperature-scaled soft labels.

---

## 7. Real-World Engineering Applications

1. **Supply-Chain Foundation Model Vetting**: Scanning third-party open-source checkpoints for secret backdoor triggers before enterprise deployment.
2. **Autonomous Driving Vision Auditing**: Verifying perception neural networks against physical sticker triggers on traffic stop signs.
3. **Medical Imaging Dataset Cleansing**: Sanitizing radiological image repositories against clean-label poisoning attacks prior to training diagnostic classifiers.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.safetensors`:
1. `spectral_scores_ref`: Covariance SVD spectral outlier scores.
2. `mad_anomaly_ref`: Neural Cleanse MAD anomaly index vector.
3. `entropy_benign_ref`: STRIP entropy on clean benign inputs.
4. `entropy_trojan_ref`: STRIP entropy on trojan triggered inputs.
5. `x_trojan_ref`: Perturbed input vector with blended trojan trigger.

### Synchronized Ecosystem Artifacts
- **Notebook:** [06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.ipynb](../../notebooks/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.ipynb)
- **PyTorch 2 Module:** [06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.py](../../pytorch2/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.py)
- **JAX Module:** [06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.py](../../jax/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.py)
- **Candle Module:** [06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.rs](../../candle/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.rs)
- **Hardware Kernel:** [06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.rs](../../kernels/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.rs)
