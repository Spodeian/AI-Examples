# AI Watermarking, AI Detection, and Model Obfuscation

## 1. Mathematical Foundations & Lineage

As generative models approach human parity, provenance verification, synthetic content detection, and intellectual property (IP) protection become critical safety infrastructure:
1. **Spread-Spectrum Watermarking (Cox et al. 1997; Uchida et al. 2017)**: Embedded secret signatures into spectral audio/image bands or neural network weights via regularization losses without affecting primary task utility.
2. **Green/Red-List Sequence Watermarking (Kirchenbauer et al. 2023)**: Injected soft statistical watermarks into autoregressive text generation by partitioning vocabulary into pseudo-random "green" and "red" lists conditioned on preceding tokens, biasing logits towards green tokens.
3. **Distortion-Free Gumbel & Tournament Watermarks (Aaronson 2023; Dathathri et al. 2024 / SynthID)**: Preserved true predictive distribution entropy across text, code, and multimodal generation while embedding verifiable pseudorandom permutations.
4. **Zero-Shot AI Detection (Mitchell et al. 2023 / DetectGPT; Bao et al. 2023 / Fast-DetectGPT)**: Identified machine-generated text by detecting negative log-likelihood curvature; machine text tends to occupy local probability modes, exhibiting sharp negative curvature under perturbations that human text lacks.
5. **Passport Layers & Intellectual Property Obfuscation (Fan et al. 2019)**: Integrated secret cryptographic keys into normalization scale and bias parameters, rendering neural networks functionally unusable (collapsing to random chance) unless authorized keys are supplied at inference time.

---

## 2. Multi-Modal Watermarking: Logit Biasing and Statistical Detection

### Kirchenbauer Green/Red-List Mechanism
At autoregressive step $t$, given preceding token $x_{t-1}$ with hash seed $h(x_{t-1})$:
1. Partition vocabulary $V$ into green list $G$ of size $\gamma |V|$ and red list $R$ of size $(1-\gamma)|V|$ (typically $\gamma = 0.5$).
2. Apply logit bias $\delta > 0$ to green-list tokens:
   $$l'_v = \begin{cases} l_v + \delta & \text{if } v \in G \\ l_v & \text{if } v \in R \end{cases}$$
3. Sample token $x_t \sim \text{softmax}(l')$.

### One-Tailed $z$-Score Hypothesis Testing
Given a candidate sequence of length $T$ with $|S_G|$ green tokens:
Under the null hypothesis $H_0$ (human un-watermarked text), $|S_G| \sim \text{Binomial}(T, \gamma)$ with mean $\mu = \gamma T$ and variance $\sigma^2 = T \gamma (1 - \gamma)$.
The one-tailed $z$-score is:
$$z = \frac{|S_G| - \gamma T}{\sqrt{T \gamma (1 - \gamma) + 10^{-8}}}$$
A critical value $z > 4.0$ yields a false-positive p-value $p < 3.2 \times 10^{-5}$, providing statistical certainty of synthetic origin.

---

## 3. Zero-Shot AI Detection: Fast-DetectGPT Curvature

Standard DetectGPT requires generating $M \approx 100$ perturbations per candidate text via masked language models (T5), incurring significant latency.
**Fast-DetectGPT** evaluates conditional probability curvature in a **single forward pass** by comparing the target model's log-probability against an analytical reference distribution or sampled expectations:
$$\mathbf{d}(x) = \frac{1}{T} \sum_{t=1}^T \left( \log p_\theta(x_t \mid x_{<t}) - \log p_{\text{ref}}(x_t \mid x_{<t}) \right)$$
Because machine text is generated directly from the target model's logits, its average log-likelihood significantly exceeds reference baseline expectations ($\mathbf{d}(x) \gg 0$), yielding a $340\times$ inference speedup.

---

## 4. Model Obfuscation & IP Protection: Passport Layers

To prevent unauthorized deployment, distillation, or theft of proprietary foundation model weights, Passport Layers gate internal representations behind cryptographic keys:
1. **Key-Conditioned Affine Scaling**:
   $$s = \sigma(K \cdot W_{\text{pass}}), \quad s \in (0, 1)^D$$
   where $K \in \mathbb{R}^{D_k}$ is the private passport key and $W_{\text{pass}}$ is a learned projection matrix.
2. **Layer Transformation**:
   $$y = (x W + b) \odot s$$
3. **Security Invariant**:
   - Under authorized key $K_{\text{auth}}$: $s \approx 1$, allowing normal network activations to propagate without performance degradation.
   - Under unauthorized or random key $K_{\text{fake}}$: gating factors $s$ scramble features across channels, dropping classification accuracy to random guessing ($\sim 10\%$).

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`watermark_logit_bias`** | Broadcast mask add | `jnp.expand_dims` add | Vector loop add | `WATERMARK_LOGIT_BIAS_WGSL` |
| **`watermark_z_score`** | Vectorized binomial z-score | Vectorized z-score | Vector loop sqrt & div | Fused z-score evaluator |
| **`fast_detect_gpt`** | Difference mean across seq | `jnp.mean` across seq | Vector loop mean | Mean discrepancy reducer |
| **`passport_layer`** | Sigmoid dot + elementwise | `jax.nn.sigmoid` + mul | Vectorized dot & sigmoid | Fused passport gate |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Syntax Corruption from High Logit Bias on Low-Entropy Tokens**:
   - *Symptom*: Watermarked code generators output invalid syntax, broken brackets, or invalid chemical SMILES formulas.
   - *Cause*: Constant bias $\delta \gg 0$ applied to deterministic syntax tokens where the red-list token was mandatory.
   - *Fix*: Context-aware adaptive biasing: scale $\delta_t = \delta \cdot \mathbb{I}[H(p_t) > \tau]$, suppressing watermarking on low-entropy tokens ($H < \tau$).
2. **Paraphrase & Translation Scrubbing**:
   - *Symptom*: Adversaries evade watermarking detection by passing text through light machine translation or paraphrasing models.
   - *Cause*: Green/red lists depend on exact previous token hashes $h(x_{t-1})$; localized edits break hash alignment.
   - *Fix*: Robust window hashing $h(x_{t-k}, \dots, x_{t-1})$ or semantic embedding watermarks.
3. **Passport Key Forgery via Optimization**:
   - *Symptom*: An attacker recovers a valid passport key by minimizing cross-entropy loss on public validation data.
   - *Cause*: Passport layers placed only in the final linear head with low key dimensionality $D_k$.
   - *Fix*: Distribute passport layers across multiple deep residual blocks and enforce sign-loss constraints.

---

## 7. Real-World Engineering Applications

1. **Regulatory Compliance (EU AI Act & US Executive Order)**: Machine-generated content provenance and synthetic media tagging.
2. **Educational Integrity & Plagiarism Detection**: Distinguishing AI-assisted student essays from human writing via calibrated statistical $z$-scores.
3. **Model Weights IP Licensing**: Shipping on-premise foundation models to enterprise customers with cryptographic passport dongles.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.safetensors`:
1. `watermarked_logits_ref`: Biased token logits with green-list augmentation.
2. `z_scores_ref`: One-tailed hypothesis testing $z$-scores.
3. `detect_scores_ref`: Fast-DetectGPT single-pass likelihood discrepancy scores.
4. `passport_out_ref`: Passport key-gated representation vectors.

### Synchronized Ecosystem Artifacts
- **Notebook:** [08_ai_watermarking_ai_detection_and_model_obfuscation.ipynb](../../notebooks/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.ipynb)
- **PyTorch 2 Module:** [08_ai_watermarking_ai_detection_and_model_obfuscation.py](../../pytorch2/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.py)
- **JAX Module:** [08_ai_watermarking_ai_detection_and_model_obfuscation.py](../../jax/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.py)
- **Candle Module:** [08_ai_watermarking_ai_detection_and_model_obfuscation.rs](../../candle/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.rs)
- **Hardware Kernel:** [08_ai_watermarking_ai_detection_and_model_obfuscation.rs](../../kernels/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.rs)
