# Machine Unlearning: SISA, SCRUB, NPO, and RMU

## 1. Mathematical Foundations & Lineage

The increasing regulatory requirements (e.g. EU GDPR "Right to be Forgotten", US FTC algorithmic disgorgement) and AI safety mandates (e.g. removing CBRN biological, chemical, radiological, and cyber weapons knowledge from foundation models) require selectively erasing specific training data or concept capabilities without retraining models from scratch:
1. **SISA (Sharded, Isolated, Sliced, Aggregated Training — Bourtoule et al. 2021)**: Partitions the training corpus into $S$ isolated shards and $R$ slices. When an unlearning request arrives, only the single affected shard is retrained from the closest slice checkpoint, reducing retraining compute by up to $\mathcal{O}(S \cdot R)$.
2. **SCRUB (Student-Teacher Unlearning via Minimax — Kurmanji et al. 2023)**: Formulates unlearning as a minimax game between student and teacher models, maximizing loss on the forget set while minimizing KL divergence to the frozen teacher on the retain set.
3. **Negative Preference Optimization (NPO — Zhang et al. 2024)**: Solves the catastrophic collapse of naive Gradient Ascent ($\min_\theta - \mathcal{L}_{\text{CE}}$) by optimizing a bounded preference loss $\mathcal{L}_{\text{NPO}} = \frac{2}{\beta} \text{softplus}\left( \beta \log \frac{\pi_\theta(y_f|x)}{\pi_{\text{ref}}(y_f|x)} \right)$ that naturally vanishes as $\pi_\theta(y_f|x) \to 0$.
4. **RMU (Representation Misdirection for Unlearning — Li et al. 2024 WMDP)**: Steers internal representations of forget-set tokens towards a random Gaussian target vector while anchoring retain-set activations to the reference model, excising hazardous capabilities from deep layers.

---

## 2. Negative Preference Optimization (NPO) Formulation

Naive Gradient Ascent ($-\mathcal{L}_{\text{CE}}(y_f)$) suffers from gradient explosion: as confidence in $y_f$ drops, gradients continue pushing weights into low-entropy degeneracies, destroying general language fluency in fewer than 10 steps.

NPO reformulates unlearning as negative alignment relative to a reference model $\pi_{\text{ref}}$:
$$\mathcal{L}_{\text{NPO}}(\theta) = -\frac{2}{\beta} \log \sigma \left( -\beta \log \frac{\pi_\theta(y_f|x)}{\pi_{\text{ref}}(y_f|x)} \right) = \frac{2}{\beta} \text{softplus} \left( \beta \left( \log \pi_\theta(y_f|x) - \log \pi_{\text{ref}}(y_f|x) \right) \right)$$

### Gradient Dynamics & Vanishing Gradient Property
$$\nabla_\theta \mathcal{L}_{\text{NPO}} = 2 \cdot \sigma \left( \beta \log \frac{\pi_\theta(y_f|x)}{\pi_{\text{ref}}(y_f|x)} \right) \nabla_\theta \log \pi_\theta(y_f|x)$$
When $\pi_\theta(y_f|x) \ll \pi_{\text{ref}}(y_f|x)$, the gating term $\sigma(\cdot) \to 0$, automatically halting optimization and preserving general reasoning capabilities!

---

## 3. Representation Misdirection for Unlearning (RMU)

Instead of altering output tokens alone, RMU surgically neutralizes internal representations at intermediate layer $l$:
$$\mathcal{L}_{\text{RMU}}(\theta) = \frac{1}{|D_f|} \sum_{x \in D_f} \|h_\theta^{(l)}(x) - u_{\text{random}}\|_2^2 + \alpha \cdot \frac{1}{|D_r|} \sum_{x \in D_r} \|h_\theta^{(l)}(x) - h_{\text{ref}}^{(l)}(x)\|_2^2$$
where $u_{\text{random}} \sim \mathcal{N}(0, I)$ is a fixed target vector and $\alpha > 0$ controls retention penalty on benign queries.

---

## 4. Sharded & Minimax Approaches (SISA & SCRUB)

### SISA Sharding & Checkpoint Slicing
Corpus $D$ is split into $S$ disjoint shards $D_1, \dots, D_S$. Each shard is partitioned into $R$ slices. When sample $x \in D_s$ must be forgotten, only shard $s$ is retrained starting from slice checkpoint preceding $x$, then re-aggregated into ensemble $\bar{P} = \frac{1}{S} \sum_s P_s$.

### SCRUB Minimax Objective
$$\mathcal{L}_{\text{SCRUB}} = -\min(\mathcal{L}_{\text{CE}}(f_\theta(x_f), y_f), \tau) + \gamma \cdot D_{\text{KL}}(\pi_{\text{ref}}(x_r) \parallel \pi_\theta(x_r))$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`npo_loss`** | `F.softplus` + log ratio | `jax.nn.softplus` | Numerically stable softplus | `NPO_LOSS_WGSL` |
| **`rmu_loss`** | Fused MSE difference | `jnp.mean` squared diff | Native Tensor subtraction | Elementwise squared diff |
| **`scrub_loss`** | Clamped CE + KL | `jnp.clip` + logsumexp | Vectorized KL accumulator | Dual loss reducer |
| **`sisa_ensemble`** | `torch.mean(probs, dim=0)` | `jnp.mean(probs, axis=0)` | `candle` shard mean | Multi-shard probability averager |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Gradient Ascent Runaway & Fluency Collapse**:
   - *Symptom*: Perplexity on benign text shoots from 12 to $>10,000$ after 5 steps of gradient ascent unlearning.
   - *Cause*: Unbounded $-\mathcal{L}_{\text{CE}}$ drives logits to $-\infty$, disrupting LayerNorm statistics and token embeddings.
   - *Fix*: Replace gradient ascent with NPO or DPO-negative preference loss.
2. **Superficial Refusal Scrubbing (Re-learning Vulnerability)**:
   - *Symptom*: Model refuses forget-set prompts, but fine-tuning on only 10 benign samples or linear probing recovers the forgotten knowledge.
   - *Cause*: Unlearning only altered the final classification head without clearing latent representations.
   - *Fix*: Apply RMU to intermediate MLP/attention representations and evaluate with representation probing.
3. **SISA Shard Variance & Calibration Mismatch**:
   - *Symptom*: High prediction disagreement and poor calibration across unlearned shard ensembles.
   - *Cause*: Sub-optimal shard partitioning with class imbalances across shards.
   - *Fix*: Stratified sharding and temperature-scaled probability calibration prior to aggregation.

---

## 7. Real-World Engineering Applications

1. **Biosecurity Safety Scrubbing (WMDP Benchmark)**: Erasing hazardous chemical, biological, and radiological weapon synthesis procedures from large language models without degrading biology and chemistry research capabilities.
2. **GDPR Right-to-be-Forgotten Compliance**: Certified deletion of user personal identifiable information (PII) from trained recommendation systems.
3. **Copyright & Proprietary Code Redaction**: Removing proprietary codebases and copyrighted text from code intelligence assistants.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.safetensors`:
1. `npo_ref`: Negative Preference Optimization loss.
2. `rmu_ref`: Representation Misdirection for Unlearning loss.
3. `scrub_ref`: Bounded minimax student-teacher unlearning loss.
4. `sisa_ref`: Multi-shard ensemble predicted probability distribution.

### Synchronized Ecosystem Artifacts
- **Notebook:** [03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.ipynb](../../notebooks/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.ipynb)
- **PyTorch 2 Module:** [03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.py](../../pytorch2/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.py)
- **JAX Module:** [03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.py](../../jax/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.py)
- **Candle Module:** [03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.rs](../../candle/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.rs)
- **Hardware Kernel:** [03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.rs](../../kernels/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.rs)
