# Direct Preference Optimization: DPO, IPO, KTO, and SimPO

## 1. Mathematical Foundations & Lineage

Direct preference alignment methods dispense with explicit reward modeling and reinforcement learning loops by analytically substituting the closed-form optimal policy into the Bradley-Terry preference likelihood.

The developmental lineage spans:
1. **DPO (Direct Preference Optimization / Rafailov et al. 2023)**: Proved that under KL-regularized reward maximization, the optimal reward satisfies $r^*(x, y) = \beta \ln \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \ln Z(x)$. Substituting this into the Bradley-Terry objective cancels the partition function $Z(x)$, yielding a stable closed-form classification loss directly on policy parameters.
2. **IPO (Identity Preference Optimization / Azar et al. 2023)**: Highlighted that DPO's cross-entropy loss saturates only when likelihood ratios approach infinity, encouraging the policy to memorize paired data. IPO regularizes the general preference objective with a quadratic penalty without assuming a Bradley-Terry choice model.
3. **KTO (Kahneman-Tversky Optimization / Ethayarajh et al. 2024)**: Grounded in behavioral economics and Prospect Theory, KTO aligns models using unpaired binary feedback (whether an output is desirable or undesirable) using loss aversion multipliers $\lambda_D, \lambda_U$.
4. **SimPO (Simple Preference Optimization / Meng et al. 2024)**: Observed that DPO's cumulative log-probabilities bias models toward lengthy outputs. SimPO normalizes log-probabilities by response length and introduces an explicit target margin $\gamma > 0$, eliminating the reference model $\pi_{\text{ref}}$ entirely while outperforming DPO on MT-Bench and Arena-Hard.

---

## 2. The Implicit Reward Formulation & DPO

Under the standard RLHF objective:
$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r(x, y) \right] - \beta D_{\text{KL}}\left( \pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x) \right)$$
the exact analytical optimal policy is:
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
Solving for the ground-truth reward $r(x, y)$:
$$r(x, y) = \beta \ln \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \ln Z(x)$$
Substituting this into the pairwise Bradley-Terry preference probability $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$ causes the prompt partition function $\beta \ln Z(x)$ to cancel out exactly:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E} \left[ \ln \sigma\left( \beta \left( \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]$$

---

## 3. Non-Saturating IPO, Reference-Free SimPO & Unpaired KTO

### Identity Preference Optimization (IPO)
To avoid over-optimizing likelihood ratios on easy pairs, IPO enforces a target log-ratio margin:
$$\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E} \left[ \left( \beta \left( \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) - \frac{1}{2\tau} \right)^2 \right]$$
The quadratic penalty prevents gradients from vanishing while stopping the policy from diverging excessively from the reference model.

### Simple Preference Optimization (SimPO)
SimPO eliminates the reference model by combining average token log-likelihood and a positive margin $\gamma$:
$$\bar{p}_\theta(y \mid x) = \frac{1}{|y|} \sum_{t=1}^{|y|} \ln \pi_\theta(y_t \mid x, y_{<t})$$
$$\mathcal{L}_{\text{SimPO}}(\theta) = \mathbb{E} \left[ \text{softplus}\left( -\left( \beta (\bar{p}_\theta(y_w \mid x) - \bar{p}_\theta(y_l \mid x)) - \gamma \right) \right) \right]$$
By dividing by sequence length $|y|$, SimPO completely stops verbosity exploitation without requiring separate length penalties.

### Kahneman-Tversky Optimization (KTO)
In real-world applications, collecting paired preferences $(y_w, y_l)$ is expensive, while collecting binary upvotes/downvotes is trivial. KTO optimizes unpaired samples using asymmetric loss weights:
$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x, y)} \left[ w(y) \cdot \sigma\left( \text{sign}(y) \cdot \beta \left( \ln \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) \right) \right]$$
where $w(y) = \lambda_D$ for desirable completions and $w(y) = \lambda_U$ ($\lambda_U > \lambda_D$) for undesirable completions, reflecting human loss aversion.

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`dpo_loss`** | `mean(softplus(-beta * (pi_diff - ref_diff)))` | `jnp.mean(jax.nn.softplus(-logits))` | Numerically safe softplus sum | `DPO_LOGITS_WGSL` |
| **`ipo_loss`** | `mean((logits - 1/(2*tau))^2)` | `jnp.mean((logits - target)**2)` | Quadratic margin penalty sum | Fused squared difference |
| **`simpo_loss`**| `mean(softplus(-(beta*(norm_w - norm_l) - gamma)))` | Length normalized softplus sum | Length normalized softplus | Fused SimPO margin reduction |
| **`kto_loss`** | Unpaired asymmetric sigmoid weighting | `jnp.where` asymmetric weighting | Prospect theory weighted sigmoid | Fused loss aversion kernel |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Length Bias & Verbosity Exploitation in DPO**:
   - *Symptom*: Fine-tuned model outputs become excessively wordy and repetitive because unnormalized sequence log-probabilities scale with sequence length.
   - *Cause*: Cumulative token log-probabilities $\sum \ln \pi(y_t)$ allow longer mediocre responses to accumulate higher log-ratios than concise high-quality answers.
   - *Fix*: Apply **SimPO** length-normalization $\frac{1}{|y|} \sum \ln \pi(y_t)$ or append explicit length penalties.
2. **Reference Model GPU VRAM Saturation**:
   - *Symptom*: Out-of-memory errors during multi-GPU DPO distributed training.
   - *Cause*: Holding both active policy $\pi_\theta$ and frozen reference model $\pi_{\text{ref}}$ doubles memory overhead.
   - *Fix*: Adopt **SimPO**, which operates reference-free, freeing $50\%$ of GPU memory.
3. **Over-Optimization on Mislabelled Preferences in DPO**:
   - *Symptom*: Severe perplexity degradation and degradation on out-of-distribution prompts.
   - *Cause*: Logistic cross-entropy forces $\pi_\theta(y_w) \to 1$ and $\pi_\theta(y_l) \to 0$ without bounds.
   - *Fix*: Use **IPO** with a bounded target margin $\frac{1}{2\tau}$.

---

## 6. Cross-Domain Generalisation Matrix

1. **Open-Source Chat Alignment (Zephyr-7B / Llama 3 Instruct)**: DPO and SimPO replace complex PPO clusters, aligning 8B-70B models on single GPU nodes.
2. **Code Preference Tuning**: SimPO trains models to prefer optimal, concise algorithms over bloated code implementations without verbosity drift.
3. **Production Web Service Feedback Loops**: KTO directly consumes raw user thumbs-up / thumbs-down telemetry without requiring paired synthetic comparisons.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `dpo_loss_ref`: Direct Preference Optimization loss with $\beta=0.1$.
3. `ipo_loss_ref`: Identity Preference Optimization quadratic loss with $\tau=0.5$.
4. `simpo_loss_ref`: Simple Preference Optimization length-normalized loss with $\gamma=0.5$.
5. `kto_loss_ref`: Kahneman-Tversky Optimization loss across desirable and undesirable completions.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.py`](../../../jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
