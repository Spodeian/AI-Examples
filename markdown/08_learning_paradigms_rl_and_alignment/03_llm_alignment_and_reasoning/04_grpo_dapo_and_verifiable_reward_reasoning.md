# GRPO: Group Relative Policy Optimization and Verifiable Reward Reasoning

## 1. Mathematical Foundations & Lineage

Group Relative Policy Optimization (GRPO) and verifiable reward reinforcement learning form the computational foundation of modern reasoning language models (DeepSeekMath, DeepSeek-R1), enabling models to discover complex chain-of-thought verification, backtracking, and self-correction without relying on learned value networks.

The conceptual lineage comprises:
1. **PPO Actor-Critic RL (Schulman et al. 2017)**: Established clipped surrogate objectives with generalized advantage estimation, but required maintaining a critic LLM of equivalent parameter size.
2. **DeepSeekMath & GRPO (Shao et al. Feb 2024)**: Introduced Group Relative Policy Optimization, demonstrating that normalizing scalar verifiable rewards across a group of $G$ rollouts per prompt completely eliminates the critic network while outperforming PPO on mathematical benchmarks.
3. **DeepSeek-R1 & R1-Zero (DeepSeek-AI Jan 2025)**: Revealed that large language models trained purely via GRPO on rule-based accuracy and formatting rewards autonomously discover test-time search, reasoning reflection, and dynamic test allocation without human demonstration data.
4. **Verifiable Rule-Based Rewards (GSM8K, MATH, HumanEval)**: Substituted subjective neural preference reward models with objective execution verifiers (Python interpreters, SymPy algebra checkers, regex formatting validators).

---

## 2. The GRPO Group Relative Advantage Estimator

Standard PPO estimates advantages $\hat{A}_t$ via a separate critic network $V_\psi(s_t)$. For a 70B parameter policy, the critic requires an additional 70B parameters, doubling VRAM requirements.

GRPO eliminates the critic by sampling a group of $G$ completions $\{o_1, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$ for each prompt $q$.
Given sequence-level scalar rewards $\{R_1, \dots, R_G\}$, GRPO normalizes advantages relative to the group statistics:
$$\mu = \frac{1}{G} \sum_{i=1}^G R_i$$
$$\sigma = \sqrt{\frac{1}{G} \sum_{i=1}^G (R_i - \mu)^2 + \epsilon}$$
$$\hat{A}_i = \frac{R_i - \mu}{\sigma}$$
Each completion $o_i$ is evaluated by how much better or worse it performs relative to its peer completions on the exact same problem prompt.

---

## 3. Verifiable Rule Rewards, K3 KL Regularization & Clipped Loss

### Composite Verifiable Rewards
To incentivize long-chain reasoning while maintaining structured output protocols, rewards are computed purely via deterministic rules:
$$R_i = w_{\text{acc}} \cdot \mathbb{I}(\text{Answer matches ground truth}) + w_{\text{format}} \cdot \mathbb{I}(\text{Reasoning enclosed in } \texttt{<think>} \dots \texttt{</think>})$$
Because verifiable rewards are ground-truth and un-hackable, the model cannot engage in reward hacking against an over-fitted neural proxy.

### Schulman's $k_3$ Unbiased Low-Variance KL Estimator
To anchor the active policy $\pi_\theta$ to the reference model $\pi_{\text{ref}}$, GRPO uses Schulman's $k_3$ estimator:
$$\Delta_{i, t} = \ln \pi_\theta(o_{i, t} \mid q, o_{i, <t}) - \ln \pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})$$
$$D_{\text{KL}}^{k_3}(\pi_\theta \| \pi_{\text{ref}})_{i, t} = e^{-\Delta_{i, t}} - 1 + \Delta_{i, t}$$
The $k_3$ estimator is strictly non-negative ($D_{\text{KL}}^{k_3} \ge 0$), has minimum variance at $\pi_\theta \approx \pi_{\text{ref}}$, and satisfies $\mathbb{E}[D_{\text{KL}}^{k_3}] = D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$.

### The Full GRPO Objective
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left( r_{i, t}(\theta) \hat{A}_i, \, \text{clip}(r_{i, t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}^{k_3}(\pi_\theta \| \pi_{\text{ref}})_{i, t} \right]$$
where $r_{i, t}(\theta) = \frac{\pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})}$.

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`grpo_group_advantages`** | `(rewards - mean) / (std + eps)` | `(rewards - mean) / (std + eps)` | Group mean and unbiased std | `GRPO_GROUP_ADVANTAGES_WGSL` |
| **`k3_kl_divergence`** | `exp(-delta) - 1.0 + delta` | `jnp.exp(-delta) - 1.0 + delta` | Vectorized elementwise float op | Parallel $k_3$ reduction |
| **`grpo_clipped_loss`** | `min(surr1, surr2) - beta * kl` | `jnp.minimum(surr1, surr2)` | Elementwise PPO ratio clamping | Fused PPO surrogate reduction |
| **`composite_verifiable_reward`** | `w_acc * acc + w_fmt * fmt` | `acc_weight * acc + fmt_weight * fmt` | Weighted linear combination | Fused rule-based combiner |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Format Degradation & Disappearance of `<think>` Tags**:
   - *Symptom*: Policy achieves zero format reward, outputs raw answers without reasoning, and reasoning accuracy plummets.
   - *Cause*: Format reward weight $w_{\text{format}} = 0$ or format penalty insufficient to overcome early gradient noise.
   - *Fix*: Maintain strict format verification $w_{\text{format}} \ge 0.5$, penalizing responses missing complete tag pairs.
2. **All-Success or All-Failure Group Stagnation**:
   - *Symptom*: On very easy or very hard problems where all $G$ completions score identically (all 0 or all 1), variance collapses to $\sigma \approx 0$.
   - *Cause*: Group advantage becomes zero ($A_i \approx 0$).
   - *Fix*: Calibrate problem difficulty distribution, filtering prompts with 0% or 100% historical group pass rates.
3. **Exploding Reasoning Traces & Repetition Loops**:
   - *Symptom*: Reasoning chains expand from 2,000 tokens to 32,000 tokens of circular self-reinforcing repetitive babble.
   - *Cause*: Insufficient KL penalty $\beta$ or lack of length penalty on incorrect answers.
   - *Fix*: Enforce token-level reference KL regularization ($\beta \in [0.01, 0.04]$).

---

## 6. Cross-Domain Generalisation Matrix

1. **Olympiad Mathematics & Theorem Proving (DeepSeek-R1 / OpenAI o1)**: GRPO trains models to perform multi-step search, alternative path testing, and proof verification.
2. **Autonomous Software Engineering & Unit Test Synthesis**: Compilers and pytest runners provide deterministic binary correctness feedback for code reasoning.
3. **Formal Verification in Lean 4 & Coq**: Proof state checkers evaluate tactic correctness, guiding interactive theorem provers.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `rewards_ref`: Composite accuracy and format rule reward evaluation.
3. `advantages_ref`: Group-normalized relative advantage estimation across $G=4$ rollouts.
4. `kl_div_ref`: Schulman's $k_3$ non-negative reference divergence.
5. `grpo_loss_ref`: Clipped surrogate loss with $\epsilon=0.2$ and $\beta=0.04$.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.py`](../../../jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
