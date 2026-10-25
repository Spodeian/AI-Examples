# RLHF: Reward Modeling and PPO vs. RLOO

## 1. Mathematical Foundations & Lineage

Reinforcement Learning from Human Feedback (RLHF) steers pre-trained autoregressive models toward human preferences, safety standards, and stylistic requirements by maximizing expected reward subject to an informational divergence trust region.

The theoretical lineage comprises:
1. **The Bradley-Terry Model (Bradley & Terry 1952, Luce 1959)**: Defined the probabilistic choice model for pairwise comparisons, forming the mathematical cornerstone of preference-based reward learning.
2. **Preference-Based Deep RL (Christiano et al. 2017)**: Demonstrated that neural reward models trained on pairwise trajectory comparisons can guide continuous control policies.
3. **PPO-RLHF (Ziegler et al. 2019, Stiennon et al. 2020, Ouyang et al. 2022)**: Combined learned Bradley-Terry reward models with 4-model actor-critic Proximal Policy Optimization (Actor, Critic, Reference Model, Reward Model) using per-token KL penalties.
4. **REINFORCE Leave-One-Out (RLOO / Kool et al. 2019, Ahmadian et al. 2024)**: Proved that a simple multi-sample Monte Carlo baseline $\frac{1}{K-1} \sum_{j \ne i} R_j$ provides an unbiased, variance-reduced advantage estimator that completely eliminates the separate critic network, reducing GPU memory by $\sim 50\%$.

---

## 2. Bradley-Terry Reward Modeling

Given a prompt $x$ and two candidate completions—a preferred response $y_w$ and a dispreferred response $y_l$—the Bradley-Terry preference probability is:
$$P(y_w \succ y_l \mid x) = \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right)$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$.

### Loss Formulation
Minimizing the negative log-likelihood of human pairwise preferences yields:
$$\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \ln \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right] = \mathbb{E} \left[ \text{softplus}\left( -(r_\theta(x, y_w) - r_\theta(x, y_l)) \right) \right]$$
where $\text{softplus}(z) = \ln(1 + e^z)$ provides numerical stability against exponential overflow when $r_w \gg r_l$.

---

## 3. PPO Token-Level Reward Shaping vs. Critic-Free RLOO

### PPO Token-Level KL Shaping
In PPO-RLHF, an actor $\pi_\theta$ generates sequence $y_{1:T}$. To prevent policy collapse to ungrammatical reward-hacking optima, each token receives an instantaneous penalty proportional to its divergence from frozen reference model $\pi_{\text{ref}}$:
$$r_t = \begin{cases} -\beta \left( \ln \pi_\theta(y_t \mid y_{<t}) - \ln \pi_{\text{ref}}(y_t \mid y_{<t}) \right) & \text{if } t < T \\ r_\phi(x, y) - \beta \left( \ln \pi_\theta(y_T \mid y_{<T}) - \ln \pi_{\text{ref}}(y_T \mid y_{<T}) \right) & \text{if } t = T \end{cases}$$
A learned critic $V_\psi(s_t)$ predicts expected discounted future return from token $t$, enabling Generalized Advantage Estimation (GAE).

### RLOO (REINFORCE Leave-One-Out)
While PPO requires maintaining four simultaneous LLMs in GPU memory, RLOO achieves comparable or superior sample efficiency without a learned critic.

For each prompt $x$, RLOO samples $K \ge 2$ independent completions $\{y_1, \dots, y_K\} \sim \pi_\theta(\cdot \mid x)$ and computes total trajectory score $R_i = r_\phi(x, y_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$.
The advantage for sample $i$ is calculated using the average score of all *other* $K-1$ sampled completions as an unbiased baseline:
$$b_i = \frac{1}{K - 1} \sum_{j \ne i} R_j = \frac{\left( \sum_{j=1}^K R_j \right) - R_i}{K - 1}$$
$$\hat{A}_i = R_i - b_i$$
Notice that $\sum_{i=1}^K \hat{A}_i = 0$ strictly holds for every prompt batch. The policy gradient loss simplifies to:
$$\mathcal{L}_{\text{RLOO}}(\theta) = -\frac{1}{B \cdot K} \sum_{b=1}^B \sum_{k=1}^K \hat{A}_{b, k} \sum_{t=1}^{|y_{b, k}|} \ln \pi_\theta(y_{b, k, t} \mid y_{b, k, <t})$$

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`bradley_terry_reward_loss`** | `mean(softplus(-(rw - rl)))` | `jnp.mean(jax.nn.softplus(-diff))` | Numerically safe softplus sum | Fused binary cross-entropy |
| **`rlhf_token_level_rewards`** | `-beta * kl; rew[:, -1] += score` | `rewards.at[:, -1].add(score)` | In-place token trajectory buffer | Parallel token KL + score |
| **`rloo_leave_one_out_advantages`**| `scores - (sum(s) - s)/(K - 1)` | `scores - (total - s)/(k - 1)` | Vector leave-one-out accumulator | `RLOO_ADVANTAGES_WGSL` |
| **`rloo_policy_gradient_loss`** | `-mean(advantages * log_probs_sum)`| `-jnp.mean(adv * log_probs_sum)` | Negative dot product reduction | Fused REINFORCE reduction |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Reward Model Over-Optimization (Goodhart's Law)**:
   - *Symptom*: High evaluated reward model scores accompanied by degraded, repetitive, or sycophantic text.
   - *Cause*: Insufficient KL penalty ($\beta \to 0$), allowing policy to exploit blind spots in the reward model.
   - *Fix*: Calibrate $\beta \in [0.01, 0.1]$ and evaluate with an independent holdout reward model or LLM judge.
2. **PPO Critic Estimation Drift**:
   - *Symptom*: Critic value loss explodes, triggering erratic actor policy updates and catastrophic collapse.
   - *Cause*: Mismatched learning rates between actor and critic, or non-stationary returns during rapid policy iteration.
   - *Fix*: Transition from PPO to **RLOO** or **GRPO**, completely removing critic parameterization.
3. **RLOO Degeneracy Under $K=1$**:
   - *Symptom*: Division by zero ($K-1 = 0$) in leave-one-out baseline calculation.
   - *Cause*: Sampling only 1 completion per prompt.
   - *Fix*: Enforce $K \ge 2$ (typically $K \in [4, 8]$) for effective variance reduction.

---

## 6. Cross-Domain Generalisation Matrix

1. **Helpful & Harmless Dialogue Agents (Claude / ChatGPT)**: Multi-turn pairwise reward models enforce non-toxicity, honesty, and safety compliance.
2. **Code Synthesis Alignment**: Unit test execution results provide ground-truth binary rewards; RLOO optimizes pass@1 code generation without critic overhead.
3. **Instruction Following Under Length Penalties**: Reward modeling penalizes verbosity and truncation, aligning models with strict executive summary length constraints.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `bt_loss_ref`: Bradley-Terry softplus preference loss across paired completions.
3. `rlhf_rewards_ref`: Token-level shaped reward trajectory with terminal score injection.
4. `rloo_adv_ref`: Leave-one-out zero-sum advantage vector across $K=4$ completions.
5. `rloo_loss_ref`: REINFORCE advantage policy gradient loss.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.py`](../../../jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
