# Test-Time Compute: Process Reward Models (PRMs), Best-of-N Reranking, and MCTS

## 1. Mathematical Foundations & Lineage

Test-time compute scaling represents a pivotal paradigm shift in reasoning AI (exemplified by OpenAI o1, o3, and DeepSeek-R1), demonstrating that allocating compute dynamically during inference—via verification, parallel search, and tree exploration—yields performance improvements orthogonal to parameter scaling during pretraining.

The conceptual lineage comprises:
1. **Outcome Reward Models (ORMs) & Rejection Sampling (Cobbe et al. 2021)**: Evaluated only the terminal answer of a reasoning chain, suffering from credit assignment ambiguity on intermediate reasoning errors.
2. **Process Reward Models (PRMs) & "Let's Verify Step by Step" (Lightman et al. 2023)**: Assigned step-level correctness probabilities to every step in a mathematical derivation, providing dense step-by-step credit assignment.
3. **Best-of-$N$ (BoN) Test-Time Compute Scaling (Brown et al. 2024)**: Explored parallel rollout scaling, showing power-law accuracy gains with increasing samples evaluated by a combination of ORM and PRM verifiers.
4. **Monte Carlo Tree Search (MCTS) & AlphaZero Reasoning (Silver et al. 2017, Snell et al. 2024)**: Applied Predictor Upper Confidence bounds for Trees (PUCT) to navigate branching step spaces, combining step policy priors with process value rollouts.

---

## 2. Process Reward Model (PRM) Aggregation Mechanics

Given a multi-step candidate solution consisting of $S$ reasoning steps, an autoregressive PRM assigns step correctness probabilities $p_s \in (0, 1]$ for each step $s \in \{1, \dots, S\}$, masked by binary indicator $m_s \in \{0, 1\}$.

### Product Aggregation (Joint Probability)
Under the assumption of step conditional independence:
$$R_{\text{prod}} = \prod_{s=1}^S p_s^{m_s} = \exp\left( \sum_{s=1}^S m_s \ln(\max(p_s, \epsilon)) \right)$$
Product aggregation penalizes longer reasoning chains unless every step maintains exceptionally high confidence.

### Minimum Aggregation (Weakest Link Principle)
In formal logic and mathematics, a single incorrect deduction invalidates the entire conclusion:
$$R_{\text{min}} = \min_{s: m_s = 1} p_s$$
Invalid/padding steps ($m_s = 0$) are replaced with $1.0$ so they do not artificially constrain the minimum.

---

## 3. Best-of-N Reranking & MCTS PUCT Selection

### Combined ORM-PRM Best-of-$N$ Reranking
Given $N$ parallel candidate trajectories sampled from policy $\pi_\theta(\cdot \mid q)$:
$$S_i = \alpha \cdot R_{\text{ORM}}(i) + (1 - \alpha) \cdot R_{\text{PRM}}(i)$$
$$i^* = \arg\max_{i \in \{1, \dots, N\}} S_i$$
Combining terminal outcome confidence with step soundness protects against both local step deception and terminal false positives.

### AlphaZero PUCT Step Selection
For state node $s$ with child actions (next reasoning steps) $a \in \mathcal{A}$:
$$U(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_{b} N(s, b)}}{1 + N(s, a)}$$
$$a^* = \arg\max_{a \in \mathcal{A}} U(s, a)$$
where $Q(s, a)$ is the running average PRM evaluation, $P(s, a)$ is the policy generation prior, and $N(s, a)$ is the visit count.

### Incremental Online Backpropagation
When a simulation rollout or step verification returns value $V$:
$$N'(s, a^*) = N(s, a^*) + 1$$
$$Q'(s, a^*) = Q(s, a^*) + \frac{V - Q(s, a^*)}{N'(s, a^*)}$$

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`prm_product_aggregation`** | `torch.exp((torch.log(p) * m).sum(-1))` | `jnp.exp(jnp.sum(jnp.log(p) * m, -1))` | Row-wise log-sum-exp reduction | `prm_product_aggregation_cpu` |
| **`prm_min_aggregation`** | `torch.where(m > 0.5, p, 1.0).min(-1)` | `jnp.min(jnp.where(m > 0.5, p, 1.0), -1)` | Masked minimum reduction | `prm_min_aggregation_cpu` |
| **`best_of_n_rerank`** | `alpha * orm + (1 - alpha) * prm` | `alpha * orm + (1 - alpha) * prm` | Weighted scoring and argmax | Fused top-1 index selection |
| **`puct_node_selection`** | `Q + c * P * sqrt(N_tot) / (1 + N)` | `Q + c * P * jnp.sqrt(N_tot) / (1 + N)` | Vectorized PUCT formula | `PUCT_SELECTION_WGSL` |
| **`mcts_backpropagation`** | In-place tensor copy & update | Functional `.at[a].set(...)` | In-place buffer running average | Running mean accumulator |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Length Bias & Product Aggregation Collapse**:
   - *Symptom*: Product aggregation strictly selects trivial 1-step shortcuts and rejects thorough 10-step proofs because multiplying 10 numbers $< 1$ yields exponential decay.
   - *Cause*: $R_{\text{prod}} = \prod_{s=1}^S p_s$ penalizes chain length rather than sound reasoning depth.
   - *Fix*: Use length-normalized geometric mean $\exp\left(\frac{1}{S} \sum_s \ln p_s\right)$ or the minimum aggregation $R_{\text{min}}$.
2. **PRM False Positive Hallucination (Hacking the Verifier)**:
   - *Symptom*: Search branches exploit subtle blind spots in PRMs by producing superficially convincing but mathematically invalid lemmas with $p_s \approx 0.99$.
   - *Cause*: PRM trained on unrepresentative out-of-distribution synthetic error distributions.
   - *Fix*: Hybridize PRM step scores with deterministic code execution checks or ORM terminal consensus.
3. **MCTS Search Starvation**:
   - *Symptom*: Tree search fails to explore diverse alternative hypotheses, repeatedly expanding the first greedy path.
   - *Cause*: Exploration constant $c_{\text{puct}}$ set too low relative to scale of $Q$ values.
   - *Fix*: Tune $c_{\text{puct}} \in [1.0, 2.5]$ and normalize $Q$ values dynamically into $[0, 1]$ via min-max tree bounds.

---

## 6. Cross-Domain Generalisation Matrix

1. **Complex Mathematical Proofs (AIME, USAMO, Putnam)**: PRMs evaluate individual derivation steps, enabling MCTS to backtrack upon encountering algebraic errors.
2. **Multi-File Software Refactoring**: PRMs evaluate whether each intermediate AST patch compiles and passes unit tests before proceeding to downstream modules.
3. **Retrosynthetic Chemical Planning**: Search trees navigate plausible reaction steps, using molecular stability verifiers as process reward models.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `prm_prod_ref`: Product aggregation over $S=5$ steps.
3. `prm_min_ref`: Min aggregation (weakest link) over $S=5$ steps.
4. `combined_scores_ref` & `best_idx_ref`: Best-of-$N$ ORM-PRM reranking.
5. `puct_scores_ref` & `selected_action_ref`: AlphaZero PUCT exploration scores across $A=5$ branches.
6. `q_updated_ref` & `visits_updated_ref`: Online incremental MCTS backpropagation update.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.py`](../../../jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
