# Multi-Agent RL: QMIX, MAPPO, CFR, and Agentic Reasoning Loops

## 1. Mathematical Foundations & Lineage

Multi-Agent Reinforcement Learning (MARL) coordinates ensembles of distributed autonomous agents operating in shared environments under cooperative, competitive, or mixed-motive incentives.

The field traces its modern lineage through:
1. **Independent Q-Learning (IQL / Tan 1993)**: Each agent independently updates its own $Q$-function while treating peers as part of the environment, suffering severe non-stationarity.
2. **Value-Decomposition Networks (VDN / Sunehag et al. 2017)**: Introduced linear additive joint value factorization $Q_{\text{tot}}(\mathbf{s}, \mathbf{a}) = \sum_{i=1}^N Q_i(s_i, a_i)$.
3. **QMIX (Rashid et al. 2018)**: Replaced linear additivity with monotonic nonlinear mixing networks via state-conditioned hypernetworks enforcing $\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0$, unlocking complex collaborative behaviors in StarCraft Multi-Agent Challenge (SMAC).
4. **Multi-Agent PPO (MAPPO / Yu et al. 2022)**: Demonstrated that standard on-policy actor-critic with centralized value estimation $V(\mathbf{s})$ matches or exceeds specialized value-factorization methods across cooperative multi-agent benchmarks.
5. **Counterfactual Regret Minimization (CFR / Zinkevich et al. 2007, Brown & Sandholm 2019)**: Provably converges to Nash equilibria in zero-sum imperfect-information games by iteratively accumulating and matching counterfactual regrets.
6. **Agentic Reasoning & Debate Loops (Du et al. 2023, Liang et al. 2023)**: Extended multi-agent game-theoretic principles to Foundation Model debate, where consensus and cross-agent critique eliminate individual hallucination.

---

## 2. The Individual-Global-Max (IGM) Principle & QMIX Monotonicity

In Centralized Training with Decentralized Execution (CTDE), agents make local decisions based only on individual history $a_i = \arg\max_{a_i} Q_i(\tau_i, a_i)$, yet training optimizes a joint utility $Q_{\text{tot}}(\mathbf{s}, \mathbf{a})$.

### The IGM Condition
Decentralized greedy action selection aligns with joint optimization if and only if:
$$\arg\max_{\mathbf{a}} Q_{\text{tot}}(\mathbf{s}, \mathbf{a}) = \begin{pmatrix} \arg\max_{a_1} Q_1(\tau_1, a_1) \\ \vdots \\ \arg\max_{a_N} Q_N(\tau_N, a_N) \end{pmatrix}$$

### Monotonic Hypernetwork Mixing
QMIX guarantees IGM by enforcing monotonicity:
$$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0 \quad \forall i \in \{1, \dots, N\}$$
A hypernetwork consumes global state $\mathbf{s}$ and generates weights $W_1 = |W_1^{\text{raw}}(\mathbf{s})| \ge 0$ and $W_2 = |W_2^{\text{raw}}(\mathbf{s})| \ge 0$. The forward pass computes:
$$\mathbf{h} = \text{ELU}\left( \mathbf{Q}_{\text{agents}} W_1 + \mathbf{b}_1 \right), \quad Q_{\text{tot}} = \mathbf{h} W_2 + b_2$$
Because all weights $W_1, W_2$ are strictly non-negative, any increase in an individual agent's utility $Q_i$ monotonically increases joint $Q_{\text{tot}}$.

---

## 3. MAPPO Centralized GAE, CFR Regret Matching & Agentic Debate

### MAPPO Centralized GAE
Decentralized policies $\pi_{\theta_i}(a_i \mid o_i)$ are guided during training by a centralized critic $V_\phi(\mathbf{s})$ conditioned on the true global state $\mathbf{s}$. Team Generalized Advantage Estimation (GAE) computes:
$$\delta_t = r_t + \gamma V_\phi(\mathbf{s}_{t+1})(1 - d_t) - V_\phi(\mathbf{s}_t)$$
$$\hat{A}_t^{\text{central}} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

### Counterfactual Regret Minimization (CFR)
In extensive-form games with information sets $I \in \mathcal{I}$, cumulative regret $R^T(I, a)$ measures the counterfactual reward forfeited by not playing action $a$ up to step $T$. Regret matching derives strategy $\sigma^{T+1}(I, a)$ via:
$$R^{T,+}(I, a) = \max(R^T(I, a), 0)$$
$$\sigma^{T+1}(I, a) = \begin{cases} \frac{R^{T,+}(I, a)}{\sum_{a' \in A(I)} R^{T,+}(I, a')} & \text{if } \sum_{a'} R^{T,+}(I, a') > 0 \\ \frac{1}{|A(I)|} & \text{otherwise} \end{cases}$$

### Multi-Agent Deliberation & Consensus Metrics
In agentic reasoning loops, $N$ models propose chain-of-thought solutions $\mathbf{e}_i \in \mathbb{R}^D$. Iterative debate rounds converge as pairwise consensus approaches 1.0:
$$S(\mathbf{e}_1, \dots, \mathbf{e}_N) = \frac{1}{N(N-1)} \sum_{i \ne j} \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\max\left(\|\mathbf{e}_i\|_2 \|\mathbf{e}_j\|_2, \, \epsilon\right)}$$

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`qmix_monotonic_mixing`** | `bmm(agent_qs, abs(w1))` + ELU | `jnp.matmul` + `jax.nn.elu` | Two-stage nested GEMM + ELU | Parallel absolute GEMM |
| **`mappo_centralized_gae`** | Reversed temporal accumulation loop | `reversed(range(t))` scan | Reverse step GAE buffer | Fused backward delta reduction |
| **`cfr_regret_matching`** | `clamp(min=0)` + `where(sum>0)` | `jnp.maximum` + `jnp.where` | Vector positive sum accumulator | `CFR_REGRET_MATCHING_WGSL` |
| **`agentic_consensus_metric`**| Off-diagonal pairwise cosine mean | `(sum(sim) - n) / (n*(n-1))` | Pairwise normalized dot-product | Fused cosine reduction |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Relative Overgeneralization in Monotonic Mixing**:
   - *Symptom*: QMIX converges to a suboptimal joint policy in payoff matrices with a non-monotonic cooperative penalty (e.g. the Matrix Climb Game).
   - *Cause*: The monotonicity constraint $\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0$ restricts the hypothesis class, rendering non-monotonic optimal joint actions unreachable.
   - *Fix*: Augment QMIX with Weighted QMIX (WQMIX) or Qatten attention-weighted transformation to penalize non-monotonic updates.
2. **Lazy Agent Problem in Centralized Critics**:
   - *Symptom*: In a team of four agents, one agent performs all actions while the remaining three remain motionless.
   - *Cause*: Centralized team reward $r_t$ provides zero gradient distinguishing individual credit from team success.
   - *Fix*: Apply Counterfactual Multi-Agent (COMA) advantage baselines $A^i(s, \mathbf{a}) = Q(s, \mathbf{a}) - \sum_{a'^i} \pi^i(a'^i \mid \tau^i) Q(s, (a'^i, \mathbf{a}^{-i}))$.
3. **Exploding Regret Under Non-Stationary Opponents in CFR**:
   - *Symptom*: Regret values oscillate wildly without strategy convergence in multiplayer games ($N > 2$).
   - *Cause*: Standard CFR guarantees Nash equilibrium convergence only in two-player zero-sum games.
   - *Fix*: Use Discounted CFR (DCFR) or Linear CFR (LCFR) with cubic/quadratic discounting on early iterations.

---

## 6. Cross-Domain Generalisation Matrix

1. **Autonomous Warehouse Fleet Dispatch (Kiva Systems)**: QMIX coordinates hundreds of automated guided vehicles (AGVs), resolving gridlocks and optimizing package transit without collision.
2. **Imperfect Information Financial Trading & Auctions**: CFR computes robust minimax bidding strategies against adversarial participants with private order-book state.
3. **Multi-Agent Coding & Formal Verification (Agentic Debate)**: Debating LLM agents verify formal Lean/Coq proofs and test cases; consensus metrics determine when to terminate reasoning rounds.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `qmix_q_tot_ref`: Forward evaluation of 2-layer monotonic hypernetwork mixing with ELU activation.
3. `mappo_advantages_ref`: Centralized multi-agent GAE across episode trajectories with terminal flags.
4. `cfr_strategy_ref`: Normalized regret matching strategy including uniform fallback for all-negative regrets.
5. `consensus_ref`: Normalized off-diagonal pairwise cosine consensus metric across agent thoughts.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.py`](../../../jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
