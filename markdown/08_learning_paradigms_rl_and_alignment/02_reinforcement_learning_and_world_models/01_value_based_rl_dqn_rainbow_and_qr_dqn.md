# Value-Based Reinforcement Learning: DQN, Rainbow, and QR-DQN

## 1. Mathematical Foundations & Lineage

Value-based reinforcement learning seeks an optimal policy $\pi^*$ in a Markov Decision Process (MDP) $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ by estimating the optimal action-value function:
$$Q^*(s, a) = \max_\pi \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t r_t \;\middle|\; s_0=s, a_0=a \right]$$

The lineage of deep value-based methods proceeds as follows:
1. **DQN (Mnih et al. 2015)**: Stabilizes neural Q-learning via experience replay buffers $\mathcal{D}$ and periodic target networks $Q_{\bar{\theta}}$.
2. **Double DQN (van Hasselt et al. 2016)**: Eliminates maximization bias by decoupling greedy action selection from action evaluation.
3. **Prioritized Experience Replay (PER, Schaul et al. 2016)**: Samples transitions non-uniformly proportional to temporal difference (TD) error magnitude $|\delta_i|^\alpha$, with importance-sampling correction.
4. **Dueling Architecture (Wang et al. 2016)**: Decomposes $Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')$ to learn state value independently of action advantages.
5. **Rainbow (Hessel et al. 2018)**: Integrates Double Q, PER, Dueling, Multi-step returns, Distributional RL, and NoisyNets.
6. **QR-DQN (Dabney et al. 2018)**: Replaces fixed categorical support projections (C51) with quantile regression over empirical return distributions.

---

## 2. Double DQN, Dueling Decomposition & PER

### Double DQN Target Formulation
Standard DQN targets $y = r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a')$ overestimate values because $\mathbb{E}[\max X] \ge \max \mathbb{E}[X]$. Double DQN decouples selection and evaluation:
$$a^* = \arg\max_{a'} Q_\theta(s', a')$$
$$y^{\text{Double}} = r + \gamma (1 - d) Q_{\bar{\theta}}(s', a^*)$$

### Dueling Architecture
Value streams $V(s) \in \mathbb{R}^1$ and advantage streams $A(s, a) \in \mathbb{R}^{|\mathcal{A}|}$ are combined with mean-centering identifiability:
$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$
Subtracting the mean ensures $\sum_a (Q(s, a) - V(s)) = 0$, guaranteeing that $V(s)$ accurately tracks state value.

### Prioritized Experience Replay (PER)
Transition $i$ is sampled with probability $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$, where $p_i = |\delta_i| + \epsilon$.
Unbiased stochastic gradient updates require importance-sampling weights:
$$w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta, \quad \tilde{w}_i = \frac{w_i}{\max_j w_j}$$
$$\mathcal{L}_{\text{PER}}(\theta) = \frac{1}{B} \sum_{i=1}^B \tilde{w}_i \cdot \ell_{\text{Huber}}(\delta_i)$$

---

## 3. Distributional RL & Quantile Regression (QR-DQN)

Rather than learning scalar expected return $Q(s, a) = \mathbb{E}[Z(s, a)]$, distributional RL models the full distribution of discounted returns $Z(s, a)$.

In QR-DQN (Dabney et al. 2018), $Z(s, a)$ is approximated by a uniform mixture of $N$ Dirac quantiles:
$$Z_\theta(s, a) = \frac{1}{N} \sum_{i=1}^N \delta_{\theta_i(s, a)}$$
assigned to target cumulative probabilities $\tau_i = \frac{2i - 1}{2N}$ for $i \in \{1, \dots, N\}$.

The quantile Huber loss between predicted quantiles $\theta_i$ and target quantiles $\hat{\mathcal{T}} \theta_j = r + \gamma (1-d) \theta_j(s', a^*)$ is:
$$\mathcal{L}_{\text{QR}}(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \rho_{\tau_i}^\kappa \left( \hat{\mathcal{T}} \theta_j - \theta_i \right)$$
where the quantile Huber regression penalty is defined as:
$$\rho_\tau^\kappa(u) = |\tau - \mathbb{I}(u < 0)| \cdot \frac{\mathcal{L}_\kappa(u)}{\kappa}$$
$$\mathcal{L}_\kappa(u) = \begin{cases} \frac{1}{2} u^2 & \text{if } |u| \le \kappa \\ \kappa \left( |u| - \frac{1}{2} \kappa \right) & \text{otherwise} \end{cases}$$

---

## 4. Multi-Ecosystem Implementation Matrix

| Method | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`dueling_q_values`** | `value + advantage - adv.mean(-1)` | `jnp.mean` + broadcast add | `Tensor::mean_keepdim` + `broadcast_sub` | `DUELING_Q_WGSL` + CPU SIMD |
| **`double_dqn_targets`** | `gather(q_target, argmax(q_online))` | `jnp.argmax` index array | `argmax_keepdim` + `gather` | Parallel target gather |
| **`per_loss`** | `is_weights * (q - y)^2` | Vectorized elementwise loss | Weighted mean reduction | Fused IS Huber kernel |
| **`quantile_huber_loss`** | Pairwise broadcast diff `[B, N, N]` | `expand_dims` + `jnp.where` | Nested quantile loop | Quantile Huber reduction |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Maximization Bias Explosion**:
   - *Symptom*: Q-values steadily inflate while actual policy returns stagnate or collapse.
   - *Cause*: Standard single-network DQN target $\max_a Q(s', a)$ inherits upward variance bias.
   - *Fix*: Decouple selection $a^* = \arg\max Q_\theta$ and evaluation $Q_{\bar{\theta}}(s', a^*)$ via Double DQN.
2. **PER Representation Degeneracy & Premature Convergence**:
   - *Symptom*: Loss drops immediately to near zero, but the agent cycles through a small set of states.
   - *Cause*: Setting $\beta = 0$ in PER samples only high-TD transitions without IS correction, severely distorting the state distribution gradient.
   - *Fix*: Anneal $\beta$ from $0.4 \to 1.0$ linearly over training.
3. **QR-DQN Quantile Crossing & Gradient Saturation**:
   - *Symptom*: Quantile predictions $\theta_1, \dots, \theta_N$ invert order or collapse to identical values.
   - *Cause*: Large Huber $\kappa$ or unscaled learning rates causing out-of-order gradient updates.
   - *Fix*: Fix $\tau_i = \frac{2i-1}{2N}$ and ensure $\kappa = 1.0$ with AdamW ($\text{lr} \le 10^{-4}$).

---

## 6. Cross-Domain Generalisation Matrix

1. **Autonomous Driving**: Distributional value estimation (QR-DQN) separates epistemic sensor noise from the heavy-tailed tail-risk probability of rare vehicle collisions.
2. **Automated Trading & Portfolio Management**: Multi-step distributional Bellman operators evaluate value-at-risk ($\text{VaR}_\tau$) directly across multi-asset order books without Gaussian return assumptions.
3. **Power Grid Frequency Regulation**: Dueling value-advantage decomposition isolates baseline grid equilibrium value $V(s)$ from high-frequency sub-second battery dispatch adjustments $A(s, a)$.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.safetensors`:
1. `linear_anchor`: Canonical multi-backend linear transformation.
2. `dueling_q_ref`: State-value and action-advantage dueling composition.
3. `ddqn_targets_ref`: Double DQN decoupled action selection and target evaluation.
4. `per_loss_ref`: Importance-weighted TD error reduction.
5. `qr_loss_ref`: Pairwise quantile Huber regression loss over $N=8$ return quantiles.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.py`](../../../jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
