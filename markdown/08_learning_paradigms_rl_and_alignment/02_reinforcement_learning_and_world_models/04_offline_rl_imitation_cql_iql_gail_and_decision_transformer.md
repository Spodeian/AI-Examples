# Offline Reinforcement Learning: CQL, IQL, GAIL, and Decision Transformer

## 1. Mathematical Foundations & Lineage

Standard reinforcement learning relies on active environment interaction where exploration discovers new state-action pairs. In real-world applications (clinical healthcare, autonomous transport, robotics manufacturing), active environment rollouts are prohibitively risky or expensive. Offline RL (Batch RL) learns optimal policies entirely from fixed logged datasets $\mathcal{D} = \{(s, a, r, s')\}$.

The lineage of offline and imitation methods includes:
1. **Behavioral Cloning (Pomerleau 1988)**: Supervised imitation $\min_\theta \mathbb{E}_{(s, a) \sim \mathcal{D}}[\|\pi_\theta(s) - a\|^2]$, vulnerable to compounding errors $\mathcal{O}(T^2)$ during closed-loop rollout.
2. **Generative Adversarial Imitation Learning (GAIL / Ho & Ermon 2016)**: Formulates imitation as matching occupancy measures $\rho_\pi(s, a)$ to expert distributions $\rho_E(s, a)$ via minimax discriminator games.
3. **Conservative Q-Learning (CQL / Kumar et al. 2020)**: Explicitly penalizes Q-values on out-of-distribution (OOD) actions while maximizing values on in-distribution actions, proving that conservative value estimates prevent distribution shift exploitation.
4. **Implicit Q-Learning (IQL / Kostrikov et al. 2021)**: Completely avoids evaluating unseen actions by fitting state values $V(s)$ using asymmetric expectile regression directly on dataset transitions.
5. **Decision Transformer (Chen et al. 2021)**: Casts offline RL as conditional sequence modeling, autoregressively generating actions conditioned on target return-to-go $\hat{R}_t = \sum_{t'=t}^T r_{t'}$.

---

## 2. Conservative Q-Learning (CQL)

Standard Q-learning on offline datasets hallucinates artificially high Q-values on unseen actions because $\max_a Q(s, a)$ picks out-of-distribution actions where the neural network's epistemic uncertainty is greatest.

CQL adds a conservative regularizer to the standard Bellman error:
$$\min_Q \alpha \cdot \mathcal{R}_{\text{CQL}}(Q) + \frac{1}{2} \mathbb{E}_{(s, a, s') \sim \mathcal{D}} \left[ \left( Q(s, a) - \left( r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') \right) \right)^2 \right]$$
For discrete action spaces, the conservative gap penalty is:
$$\mathcal{R}_{\text{CQL}}(Q) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta(\cdot \mid s)} [Q(s, a)] \right]$$
Kumar et al. (2020) prove that for sufficiently large $\alpha$, the expected policy value under $Q$ is a point-wise lower bound on the true policy value: $\mathbb{E}[Q^\pi(s, a)] \le \mathbb{E}[Q_{\text{true}}^\pi(s, a)]$.

---

## 3. Implicit Q-Learning (IQL) & In-Sample Expectiles

While CQL constrains OOD Q-values with an explicit penalty, IQL completely eliminates the need to query unseen actions during value estimation.

### Asymmetric Expectile Regression
IQL fits a state-value function $V_\psi(s)$ to upper expectiles of the critic $Q_\theta(s, a)$ using only transitions present in the dataset:
$$L_V(\psi) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left[ L_2^\tau \left( Q_\theta(s, a) - V_\psi(s) \right) \right]$$
where the asymmetric $L_2^\tau$ expectile loss is defined as:
$$L_2^\tau(u) = |\tau - \mathbb{I}(u < 0)| \cdot u^2 = \begin{cases} \tau u^2 & \text{if } u > 0 \\ (1 - \tau) u^2 & \text{if } u \le 0 \end{cases}$$
For $\tau \in (0.5, 1.0)$ (typically $\tau = 0.7$ or $0.9$), $V_\psi(s)$ estimates the upper quantile/expectile of dataset returns from state $s$, approximating $\max_a Q(s, a)$ without ever evaluating counterfactual or out-of-distribution actions!

### Advantage-Weighted Policy Extraction
Once $V_\psi(s)$ and $Q_\theta(s, a)$ are learned, the actor policy is extracted via Advantage-Weighted Regression (AWR):
$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left[ \exp\left( \beta (Q_\theta(s, a) - V_\psi(s)) \right) \cdot \| \pi_\phi(s) - a \|_2^2 \right]$$
where weights are clamped to prevent numerical explosion.

---

## 4. Multi-Ecosystem Implementation Matrix

| Method | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`cql_penalty`** | `torch.logsumexp - data_q` | `jax.nn.logsumexp - data_q` | LogSumExp reduction | Fused LogSumExp difference |
| **`iql_expectile_loss`** | `torch.where(diff > 0, tau, 1-tau) * u^2` | `jnp.where` expectile | Branchless elementwise | `IQL_EXPECTILE_LOSS_WGSL` + CPU SIMD |
| **`iql_actor_loss`** | `clamp(exp(beta * adv)) * sq_err` | `jnp.clip` + weighted sum | Vectorized AWR reduction | Parallel weighted MSE |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Compounding Covariate Shift in Behavioral Cloning**:
   - *Symptom*: Agent succeeds initially, then drifts off course and crashes catastrophically.
   - *Cause*: Standard supervised learning assumes i.i.d. states. Once an agent makes a minor error, it encounters states never seen in $\mathcal{D}$, causing compounding $\mathcal{O}(T^2)$ errors.
   - *Fix*: Use offline RL (IQL/CQL) with value-guided correction rather than pure BC.
2. **OOD Action Overestimation in Vanilla Offline Q-Learning**:
   - *Symptom*: Critic predicts $Q(s, a) = +10^6$ on offline datasets with actual returns $\le 100$.
   - *Cause*: Bellman target $\max_{a'} Q(s', a')$ queries out-of-distribution actions, picking up function approximation extrapolation spikes.
   - *Fix*: Apply CQL conservative gap regularization or in-sample IQL expectile regression.
3. **Unbounded Advantage Weights in AWR**:
   - *Symptom*: Policy gradients produce `NaN` or single-sample overfitting during actor extraction.
   - *Cause*: $\exp(\beta (Q - V))$ explodes when $Q - V \gg 0$.
   - *Fix*: Clamp weights to a maximum ceiling (e.g., $w \le 100.0$).

---

## 6. Cross-Domain Generalisation Matrix

1. **Healthcare Treatment Policy Optimization**: IQL derives optimal medication dosing from historical electronic health records without running unethical random trials on patients.
2. **Autonomous Driving Fleet Upgrades**: CQL extracts superior highway navigation policies from petabytes of human driving logs containing sub-optimal human driving.
3. **Industrial HVAC Energy Optimization**: Decision Transformer models historical building thermal sensor logs, conditioning on low electricity consumption targets to maintain comfortable indoor temperature.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `cql_penalty_ref`: LogSumExp minus dataset Q-value gap penalty.
3. `iql_expectile_ref`: Asymmetric $L_2^\tau$ expectile loss with $\tau = 0.7$.
4. `iql_actor_ref`: Advantage-weighted regression loss with exponential clamping.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.py`](../../../jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
