# Continuous Control: DDPG, TD3, and SAC

## 1. Mathematical Foundations & Lineage

Continuous action spaces $\mathcal{A} \subseteq \mathbb{R}^D$ render discrete $\max_{a \in \mathcal{A}} Q(s, a)$ operations intractable. Modern continuous actor-critic algorithms bridge value iteration and policy optimization by parameterizing policies with continuous outputs:
1. **DDPG (Lillicrap et al. 2015)**: Extends Deterministic Policy Gradient (Silver et al. 2014) to deep networks: $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_a Q_\phi(s, a)|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)]$.
2. **TD3 (Twin Delayed DDPG / Fujimoto et al. 2018)**: Identifies function approximation overestimation in continuous control and introduces three foundational stabilizers: clipped double Q-learning, delayed policy updates, and target action smoothing.
3. **SAC (Soft Actor-Critic / Haarnoja et al. 2018)**: Maximizes the maximum-entropy RL objective $\mathbb{E}[R(s, a) + \alpha \mathcal{H}(\pi(\cdot \mid s))]$ via squashed Gaussian policies with exact change-of-variables Jacobian accounting and automated entropy temperature adjustment.

---

## 2. TD3 Stabilizers & Clipped Double-Q

TD3 prevents overestimation and stabilizes deterministic policy training via three mechanisms:

### 1. Clipped Double Q-Learning
Two independent critics $Q_{\phi_1}$ and $Q_{\phi_2}$ are maintained. The Bellman target uses the minimum predicted value:
$$y = r + \gamma (1 - d) \min_{i \in \{1, 2\}} Q_{\bar{\phi}_i}(s', \tilde{a}')$$
This provides a lower bound on expected value, reversing overestimation bias into benign underestimation.

### 2. Target Policy Smoothing
To prevent policies from exploiting sharp peaks in the critic value landscape:
$$\tilde{a}' = \text{clip}\left( \mu_{\bar{\theta}}(s') + \text{clip}(\epsilon, -c, c), \, a_{\min}, \, a_{\max} \right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

### 3. Delayed Policy Updates
Critic parameters $\phi_1, \phi_2$ are updated every step, while the actor $\theta$ and target networks $\bar{\theta}, \bar{\phi}$ are updated only once every $d$ steps (typically $d=2$).

---

## 3. Soft Actor-Critic (SAC) & Squashed Gaussian Change-of-Variables

SAC optimizes policies to maximize both task reward and policy entropy:
$$J_{\text{SAC}}(\pi) = \sum_{t=0}^\infty \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right]$$

### Squashed Gaussian Change-of-Variables
To bound continuous actions to $[-1, 1]^D$, SAC samples an unbounded latent $u \sim \mathcal{N}(\mu(s), \text{diag}(\sigma(s)^2))$ and applies invertible squashing:
$$a = \tanh(u)$$
By the probability change-of-variables formula:
$$p_A(a \mid s) = p_U(u \mid s) \cdot \left| \det \left( \frac{da}{du} \right) \right|^{-1}$$
Because $\frac{da_i}{du_i} = 1 - \tanh^2(u_i) = 1 - a_i^2$, the log-determinant of the diagonal Jacobian decomposes into:
$$\log \pi(a \mid s) = \log \mu(u \mid s) - \sum_{i=1}^D \log(1 - a_i^2 + \epsilon)$$
where $\epsilon > 0$ avoids numerical divergence when $|a_i| \to 1.0$.

### Automated Temperature Auto-Tuning
Temperature $\alpha$ is dynamically adjusted by gradient descent on the dual objective:
$$\mathcal{L}(\alpha) = \hat{\mathbb{E}}_{a \sim \pi} \left[ -\alpha \left( \log \pi(a \mid s) + \bar{\mathcal{H}} \right) \right]$$
where $\bar{\mathcal{H}} = -D$ is the target entropy (one natural unit of negative entropy per degree of freedom).

---

## 4. Multi-Ecosystem Implementation Matrix

| Method | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`td3_target_smoothing`** | `torch.clamp(a + clamp(noise))` | `jnp.clip` composition | Elementwise nested clamp | Parallel clipping kernel |
| **`td3_clipped_critic_target`** | `r + gamma * (1-d) * min(q1, q2)` | `jnp.minimum` + broadcast | Vector minimum | Fused target reduction |
| **`sac_squashed_gaussian_log_prob`** | `tanh` + Jacobian correction | Vectorized diagonal sum | Elementwise tanh + log-sum | `SQUASHED_GAUSSIAN_JACOBIAN_WGSL` |
| **`sac_temperature_loss`** | `-mean(alpha * (log_p + H_target))` | Vectorized scalar product | Scalar accumulator | Temperature SGD step |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Omitting the Tanh Jacobian Correction in SAC**:
   - *Symptom*: SAC agent policies saturate at action boundaries $\pm 1.0$ and diverge.
   - *Cause*: Neglecting $-\sum_i \log(1 - a_i^2 + \epsilon)$ miscalculates entropy $\mathcal{H}$, creating an artificial gradient pulling actions toward extreme bounds.
   - *Fix*: Always include the change-of-variables log-determinant in the actor policy gradient.
2. **Actor-Critic Covariance Churn in DDPG**:
   - *Symptom*: Continuous control returns suddenly drop to near zero after steady learning.
   - *Cause*: Moving the actor and critic at the same rate creates non-stationary targets with positive feedback loops.
   - *Fix*: Adopt TD3's delayed policy updates ($d=2$) and twin clipped critics.
3. **Fixed Temperature $\alpha$ Mismatch**:
   - *Symptom*: Policy fails to solve fine-motor control (temperature too high) or falls into suboptimal early local optima (temperature too low).
   - *Fix*: Auto-tune $\alpha$ dynamically with target entropy $\bar{\mathcal{H}} = -\text{dim}(\mathcal{A})$.

---

## 6. Cross-Domain Generalisation Matrix

1. **Robotic Manipulation & Dexterous Grasping**: SAC's maximum-entropy objective ensures compliance and multimodal exploration across high-degree-of-freedom robotic arms.
2. **Autonomous Drone Flight in Turbulent Wind**: TD3's target policy smoothing acts as an intrinsic regularizer against sensor jitter and aerodynamic wind gusts.
3. **Industrial Chemical Process Control**: Continuous valve pressure and reactant inflow adjustment under strict safety boundaries $[-1, 1]$.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `smoothed_actions_ref`: Double-clipped action smoothing over continuous actuator channels.
3. `td3_targets_ref`: Twin-critic minimum Bellman target evaluation.
4. `sac_actions_ref` and `sac_log_probs_ref`: Squashed Gaussian forward mapping and analytical log-probabilities.
5. `alpha_loss_ref`: Maximum-entropy temperature adjustment loss.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.py`](../../../jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
