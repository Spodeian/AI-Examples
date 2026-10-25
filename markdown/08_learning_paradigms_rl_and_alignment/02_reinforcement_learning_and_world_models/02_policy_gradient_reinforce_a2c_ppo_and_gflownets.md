# Policy Gradient Methods: REINFORCE, A2C, PPO, and GFlowNets

## 1. Mathematical Foundations & Lineage

Policy gradient methods optimize parameterized stochastic policies $\pi_\theta(a \mid s)$ directly by ascending the expected discounted cumulative return:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]$$

The lineage traces key breakthroughs in variance reduction and mode discovery:
1. **REINFORCE (Williams 1992)**: Derives the Policy Gradient Theorem $\nabla_\theta J(\theta) = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t]$ via the log-derivative trick.
2. **Actor-Critic (A2C / Sutton et al. 1999)**: Replaces Monte Carlo returns $G_t$ with an advantage function $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$, reducing variance without introducing bias when $V$ is the true value function.
3. **GAE (Schulman et al. 2015)**: Exponentially interpolates between 1-step TD residuals and full Monte Carlo returns via GAE($\gamma, \lambda$).
4. **PPO-Clip (Schulman et al. 2017)**: Eliminates destructive policy updates via first-order clipped probability ratios without requiring expensive second-order Fisher-information inversions (TRPO).
5. **GFlowNets (Bengio et al. 2021; Malkin et al. 2022)**: Shifts objective from reward maximization (which collapses onto a single mode) to sampling composite structures with probability proportional to reward $P(x) \propto R(x)$ via Trajectory Balance.

---

## 2. Policy Gradient Theorem, GAE & PPO-Clip

### Generalized Advantage Estimation (GAE)
Given temporal difference (TD) residual $\delta_t^V = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t)$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l (1 - d_{t+l}) \delta_{t+l}^V$$
- $\lambda = 0$: $\hat{A}_t = \delta_t^V$ (low variance, biased if $V$ is imperfect).
- $\lambda = 1$: $\hat{A}_t = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$ (unbiased empirical return, high variance).
- $\lambda \in [0.9, 0.98]$ provides the optimal bias-variance tradeoff across high-dimensional control.

### PPO-Clip Surrogate Objective
Let $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t))$.
The clipped surrogate objective bounds policy updates:
$$\mathcal{L}_{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
Taking the minimum between clipped and unclipped objectives acts as a pessimistic bound, ensuring that improvements in probability $r_t > 1+\epsilon$ when $\hat{A}_t > 0$ do not trigger unbounded policy steps.

---

## 3. Generative Flow Networks (GFlowNets) & Trajectory Balance

Standard RL solves $\max_\pi \mathbb{E}[R(x)]$, converging greedily to the single highest-reward peak. However, in scientific discovery (small-molecule design, crystal structure search, formal proofs), sampling diverse multimodal candidates is critical.

GFlowNets treat generation as flow through a Directed Acyclic Graph (DAG) rooted at $s_0$, where terminal states $x$ satisfy flow consistency:
$$\sum_{s \to x} F(s \to x) = R(x)$$

The Trajectory Balance (TB) objective (Malkin et al. 2022) optimizes forward transition probabilities $P_F(s_{t+1} \mid s_t)$, backward transition probabilities $P_B(s_t \mid s_{t+1})$, and total flow $Z_\theta$:
$$\mathcal{L}_{\text{TB}}(\tau) = \left( \log \frac{Z_\theta \prod_{t=0}^{T-1} P_F(s_{t+1} \mid s_t; \theta)}{R(x) \prod_{t=0}^{T-1} P_B(s_t \mid s_{t+1}; \theta)} \right)^2$$
When $\mathcal{L}_{\text{TB}}(\tau) = 0$ for all complete trajectories $\tau = (s_0, s_1, \dots, s_T=x)$, the sampling distribution exactly matches the reward distribution:
$$P(x) = \frac{R(x)}{Z}$$

---

## 4. Multi-Ecosystem Implementation Matrix

| Method | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`ppo_clip_loss`** | `torch.min(r * A, clamp(r) * A)` | `jnp.minimum` + `jnp.clip` | Elementwise ratio loop | `PPO_CLIP_OBJECTIVE_WGSL` + CPU SIMD |
| **`compute_gae`** | Reversed recurrence loop | `jax.lax.scan` backward | Backward vector scan | Backward recurrence scan |
| **`gflownet_trajectory_balance_loss`** | `sum(log_pf) - sum(log_pb) - log_R` | `jnp.sum` + squared residual | Scalar accumulators | Fused flow matching |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Probability Space Ratio Underflow in PPO**:
   - *Symptom*: Ratios produce `NaN` or `0.0` on high-dimensional action policies.
   - *Cause*: Computing $r_t = \frac{\pi_\theta}{\pi_{\text{old}}}$ directly in probability space causes catastrophic floating-point underflow when probabilities are small ($< 10^{-30}$).
   - *Fix*: Always compute in log space: $r_t = \exp(\log \pi_\theta - \log \pi_{\text{old}})$.
2. **Missing GAE Terminal Masking**:
   - *Symptom*: Value functions explode and policies degrade at episode boundaries.
   - *Cause*: Failing to mask $(1 - d_t)$ allows value and advantage signals to leak across unrelated episode boundaries.
   - *Fix*: Multiply both $V(s_{t+1})$ and $\hat{A}_{t+1}$ by $(1 - d_t)$ at every step.
3. **Mode Collapse in Molecular RL**:
   - *Symptom*: An agent trained to synthesize drug candidates repeatedly emits identical variants of a single chemical scaffold.
   - *Cause*: PPO optimizes expected reward $\mathbb{E}[R]$, which inherently drives policy entropy to zero around the single global maximum.
   - *Fix*: Replace PPO with **GFlowNets** to sample structures with probability strictly proportional to reward $P(x) \propto R(x)$.

---

## 6. Cross-Domain Generalisation Matrix

1. **De Novo Drug Design & Retrosynthesis**: GFlowNets sample chemical reaction DAGs to generate diverse molecular structures matching binding affinity and synthesizability targets without mode collapse.
2. **High-Frequency Execution Algorithmic Trading**: PPO with GAE bounds optimal limit-order placement actions under severe market impact friction and order-book microstructure noise.
3. **Robotic Quadruped Locomotion**: PPO-Clip with GAE($\gamma=0.99, \lambda=0.95$) provides robust sim-to-real transfer over uneven terrain by penalizing joint acceleration jerk.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `ppo_loss_ref`: Pessimistic clipped surrogate objective over batch actions.
3. `gae_advantages_ref`: Reverse-time recurrence advantage estimation over a 10-step trajectory with an episode boundary.
4. `tb_loss_ref`: Trajectory Balance log-ratio residual across multi-step generative flow paths.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.py`](../../../jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
