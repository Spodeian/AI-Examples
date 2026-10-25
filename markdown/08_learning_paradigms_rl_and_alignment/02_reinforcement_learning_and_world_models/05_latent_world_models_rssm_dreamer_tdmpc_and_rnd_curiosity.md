# Latent World Models: RSSM, Dreamer, TD-MPC2, and RND Curiosity

## 1. Mathematical Foundations & Lineage

Model-based reinforcement learning builds internal representations of environmental transition dynamics to simulate futures, accelerate sample efficiency, and optimize policies via analytic backpropagation through imagined time horizons.

The lineage of latent world models traces:
1. **Dyna-Q (Sutton 1990)**: Integrates real experience with model-generated synthetic rollouts.
2. **World Models (Ha & Schmidhuber 2018)**: Decouples perception (VAE) from temporal dynamics (MDN-RNN), learning policies entirely inside compact latent dreams.
3. **PlaNet & DreamerV1–V3 (Hafner et al. 2019, 2020, 2023)**: Introduces the Recurrent State-Space Model (RSSM) with KL balancing and symlog scaling, achieving human-level Atari and robotics mastery across diverse domains without hyperparameter tuning.
4. **Random Network Distillation (RND / Burda et al. 2018)**: Derives non-vanishing intrinsic curiosity rewards from prediction errors against a frozen random projection target network.
5. **TD-MPC & TD-MPC2 (Hansen et al. 2022, 2024)**: Eliminates heavy pixel decoders by learning task-oriented latent dynamics trained solely with temporal difference value and reward prediction objectives.

---

## 2. Recurrent State-Space Models (RSSM) & KL Balancing

An RSSM partitions the latent state at time $t$ into a deterministic recurrent path $h_t$ and a stochastic state $z_t$:
- Recurrent update: $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$
- Stochastic prior: $\hat{z}_t \sim p_\theta(z_t \mid h_t)$
- Stochastic posterior: $z_t \sim q_\phi(z_t \mid h_t, x_t)$

### The Posterior Collapse Trap & KL Balancing
In standard VAE training, the variational objective minimizes symmetric KL divergence $D_{\text{KL}}(q_\phi(z_t \mid h_t, x_t) \| p_\theta(z_t \mid h_t))$. Early in training, the posterior $q_\phi$ collapses to the untrained random prior $p_\theta$ faster than the prior learns dynamics, causing the encoder to discard observation features $x_t$.

Hafner et al. (2020) resolve this via **KL Balancing**:
$$\mathcal{L}_{\text{KL}}(\theta, \phi) = \beta_{\text{prior}} \mathbb{E} \left[ D_{\text{KL}}(\text{sg}[q_\phi] \| p_\theta) \right] + \beta_{\text{post}} \mathbb{E} \left[ D_{\text{KL}}(q_\phi \| \text{sg}[p_\theta]) \right]$$
Setting $\beta_{\text{prior}} = 0.8$ and $\beta_{\text{post}} = 0.2$ trains the prior dynamics $p_\theta$ toward the accurate posterior representations four times faster than regularizing the posterior toward the prior, completely avoiding posterior collapse!

---

## 3. TD-MPC2 Latent Imagination & RND Intrinsic Curiosity

### TD-MPC2 Decoder-Free Latent Rollouts
Rather than reconstructing full high-dimensional pixel observations $\hat{x}_t = g(z_t)$, TD-MPC2 learns an implicit representation $z_t = f(x_t)$ with three lightweight heads:
1. Latent dynamics: $z_{t+1} = T(z_t, a_t)$
2. Reward predictor: $\hat{r}_t = R(z_t, a_t)$
3. Value predictor: $\hat{Q}_t = Q(z_t, a_t)$

During planning, Model Predictive Control (via MPPI or CEM) simulates $H$-step rollouts entirely in latent space:
$$G(a_{0:H-1}) = \sum_{t=0}^{H-1} \gamma^t \hat{r}_t + \gamma^H \hat{Q}(z_H, a_H)$$
avoiding pixel decoding overhead and allowing real-time 50Hz control on physical robot hardware.

### Random Network Distillation (RND) Curiosity
In sparse-reward environments, exploration stalls. RND establishes an intrinsic reward using a fixed, randomly initialized target network $f_{\text{target}}(s)$ and a trainable predictor network $\hat{f}_\theta(s)$:
$$r_i(s) = \frac{1}{D} \sum_{d=1}^D (\hat{f}_\theta(s)_d - f_{\text{target}}(s)_d)^2$$
Novel states produce large prediction errors $r_i \gg 0$, while frequently visited states allow $\hat{f}_\theta$ to converge to $f_{\text{target}}$, driving intrinsic rewards to zero without suffering from the "noisy TV" stochastic distraction problem.

---

## 4. Multi-Ecosystem Implementation Matrix

| Method | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`gaussian_kl_divergence`** | Analytical diagonal Gaussian sum | `jnp.sum` variance ratio | Elementwise float sum | `GAUSSIAN_KL_DIVERGENCE_WGSL` |
| **`rssm_kl_balancing`** | Stop-gradient detached priors | `jax.lax.stop_gradient` | Balanced scalar mean | Fused KL reduction |
| **`rnd_intrinsic_reward`** | `mean((pred - target)^2, dim=-1)` | `jnp.mean` squared diff | Row mean accumulator | Parallel MSE reduction |
| **`tdmpc_latent_rollout_return`** | `sum(gamma^t * r) + gamma^H * Q` | Power discounts + sum | Discount accumulator | Recurrent rollout return |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Posterior Collapse in Symmetric RSSM**:
   - *Symptom*: Latent states $z_t$ become independent of inputs $x_t$, world model predicts static blurry averages.
   - *Cause*: Symmetric KL divergence drives $q_\phi \to p_\theta$ before dynamics are established.
   - *Fix*: Apply **KL Balancing** ($\beta_{\text{prior}}=0.8, \beta_{\text{post}}=0.2$).
2. **Noisy TV Distraction in Forward Dynamic Curiosity**:
   - *Symptom*: Curiosity-driven agent stares indefinitely at random noise on a screen.
   - *Cause*: Next-state prediction curiosity $\min \| \hat{s}_{t+1} - s_{t+1} \|^2$ rewards unlearnable environmental aleatoric stochasticity.
   - *Fix*: Use **RND** with static fixed target projections, where intrinsic reward measures epistemic state novelty.
3. **Compound Model Exploitation in Long Latent Horizons**:
   - *Symptom*: Policy generates hallucinated infinite reward trajectories in imagination that fail on the real system.
   - *Cause*: Compounding errors in $z_{t+1} = T(z_t, a_t)$ over horizons $H > 20$.
   - *Fix*: Bound rollout horizon $H \in [5, 10]$ and anchor trajectory tails with terminal value estimates $Q(z_H, a_H)$ (TD-MPC2).

---

## 6. Cross-Domain Generalisation Matrix

1. **Legged Robotics Sim-to-Real Transfer**: DreamerV3 trains locomotion policies entirely inside latent simulations, mastering physical quadruped walking with under 1 hour of real hardware rollouts.
2. **Autonomous Chemical Reaction Vessel Optimization**: Latent world models forecast continuous temperature and pressure dynamics, avoiding catastrophic thermal runaway.
3. **Video Game Agent Exploration (Montezuma's Revenge)**: RND intrinsic curiosity drives exploration through labyrinthine dungeon puzzles where external rewards are absent for thousands of steps.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `kl_div_ref`: Analytical diagonal Gaussian relative entropy between latent distributions.
3. `kl_balanced_ref`: Weighted 0.8/0.2 KL balanced objective.
4. `rnd_rewards_ref`: Mean squared feature distillation error.
5. `tdmpc_return_ref`: Discounted latent imagination trajectory return over horizon $H=5$.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.py`](../../../jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
