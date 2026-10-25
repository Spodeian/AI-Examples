# Hierarchical RL: Options, FeUdal Networks, HER, and POMDP Active Inference

## 1. Mathematical Foundations & Lineage

Hierarchical Reinforcement Learning (HRL) resolves the curse of dimensionality and the challenge of sparse reward signals across long temporal horizons by decomposing complex tasks into multi-scale temporal and behavioral abstractions.

The conceptual lineage spans:
1. **Semi-Markov Decision Processes (SMDPs / Sutton, Precup & Singh 1999)**: Formalized the Options framework $\langle \mathcal{I}_\omega, \pi_\omega, \beta_\omega \rangle$, extending discrete time step transitions to temporally extended macro-actions.
2. **Option-Critic Architecture (Bacon, Harb & Precup 2017)**: Derived end-to-end policy gradient theorems for intra-option policies and stochastic option termination functions without manual sub-goal engineering.
3. **FeUdal Networks (Vezhnevets et al. 2017)**: Decoupled managerial temporal abstraction from worker motor execution, employing transition-direction cosine rewards in latent state space.
4. **Hindsight Experience Replay (HER / Andrychowicz et al. 2017)**: Formulated multi-goal reinforcement learning by relabeling failed trajectories with achieved terminal states as counterfactual goals, transforming sparse signals into dense learning signals.
5. **Active Inference & Variational POMDPs (Friston 2006, 2010)**: Unified Bayesian perception, belief filtering, and goal-directed action as the continuous minimization of Variational Free Energy under partial observability.

---

## 2. The Options Framework & Option-Critic Architecture

An option $\omega \in \Omega$ consists of:
- An initiation set $\mathcal{I}_\omega \subseteq \mathcal{S}$
- An intra-option policy $\pi_\omega(a \mid s)$
- A termination probability function $\beta_\omega(s) \in [0, 1]$

### Option Value Functions & Advantage
The option-value function $Q_\Omega(s, \omega)$ and state value $V(s)$ are defined as:
$$V(s) = \sum_{\omega \in \Omega} \pi_\Omega(\omega \mid s) Q_\Omega(s, \omega)$$
$$U(s', \omega) = (1 - \beta_\omega(s')) Q_\Omega(s', \omega) + \beta_\omega(s') V(s')$$

### Option Termination Gradient with Deliberation Cost
To prevent option collapse (where options either terminate at every step or never terminate), Bacon et al. introduce a deliberation cost $\eta > 0$:
$$\nabla_\theta \mathcal{L}_{\text{term}}(\theta) = \beta_\omega(s) \left[ Q_\Omega(s, \omega) - V(s) + \eta \right]$$
When the current option value $Q_\Omega(s, \omega)$ falls below $V(s) - \eta$, the gradient pushes $\beta_\omega(s) \to 1$, triggering termination and handing control back to the meta-policy.

---

## 3. FeUdal Directional Rewards, HER & Active Inference

### FeUdal Networks Latent Directional Rewards
In FeUdal Networks, the Manager generates a latent directional vector $g_t$ over a temporal horizon $c$. The Worker receives an intrinsic reward proportional to directional alignment:
$$r_t^W = \frac{(\mathbf{s}_{t+c} - \mathbf{s}_t) \cdot \mathbf{g}_t}{\max \left( \|\mathbf{s}_{t+c} - \mathbf{s}_t\|_2 \|\mathbf{g}_t\|_2, \, \epsilon \right)}$$
This cosine directional formulation ensures that the Worker is incentivized to make state progress along the Manager's proposed trajectory regardless of absolute distance scale.

### Hindsight Experience Replay (HER)
In sparse goal-conditioned RL, trajectories that fail to reach the desired goal $g^*$ provide zero reward. HER samples an achieved state $s_T$ from the trajectory and relabels the experience with $g = s_T$:
$$r(s, g) = \begin{cases} 1.0 & \text{if } \|s - g\|_2 \le \delta \\ 0.0 & \text{otherwise} \end{cases}$$
Every rollout yields positive reinforcement for the state it actually attained, bypassing reward sparsity.

### Active Inference & Variational Free Energy
In Partially Observable Markov Decision Processes (POMDPs), an agent maintains an approximate posterior belief $q_\phi(s)$ over hidden states $s$ given observations $o$. Active inference defines Variational Free Energy $\mathcal{F}$:
$$\mathcal{F} = D_{\text{KL}}\left( q_\phi(s) \,\|\, p(s) \right) - \mathbb{E}_{q_\phi(s)} \left[ \ln p(o \mid s) \right]$$
Minimizing $\mathcal{F}$ simultaneously bounds the surprise of sensations (perception) and steers beliefs toward preferred prior sensory configurations (action).

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`her_goal_relabeling`** | `(norm(s - g) <= threshold).float()` | `(norm <= threshold).astype(jnp.float32)` | L2 norm threshold accumulator | Branchless distance comparison |
| **`feudal_directional_reward`** | `sum(ds * g) / clamp(norm * norm)` | `dot / jnp.maximum(denom, eps)` | Iterative vector dot + L2 norm | `FEUDAL_DIRECTIONAL_REWARD_WGSL` |
| **`option_critic_termination_loss`** | `mean(beta * (q - v + cost))` | `jnp.mean(beta * (q - v + cost))` | Scalar deliberation loss accumulator | Fused reduction kernel |
| **`active_inference_free_energy`** | Diagonal Gaussian KL + Gaussian NLL | JIT-compiled analytical KL + NLL | Elementwise normal log-likelihood | Fused variational POMDP kernel |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Option Degeneration / Collapse**:
   - *Symptom*: Intra-option policies reduce to single-step primitive actions ($\beta_\omega \approx 1$ everywhere) or never switch ($\beta_\omega \approx 0$).
   - *Cause*: Without a deliberation penalty $\eta$, switching options incurs zero cost, making rapid micro-switching or total inertia arbitrarily favored.
   - *Fix*: Calibrate deliberation cost $\eta \in [0.01, 0.1]$ to penalize switching overhead.
2. **Subgoal Drift & Managerial Instability in FeUdal Networks**:
   - *Symptom*: Manager requests impossible transition directions $\mathbf{g}_t$, leaving Worker unable to earn rewards.
   - *Cause*: High unconstrained latent variance in the Manager output.
   - *Fix*: Normalize latent space embeddings and constrain $g_t$ to unit norm with layer normalization.
3. **HER Extreme Sparse Imbalance**:
   - *Symptom*: Relabeled replay buffers overwhelm environment goals, causing the agent to master trivial self-selected goals while failing original targets.
   - *Cause*: Excessive hindsight ratio $k$.
   - *Fix*: Maintain balanced sampling ratio (e.g. 80% hindsight relabeled, 20% original target goals).

---

## 6. Cross-Domain Generalisation Matrix

1. **Robotic Manipulation (Sparse-Reward Peg-in-Hole)**: HER transforms millions of failed insertion attempts into successful relabeled trajectories, achieving 99.8% insertion success with no reward shaping.
2. **Hierarchical Maze Navigation & Long Horizon Planning**: Option-Critic discovers specialized sub-policies for corridor traversal, door opening, and room exploration without manual partitioning.
3. **Active Epistemic Foraging in Medical Diagnosis**: Variational Free Energy active inference agents balance diagnostic information gain (epistemic value) against patient treatment costs (pragmatic value).

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `her_rewards_ref`: Hindsight goal achievement indicator under $L_2$ threshold $\delta = 0.1$.
3. `feudal_rewards_ref`: Directional cosine similarity reward with numerical epsilon clamping.
4. `option_loss_ref`: Option-Critic termination gradient loss with deliberation cost $\eta = 0.01$.
5. `free_energy_ref`: Exact diagonal Gaussian KL divergence plus observation reconstruction negative log-likelihood.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.py`](../../../jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
