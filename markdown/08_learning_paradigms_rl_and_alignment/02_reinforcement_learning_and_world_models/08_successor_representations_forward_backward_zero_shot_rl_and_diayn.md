# Successor Representations, Forward-Backward Zero-Shot RL, and DIAYN

## 1. Mathematical Foundations & Lineage

Successor representations and task-agnostic unsupervised skill discovery decouple environmental transition dynamics from task reward signals, allowing instant zero-shot policy evaluation and adaptation upon reward reassignment without environmental interaction.

The theoretical lineage comprises:
1. **Successor Representation (SR / Dayan 1993)**: Factored state-value functions into expected discounted state occupancies $M(s, s') = \mathbb{E}[\sum_t \gamma^t \mathbb{I}(s_t = s')]$ and immediate rewards $R(s')$, establishing a bridge between model-free and model-based RL.
2. **Successor Features & GPI (Barreto et al. 2017, 2018)**: Generalized SR to continuous multi-dimensional cumulant feature vectors $\boldsymbol{\phi}(s) \in \mathbb{R}^D$ and proved the Generalized Policy Improvement (GPI) theorem.
3. **Diversity Is All You Need (DIAYN / Eysenbach et al. 2018)**: Formulated task-free skill discovery as maximizing mutual information $\mathcal{I}(S; Z)$ between latent skills and environmental states, discovering walking, hopping, and flipping without extrinsic rewards.
4. **Forward-Backward (FB) Representation for Zero-Shot RL (Touati & Ollivier 2021, Blier et al. 2021)**: Established universal bilinear state-task representations $Q(s, a, z) = \langle F(s, a, z), z \rangle$, enabling true zero-shot RL on arbitrary continuous reward profiles.

---

## 2. Successor Features & Generalized Policy Improvement (GPI)

When environment rewards factor as a linear combination of environmental features:
$$r(s, a, s') = \boldsymbol{\phi}(s, a, s')^T \mathbf{w}$$
the action-value function under policy $\pi$ factors identically into:
$$Q^\pi(s, a; \mathbf{w}) = \boldsymbol{\psi}^\pi(s, a)^T \mathbf{w}$$
where the **Successor Features** $\boldsymbol{\psi}^\pi(s, a)$ satisfy the linear Bellman equation:
$$\boldsymbol{\psi}^\pi(s, a) = \mathbb{E}_\pi \left[ \boldsymbol{\phi}(s_t, a_t, s_{t+1}) + \gamma (1 - d_t) \boldsymbol{\psi}^\pi(s_{t+1}, \pi(s_{t+1})) \;\middle|\; s_t=s, a_t=a \right]$$

### Generalized Policy Improvement (GPI) Theorem
Given $K$ previously learned policies $\{\pi_1, \dots, \pi_K\}$ and their corresponding successor features $\{\boldsymbol{\psi}^{\pi_1}, \dots, \boldsymbol{\psi}^{\pi_K}\}$, for any unseen test-time reward parameter $\mathbf{w}'$, the upper-envelope GPI policy:
$$\pi_{\text{GPI}}(s) = \arg\max_{a} \max_{k=1,\dots,K} \boldsymbol{\psi}^{\pi_k}(s, a)^T \mathbf{w}'$$
is guaranteed to perform at least as well as the best single candidate policy:
$$V^{\pi_{\text{GPI}}}(s) \ge \max_{k=1,\dots,K} V^{\pi_k}(s) \quad \forall s \in \mathcal{S}$$

---

## 3. Forward-Backward Zero-Shot RL & DIAYN Skill Discovery

### Forward-Backward (FB) Representations
Rather than restricting tasks to linear reward weights over fixed handcrafted features $\boldsymbol{\phi}$, the Forward-Backward framework learns two dual representations:
- A forward map $F(s, a, z) \in \mathbb{R}^D$ embedding state-action trajectories conditioned on task latent $z$.
- A task representation $z$ (or backward map $B(s')$ embedding successor states).

Action values for arbitrary tasks evaluate via bilinear dot product:
$$Q_{\text{FB}}(s, a, z) = \sum_{d=1}^D F(s, a, z)_d \cdot z_d$$

### DIAYN Unsupervised Skill Discovery
DIAYN incentivizes an agent to discover diverse, easily distinguishable behaviors by maximizing the mutual information between state visits $S$ and discrete skill latent $Z \sim \text{Uniform}(1/K)$:
$$\mathcal{I}(S; Z) = \mathcal{H}(Z) - \mathcal{H}(Z \mid S)$$
Because the prior $p(z) = 1/K$ is fixed and uniform, maximizing mutual information reduces to minimizing conditional entropy $\mathcal{H}(Z \mid S)$, achieved by rewarding transitions with high discriminator log-likelihood:
$$r_{\text{DIAYN}}(s, z) = \ln q_\phi(z \mid s) - \ln p(z) = \ln q_\phi(z \mid s) + \ln K$$
The skill discriminator $q_\phi(z \mid s)$ is trained via cross-entropy loss:
$$\mathcal{L}_{\text{disc}}(\phi) = -\frac{1}{B} \sum_{b=1}^B \ln q_\phi(z_b \mid s_b)$$

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`successor_features_td_error`** | `phi + gamma*(1-done)*psi_next - psi` | `target - psi_current` | Vectorized elementwise TD delta | `SUCCESSOR_FEATURES_TD_ERROR_WGSL` |
| **`gpi_transfer_value`** | `max(matmul(psi, w), dim=-1)[0]` | `jnp.max(jnp.matmul(psi, w), -1)` | Iterative max candidate dot-product | Fused policy reduction |
| **`fb_zero_shot_value`** | `sum(forward_map * task_embedding, -1)`| `jnp.sum(f * z, axis=-1)` | Elementwise dot product | Bilinear inner product |
| **`diayn_skill_reward_and_loss`** | `log_softmax` + `gather` + mean | `jax.nn.log_softmax` indexing | Numerically stable log-sum-exp | Fused discriminator loss |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Feature Rank Collapse in Successor Features**:
   - *Symptom*: Agent transfers poorly to new tasks; GPI yields zero improvement over random policies.
   - *Cause*: Cumulant features $\boldsymbol{\phi}(s)$ become collinear during representation learning, collapsing the effective subspace dimension.
   - *Fix*: Enforce orthogonality or whitening constraints on $\boldsymbol{\phi}$ using a spectral penalty or contrastive loss.
2. **Static Posture Trap in DIAYN**:
   - *Symptom*: Discovered skills correspond to static pose freezing (e.g. agent falls into distinct corners and lies motionless).
   - *Cause*: Static terminal states maximize discriminator discriminability $\ln q_\phi(z \mid s)$ with minimal kinetic effort.
   - *Fix*: Provide the discriminator only with state differences $\Delta s = s_{t+1} - s_t$ or agent velocity coordinates rather than absolute coordinates.
3. **Bilinear Norm Explosion in Forward-Backward Maps**:
   - *Symptom*: $Q_{\text{FB}}$ values explode to $\pm 10^4$, causing extreme greedy policy instability.
   - *Cause*: Unbounded inner products $\langle F(s, a, z), z \rangle$ during multi-step Bellman-style bootstrapping.
   - *Fix*: Apply Layer Normalization to $F$ and $z$ or normalize embeddings to the unit hypersphere.

---

## 6. Cross-Domain Generalisation Matrix

1. **Zero-Shot Robotic Manipulation**: Successor features trained on random reaching rollouts immediately solve unseen target reaching tasks the instant target coordinate weights $\mathbf{w}$ are specified.
2. **Autonomous Locomotion Skill Discovery**: DIAYN discovers quadruped walking, trotting, crawling, and back-flipping behaviors without reward design.
3. **Multi-Objective High-Frequency Market Making**: Forward-Backward representations enable traders to dynamically update risk-aversion parameters $\mathbf{w}$ in milliseconds without retraining value networks.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `sf_td_error_ref`: Successor feature temporal difference vector error with terminal masking.
3. `gpi_values_ref`: Upper-envelope transfer Q-value across candidate policy features under task reward $\mathbf{w}$.
4. `fb_values_ref`: Bilinear forward-backward state-task inner product value evaluation.
5. `diayn_rewards_ref` & `diayn_loss_ref`: Discrete mutual information intrinsic reward and cross-entropy loss.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.py`](../../../jax/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
