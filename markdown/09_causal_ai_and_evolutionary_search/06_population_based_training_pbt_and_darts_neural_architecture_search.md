# Population-Based Training (PBT) and DARTS Neural Architecture Search

## 1. Mathematical Foundations & Lineage

Deep neural network performance is fundamentally governed by two meta-optimization problems: **hyperparameter scheduling** (learning rates, regularizers, augmentation intensities) and **architectural topology** (connectivity graphs, operation selection). Classical black-box methods (Grid Search, Random Search, Bayesian Optimization) require training hundreds of full models from scratch, demanding thousands of GPU hours.

The lineage of automated meta-optimization comprises:
1. **Reinforcement Learning NAS (Zoph & Le 2016)**: Trained an RNN controller via policy gradient to generate neural network DAGs, requiring 800 GPUs for 28 days ($> 20,000$ GPU-hours).
2. **Population-Based Training (PBT — Jaderberg et al. 2017)**: Unified parallel random search with sequential hyperparameter optimization by dynamically transferring weights and perturbing hyperparameters across concurrent worker processes in a single training run.
3. **BOHB (Bayesian Optimization & Hyperband — Falkner et al. 2018)**: Combined bandit-based early stopping with Gaussian process / KDE hyperparameter modeling.
4. **Differentiable Architecture Search (DARTS — Liu, Simonyan & Yang 2019)**: Formulated architecture search as continuous bilevel optimization over a weight-sharing supernet, searching $10^{18}$ candidate architectures in hours on a single device.

---

## 2. Population-Based Training (PBT)

PBT maintains a population of $P$ models $\{ (w_p, h_p) \}_{p=1}^P$ optimizing an objective $\mathcal{L}$. At periodic evaluation intervals $T_{\text{eval}}$:

### Exploit Phase (Selection)
Worker performance scores $S = [s_1, \dots, s_P]$ are ranked. For workers in the bottom percentile (e.g. bottom $20\%$ or $p_{\text{bot}} = \arg\min_i s_i$):
$$w_{p_{\text{bot}}} \leftarrow w_{p^*}, \quad h_{p_{\text{bot}}} \leftarrow h_{p^*}$$
where $p^* = \arg\max_i s_i$ is the incumbent top performer.

### Explore Phase (Mutation)
The copied hyperparameters are perturbed to discover improved regions of the hyperparameter space:
$$h_{p_{\text{bot}}} \leftarrow h_{p^*} \odot \boldsymbol{\gamma}, \quad \gamma_m \in \{0.8, 1.2\}$$
Unlike static Bayesian optimization that returns a single fixed hyperparameter configuration, PBT discovers an **online adaptive schedule** $h(t)$ that dynamically anneals learning rates and increases regularization as training progresses.

---

## 3. DARTS Continuous Relaxation & Bilevel Optimization

In DARTS, a neural architecture is represented as a Directed Acyclic Graph (DAG) with $N$ nodes. Each directed edge $(i, j)$ represents an information flow applying an operation $o \in \mathcal{O}$ (e.g. identity, $3\times3$ separable conv, $5\times5$ dilated conv, zero).

### Continuous Softmax Relaxation
To make the categorical choice of operation differentiable, DARTS relaxes the discrete selection into a continuous softmax mixture:
$$\bar{o}^{(i, j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i, j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i, j)})} o(x)$$
where $\alpha^{(i, j)} \in \mathbb{R}^{|\mathcal{O}|}$ are continuous architectural parameters.

### Bilevel Optimization Problem
The joint optimization problem is:
$$\min_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \quad \text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)$$
Using a first-order approximation $w^*(\alpha) \approx w$, gradients evaluate to:
$$\nabla_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \approx \nabla_\alpha \mathcal{L}_{\text{val}}(w, \alpha)$$

### Discrete Architecture Decoding
At the conclusion of search, the continuous graph is discretized by selecting the argmax operation on each edge:
$$o^*(i, j) = \arg\max_{o \in \mathcal{O}} \alpha_o^{(i, j)}$$

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`pbt_exploit_explore_step`** | Vectorized argmax/argmin copy & scale | Functional `at[bot].set` | In-place rank selection & perturbation | Fused exploit-explore dispatch |
| **`darts_mixed_op_forward`** | `softmax` + broadcast sum reduction | `jax.nn.softmax` contraction | Softmax broadcast-mul sum | `DARTS_MIXED_OP_WGSL` |
| **`darts_discrete_decode`** | `argmax(alpha, -1)` slice extraction | `jnp.argmax` index lookup | Row-wise max extraction & stack | Fused argmax operator selector |
| **`darts_architecture_entropy`** | `-sum(w * log(w + eps))` | `-jnp.sum(w * jnp.log(w + eps))` | Affine log dot reduction | Vectorized entropy reduction |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **DARTS Skip-Connection Collapse**:
   - *Symptom*: During search, parameter-free skip connections dominate every edge ($\alpha_{\text{skip}} \to 1$), leading to a degenerate architecture that fails to generalize after discretization.
   - *Cause*: Identity skip connections have no internal parameters and transmit gradients without attenuation early in training, accumulating higher architectual gradient magnitudes than parameterized convolutions.
   - *Fix*: Apply operation dropout, auxiliary entropy regularization $H(\alpha)$, or enforce a hard constraint on the maximum number of skip connections per cell.
2. **PBT Population Diversity Extinction**:
   - *Symptom*: All workers converge to identical weights and hyperparameters after a few exploit cycles.
   - *Cause*: Exploitation frequency $T_{\text{eval}}$ is too high or perturbation factor $\gamma$ is too small, destroying exploratory variance.
   - *Fix*: Increase evaluation interval $T_{\text{eval}}$ and introduce exploration resampling (with probability 0.2, draw $h$ uniformly at random from the prior).
3. **Discretization Performance Gap**:
   - *Symptom*: The supernet achieves high validation accuracy under continuous mixture weights, but accuracy collapses after argmax discretization.
   - *Cause*: Continuous relaxation relies on co-adaptation among multiple operations on the same edge.
   - *Fix*: Anneal softmax temperature $\tau \to 0$ over training epochs: $w_o = \frac{\exp(\alpha_o / \tau)}{\sum \exp(\alpha_{o'} / \tau)}$.

---

## 6. Real-World Engineering Applications

1. **Massive Reinforcement Learning (AlphaStar / OpenAI Five)**: PBT concurrently trains populations of RL agents with dynamic learning rates, entropy bonuses, and reward shaping weights across distributed clusters.
2. **Mobile & Edge Device Vision Architecture Search**: Constraining DARTS with hardware latency penalties $\mathcal{L} + \lambda \cdot \text{Latency}(\alpha)$ to discover optimal sub-millisecond MobileNet topologies.
3. **Automated LLM Fine-Tuning Schedules**: Discovering dynamic LoRA rank, learning rate, and weight decay schedules in a single distributed training run.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `pbt_weights_updated_ref` & `pbt_hyperparams_updated_ref`: PBT exploit-and-explore weight copying and hyperparameter mutation.
3. `darts_mixed_out_ref` & `darts_softmax_weights_ref`: Continuous softmax mixture forward pass.
4. `darts_selected_ops_ref` & `darts_discrete_out_ref`: Discrete architecture argmax decoding.
5. `darts_entropy_ref`: Architecture parameter entropy.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.py`](../../pytorch2/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.py)
- **JAX Reference**: [`jax/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.py`](../../jax/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.py)
- **Candle Reference**: [`candle/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.rs`](../../candle/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.rs)
- **Kernels**: [`kernels/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.rs`](../../kernels/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.rs)
- **Interactive Notebook**: [`notebooks/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.ipynb`](../../notebooks/09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../INDEX_BY_GOAL.md)
