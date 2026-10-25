# Evolution Strategies: CMA-ES, OpenAI ES, and Particle Swarm Optimization (PSO)

## 1. Mathematical Foundations & Lineage

Black-box zero-order optimization operates when analytical gradients $\nabla_\theta f(\theta)$ are unavailable, prohibitively expensive, discontinuous, or plagued by deceptive local extrema. Rather than backpropagating through computational graphs, evolutionary and swarm algorithms evaluate candidate parameter vectors across a population and iteratively adapt a sampling distribution or swarm trajectories toward high-fitness regions.

The lineage of zeroth-order optimization comprises:
1. **Rechenberg & Schwefel (1970s)**: Founded $(1+1)$-ES and $(\mu, \lambda)$-ES, introducing mutation step-size adaptation via the $1/5$th success rule.
2. **Particle Swarm Optimization (PSO — Kennedy & Eberhart 1995)**: Modeled collective animal swarm behavior where particles navigate continuous search spaces via personal-best and swarm-global-best attractors.
3. **Covariance Matrix Adaptation Evolution Strategy (CMA-ES — Hansen & Ostermeier 1996, Hansen 2006)**: The gold standard in non-linear derivative-free continuous optimization; dynamically adapts the full second-order covariance matrix $C \in \mathbb{R}^{d \times d}$ to model objective curvature, invariance to monotonic fitness transformations, and coordinate transformations.
4. **Natural Evolution Strategies (NES — Wierstra et al. 2008, 2014)**: Grounded ES in information geometry, computing natural gradients on search distribution parameters using the Fisher Information Matrix.
5. **OpenAI ES (Salimans et al. 2017)**: Demonstrated massive parallel scalability of antithetic NES for deep reinforcement learning, matching policy gradient RL without backpropagation or value functions.

---

## 2. OpenAI ES Antithetic Gradient Estimator

OpenAI ES optimizes the expected return over a Gaussian perturbation distribution $p_\theta(\psi) = \mathcal{N}(\psi; \theta, \sigma^2 I)$:
$$J(\theta) = \mathbb{E}_{\psi \sim \mathcal{N}(\theta, \sigma^2 I)} [F(\psi)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [F(\theta + \sigma \epsilon)]$$

Applying the log-derivative trick (REINFORCE / score function):
$$\nabla_\theta J(\theta) = \nabla_\theta \int F(\psi) p_\theta(\psi) d\psi = \int F(\psi) \nabla_\theta \log p_\theta(\psi) p_\theta(\psi) d\psi = \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [F(\theta + \sigma \epsilon) \cdot \epsilon]$$

### Antithetic Sampling for Variance Reduction
Sampling mirrored pairs $(\epsilon_i, -\epsilon_i)$ creates an antithetic Monte Carlo estimator that cancels odd-order noise terms:
$$\nabla_\theta J(\theta) \approx \frac{1}{2 P \sigma} \sum_{i=1}^P \left( F(\theta + \sigma \epsilon_i) - F(\theta - \sigma \epsilon_i) \right) \epsilon_i$$
This estimator enables distributed parallel evaluation across thousands of worker CPU threads with zero inter-worker gradient communication — workers only send scalar fitness values back to the orchestrator.

---

## 3. CMA-ES Covariance Matrix Adaptation

CMA-ES generates candidate populations by sampling from an evolving multivariate normal distribution:
$$x_k \sim \mathcal{N}(\mu, \sigma^2 C), \quad k = 1, \dots, \lambda$$

### Elite Selection & Recombination
Given population fitness values, candidates are sorted such that $F(x_{1:\lambda}) \le F(x_{2:\lambda}) \le \dots \le F(x_{\lambda:\lambda})$. The new population center is the weighted mean of the top $\mu_{\text{elite}}$ individuals:
$$\mu^{(t+1)} = \sum_{i=1}^{\mu_{\text{elite}}} w_i x_{i:\lambda}, \quad \text{where } \sum_{i=1}^{\mu_{\text{elite}}} w_i = 1, \; w_1 \ge w_2 \ge \dots > 0$$

### Rank-$\mu$ Covariance Matrix Update
Defining step displacements $y_i = x_{i:\lambda} - \mu^{(t)}$, the empirical covariance of the successful steps is:
$$C_\mu = \sum_{i=1}^{\mu_{\text{elite}}} w_i y_i y_i^T$$
The covariance matrix updates via geometric decay with learning rate $c_{\text{cov}}$:
$$C^{(t+1)} = (1 - c_{\text{cov}}) C^{(t)} + c_{\text{cov}} C_\mu$$
By accumulating step covariances over successive generations, $C$ learns an inverse approximation of the objective function's Hessian matrix without evaluating second derivatives.

---

## 4. Particle Swarm Optimization (PSO)

In PSO, a swarm of $N$ particles explores a $D$-dimensional space. Each particle $i$ possesses a position $x_i \in \mathbb{R}^D$, velocity $v_i \in \mathbb{R}^D$, and historical individual best position $p_i \in \mathbb{R}^D$. The swarm tracks the global best position $g^* = \arg\max_{p_i} F(p_i)$.

### Kinematic Updates
At iteration $t+1$, particle velocities and positions update according to:
$$v_i^{(t+1)} = w \cdot v_i^{(t)} + c_1 r_1 \odot (p_i^{(t)} - x_i^{(t)}) + c_2 r_2 \odot (g^{*(t)} - x_i^{(t)})$$
$$x_i^{(t+1)} = x_i^{(t)} + v_i^{(t+1)}$$
where:
- $w \in [0.4, 0.9]$ is the inertia weight controlling momentum,
- $c_1, c_2 \approx 1.4$ are cognitive and social acceleration coefficients,
- $r_1, r_2 \sim \mathcal{U}(0, 1)^D$ inject stochastic exploratory perturbations.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`openai_es_gradient_estimate`** | `sum((pos - neg) * eps) / (2 P sigma)` | JAX vectorized broadcast & sum | `sum(broadcast_mul).affine(...)` | `OPENAI_ES_ANTITHETIC_WGSL` |
| **`cma_es_mean_and_cov_step`** | `y.T @ (w * y)` rank-$\mu$ outer product | `jnp.dot` outer contraction | In-place transpose matmul & affine | Parallel rank-$\mu$ contraction |
| **`pso_velocity_and_position_step`**| Vectorized cognitive + social updates | Functional elementwise update | Chained affine and broadcast ops | Parallel particle kinematics |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Fitness Variance Collapse in Deep ES**:
   - *Symptom*: Estimated gradient norms collapse to zero or fluctuate violently as parameter dimension $D$ exceeds $10^6$.
   - *Cause*: In high dimensions, random perturbations $\sigma \epsilon$ land in catastrophic policy regimes, rendering fitness differentials $F^+ - F^-$ nearly uniform or dominated by extreme outlier returns.
   - *Fix*: Apply fitness rank transformation (virtual centering: replace raw fitness with normalized percentiles in $[-0.5, 0.5]$).
2. **CMA-ES $O(D^2)$ Memory and $O(D^3)$ Eigendecomposition Bottlenecks**:
   - *Symptom*: Out-of-memory or prohibitive computation per generation when $D > 10,000$.
   - *Cause*: Full covariance matrix storage requires $D^2$ elements and Cholesky/eigen-decomposition takes $O(D^3)$ operations.
   - *Fix*: Use Sep-CMA-ES (diagonal covariance $O(D)$) or LM-CMA-ES (limited-memory Cholesky).
3. **PSO Premature Swarm Convergence**:
   - *Symptom*: All particles collapse into a suboptimal local minimum early in training with stagnant velocity $\|v\| \to 0$.
   - *Cause*: Insufficient diversity in social attraction when $g^*$ stagnates.
   - *Fix*: Implement constriction factor $\chi$, dynamic inertia decay ($w: 0.9 \to 0.4$), or random re-initialization of stagnant particles.

---

## 7. Real-World Engineering Applications

1. **Robotics Locomotion & Contact Dynamics**: Reinforcement learning policies in non-differentiable physics simulators (MuJoCo, Isaac Sim with rigid contact discontinuities) optimize stably via OpenAI ES without simulation gradient instability.
2. **Analog Circuit & Antenna Design**: Tuning non-linear geometric and impedance parameters where physics simulations lack analytical Jacobians.
3. **Prompt Evolution & LLM Hyperparameter Search**: Discretized prompt search and hyperparameter configuration grids optimized via CMA-ES and PSO.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `es_grad_ref`: OpenAI ES antithetic gradient estimation vector.
3. `cma_mu_new_ref` & `cma_cov_new_ref`: CMA-ES recombination mean and rank-$\mu$ covariance matrix.
4. `pso_v_next_ref` & `pso_x_next_ref`: PSO updated particle velocities and positions.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 9. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.py`](../../pytorch2/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.py)
- **JAX Reference**: [`jax/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.py`](../../jax/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.py)
- **Candle Reference**: [`candle/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.rs`](../../candle/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.rs)
- **Kernels**: [`kernels/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.rs`](../../kernels/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.rs)
- **Interactive Notebook**: [`notebooks/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.ipynb`](../../notebooks/09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../INDEX_BY_GOAL.md)
