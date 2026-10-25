# Safe RL: Constrained MDP, PPO-Lagrangian, CVaR, and Control Barrier Functions

## 1. Mathematical Foundations & Lineage

Standard Reinforcement Learning maximizes expected cumulative return $\mathbb{E}[\sum_t \gamma^t R(s_t, a_t)]$ without constraining safety costs, leading agents to discover risky short-cuts or exploit reward hacking. Safe Reinforcement Learning formalizes safety through explicit constraints, tail-risk bounds, and control-theoretic barrier functions:
1. **Constrained MDPs (CMDP — Altman 1999)**: Augments the state-action transition system with separate cost signals $C(s, a)$, requiring expected cumulative costs to remain strictly below budget: $J_C(\pi) \le d_{\max}$.
2. **PPO-Lagrangian (Ray et al. 2019; Stooke et al. 2020)**: Formulates CMDPs as a saddle-point minimax problem $\min_\theta \max_{\lambda \ge 0} \mathcal{L}_{\text{PPO}}(\theta) + \lambda (J_C(\theta) - d_{\max})$, executing alternating gradient descent on policy parameters $\theta$ and dual ascent on multiplier $\lambda$.
3. **Conditional Value-at-Risk (CVaR — Rockafellar & Uryasev 2000; Chow et al. 2015)**: Replaces risk-neutral expected cost with coherent tail-risk optimization, penalizing the worst $\alpha$-fraction of catastrophe trajectories.
4. **Control Barrier Functions (CBF — Ames et al. 2019)**: Bridges machine learning and formal control theory by defining an energy-like barrier function $h(x) \ge 0$. Solving a 1-step QP safety filter provides provable forward invariance: once the system starts in the safe set $\mathcal{S} = \{x : h(x) \ge 0\}$, it never exits!

---

## 2. Constrained MDP & PPO-Lagrangian

In a CMDP, policy optimization is constrained by cost budgets:
$$\max_\pi J_R(\pi) \quad \text{s.t.} \quad J_{C_k}(\pi) \le d_k, \quad k \in \{1, \dots, K\}$$

### Primal-Dual Saddle-Point Objective
Using the Lagrangian dual formulation:
$$\min_{\theta} \max_{\lambda \ge 0} \mathcal{L}(\theta, \lambda) = -\mathbb{E} \left[ \min\left( \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^R, \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}, 1\pm\epsilon\right) A^R \right) \right] + \lambda \left( \mathbb{E}\left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^C \right] \right)$$

### Dual Ascent Step
$$\lambda \leftarrow \max(0, \lambda + \eta_\lambda (J_C(\pi_\theta) - d_{\max}))$$
When the policy violates the safety budget ($J_C > d_{\max}$), $\lambda$ increases, forcing the policy to prioritize safety. When within budget ($J_C < d_{\max}$), $\lambda$ decays, allowing reward maximization.

---

## 3. Conditional Value-at-Risk (CVaR) Tail-Risk Optimization

In safety-critical tasks, optimizing average cost is insufficient when catastrophic rare events cause irreversible damage.
Given cost distribution $Z$, the Value-at-Risk $\text{VaR}_\alpha(Z)$ is the $(1-\alpha)$-quantile. The Conditional Value-at-Risk is:
$$\text{CVaR}_\alpha(Z) = \mathbb{E}[Z \mid Z \ge \text{VaR}_\alpha(Z)] = \min_{\tau \in \mathbb{R}} \left( \tau + \frac{1}{\alpha} \mathbb{E}[\max(Z - \tau, 0)] \right)$$
CVaR is a coherent risk measure satisfying monotonicity, sub-additivity, positive homogeneity, and translation invariance.

---

## 4. Control Barrier Functions (CBF-QP Safety Filter)

Let the safe set be $\mathcal{S} = \{x \in \mathcal{X} : h(x) \ge 0\}$ where $h: \mathcal{X} \to \mathbb{R}$ is continuously differentiable.
A zeroing control barrier function satisfies:
$$\sup_{u \in \mathcal{U}} \left[ \nabla h(x)^\top f(x, u) + \alpha_{\text{cbf}}(h(x)) \right] \ge 0$$
For control-affine dynamics $\dot{x} = f(x) + g(x)u$, the real-time safety filter solves a quadratic program around nominal RL action $u_{\text{nom}}$:
$$\min_{u} \frac{1}{2} \|u - u_{\text{nom}}\|_2^2 \quad \text{s.t.} \quad a^\top u + b \ge 0$$
where $a = g(x)^\top \nabla h(x)$ and $b = \nabla h(x)^\top f(x) + \alpha_{\text{cbf}} h(x)$.
The exact closed-form solution is:
$$u^* = u_{\text{nom}} + \frac{\max(0, -(a^\top u_{\text{nom}} + b))}{\|a\|_2^2 + \epsilon} a$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`ppo_lagrangian_loss`** | PPO clipped + dual cost | `jnp.minimum` + clip | Vectorized surrogate reducer | Parallel PPO-Lagrangian reducer |
| **`lagrangian_step`** | `max(0, lam + lr * err)` | Functional max | Scalar clamp update | In-place dual updater |
| **`cvar_risk_measure`** | `torch.sort` top-K mean | `jnp.sort` tail slice | Vector sort + top-K mean | GPU radix sort reducer |
| **`cbf_qp_safety_filter`** | Closed-form vector projection | Broadcast projection | Vectorized dot product | `CBF_QP_FILTER_WGSL` |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Fixed Scalar Reward Penalty Misuse**:
   - *Symptom*: Agent with reward $R - c \cdot \text{Cost}$ either ignores safety when rewards are high, or becomes completely passive when penalties dominate.
   - *Cause*: Scalarization cannot satisfy hard budget constraints; Pareto fronts on non-convex MDPs cannot be traversed with linear combinations.
   - *Fix*: Use primal-dual PPO-Lagrangian or CBF-QP safety filtering.
2. **Lagrangian Multiplier Limit Cycling**:
   - *Symptom*: Dual variable $\lambda$ oscillates wildly, alternating between 0 and 1000 without converging to stationary budget satisfaction.
   - *Cause*: Learning rate $\eta_\lambda$ too high or unnormalized advantage estimates.
   - *Fix*: Apply PID Lagrangian control or normalize cost advantage estimates.
3. **Control Saturation in CBF Filters**:
   - *Symptom*: CBF filter computes an action $u^*$ that exceeds actuator physical saturation limits ($\|u^*\|_\infty > u_{\max}$).
   - *Cause*: Incompatible barrier parameter $\alpha_{\text{cbf}}$ or high relative degree.
   - *Fix*: Incorporate actuator box constraints directly into the QP solver or design higher-order CBFs (HOCBF).

---

## 7. Real-World Engineering Applications

1. **Autonomous Driving Collision Invariance**: Interposing a CBF-QP filter between an RL planner and vehicle drive-by-wire steering to guarantee distance to leading vehicles $h(d) \ge d_{\min}$.
2. **Industrial Robot Torque Limiting**: CMDP enforcement ensuring robot arms satisfy motor thermal and torque constraints under unknown payloads.
3. **Energy Grid Voltage Regulation**: Safe RL control of microgrid distributed energy storage to satisfy strict voltage envelope tolerances.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.safetensors`:
1. `ppo_lag_ref`: PPO-Lagrangian primal policy loss.
2. `lam_updated_ref`: Updated dual Lagrangian safety multiplier.
3. `cvar_ref`: Conditional Value-at-Risk tail-risk value.
4. `u_safe_ref`: CBF-QP filtered safe control tensor.

### Synchronized Ecosystem Artifacts
- **Notebook:** [05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.ipynb](../../notebooks/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.ipynb)
- **PyTorch 2 Module:** [05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.py](../../pytorch2/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.py)
- **JAX Module:** [05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.py](../../jax/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.py)
- **Candle Module:** [05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.rs](../../candle/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.rs)
- **Hardware Kernel:** [05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.rs](../../kernels/11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.rs)
