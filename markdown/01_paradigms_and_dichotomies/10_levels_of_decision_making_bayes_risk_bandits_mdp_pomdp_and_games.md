# Levels of Decision Making: Bayes Risk, Bandits, MDP, POMDP, and Games (Topic 10)

## 1. Executive Summary & Intuitive Mental Model

Machine learning architectures are frequently conceptualized solely as function approximators $\hat{y} = f_\theta(x)$ trained to minimize empirical surrogate losses (such as Cross-Entropy or Mean Squared Error). However, autonomous intelligence operates along a **hierarchy of decision problems** defined by the nature of environmental feedback, temporal horizons, information asymmetry, and competitive interaction.

The **6-Level Decision Hierarchy** formalizes this spectrum:
- **Level 0 (Static Bayes Risk & Chow's Reject Option):** Given calibrated conditional probabilities $P(Y|X)$, actions are selected in a single step under an asymmetric loss/cost matrix $C(a, y)$ with optimal abstention (rejection).
- **Level 1 (Contextual Bandits & Counterfactual OPE):** Actions influence immediate rewards with partial bandit feedback (unobserved counterfactual outcomes for unchosen actions), addressed via Inverse Propensity Scoring (IPS) and Doubly Robust estimators.
- **Level 2 (Closed-Loop Markov Decision Processes - MDP):** Actions alter the environment's state transition distribution $P(s'|s, a)$ across extended horizons, governed by Bellman optimality and dynamic programming.
- **Level 3 (Partially Observable MDP - POMDP & Epistemic Active Inference):** True environmental states $s$ are hidden; actions serve dual roles—epistemic (gathering information to compress belief entropy) and pragmatic (accumulating extrinsic value).
- **Level 4 (Hierarchical Options & Semi-MDP):** Temporal abstraction enables multi-step macro-actions with distinct termination conditions $\beta_\omega(s)$.
- **Level 5 (Game-Theoretic Equilibria & Mechanism Design):** Multiple rational agents interact strategically (Nash, Stackelberg bi-level leader-follower equilibria), extending to differentiable auction mechanism design (RegretNet).

### Decision Boundary: When NOT to Use
> [!WARNING]
> Do NOT formulate a Level 0 static classification problem as an RL MDP (Level 2). Reinforcement learning introduces high estimator variance, exploration overhead, and sample inefficiency when actions do not alter the subsequent state distribution. Conversely, do NOT apply static greedy Bayes decision rules (Level 0) to POMDPs (Level 3), as static rules neglect epistemic actions necessary for state identification.

---

## 2. Mathematical Foundations & First-Principles Derivations

### Level 0: Static Bayes Risk & Chow's Reject Option
Let $x \in \mathcal{X}$ be an observation and $y \in \mathcal{Y} = \{1, \dots, K\}$ be the true state/class. Given posterior $P(y|x)$ and action cost matrix $C \in \mathbb{R}^{|\mathcal{A}| \times K}$, the conditional Bayes risk of action $a \in \mathcal{A}$ is:
$$R(a|x) = \sum_{y=1}^K C(a, y) P(y|x)$$
The optimal Bayes action is:
$$a^*(x) = \arg\min_{a \in \mathcal{A}} R(a|x)$$
Under **Chow's optimal abstention rule** (Chow, 1970) with rejection cost $C_{\text{reject}}$, the decision maker rejects making a definitive prediction if the minimal expected risk exceeds $C_{\text{reject}}$:
$$\text{Decision}(x) = \begin{cases} a^*(x) & \text{if } \min_{a} R(a|x) \le C_{\text{reject}} \\ a_{\text{reject}} & \text{otherwise} \end{cases}$$

### Level 1: Contextual Bandits & Counterfactual Off-Policy Evaluation (OPE)
In contextual bandits, a logging policy $\mu(a|x)$ collected a historical dataset $\mathcal{D} = \{(x_i, a_i, r_i, p_i)\}_{i=1}^N$ with propensities $p_i = \mu(a_i|x_i)$. To evaluate a new target policy $\pi(a|x)$ without deployment:
1. **Inverse Propensity Scoring (IPS):**
   $$\hat{V}_{\text{IPS}}(\pi) = \frac{1}{N} \sum_{i=1}^N \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)} r_i$$
2. **Self-Normalized IPS (SNIPS):**
   $$\hat{V}_{\text{SNIPS}}(\pi) = \frac{\sum_{i=1}^N w_i r_i}{\sum_{i=1}^N w_i}, \quad w_i = \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)}$$
3. **Doubly Robust (DR) Estimator:**
   Combining a direct reward model $\hat{Q}(x, a)$ with IPS correction:
   $$\hat{V}_{\text{DR}}(\pi) = \frac{1}{N} \sum_{i=1}^N \left( \hat{Q}(x_i, \pi(x_i)) + \frac{\mathbb{I}(a_i = \pi(x_i))}{\mu(a_i|x_i)} [r_i - \hat{Q}(x_i, a_i)] \right)$$

### Level 2: Closed-Loop MDP & Bellman Optimality
An MDP is a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. The Bellman optimality equation for state value $V^*(s)$ is:
$$V^*(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s'|s, a) V^*(s') \right]$$
The contractive Bellman backup operator $T[V]$ satisfies $\|T[U] - T[V]\|_\infty \le \gamma \|U - V\|_\infty$.

### Level 3: POMDP Belief-State Filtering
In a POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma)$, state $s$ is unobserved. Given observation $o \in \Omega$ and previous belief $b_t(s) = P(s_t = s | h_t)$, the Bayesian belief update is:
$$\tilde{b}_{t+1}(s') = \mathcal{O}(o_{t+1}|s', a_t) \sum_{s \in \mathcal{S}} \mathcal{P}(s'|s, a_t) b_t(s)$$
$$b_{t+1}(s') = \frac{\tilde{b}_{t+1}(s')}{\sum_{s'' \in \mathcal{S}} \tilde{b}_{t+1}(s'')}$$

### Level 5: Game-Theoretic Mechanism Design & Auctions
In a differentiable 2-bidder auction with bids $b \in \mathbb{R}^2$, soft allocations are computed via temperature $\tau$:
$$p_i(b) = \frac{e^{b_i / \tau}}{\sum_j e^{b_j / \tau}}$$
Under the second-price payment rule, each bidder pays the allocation probability multiplied by the second-highest bid $\min(b)$:
$$P_i(b) = p_i(b) \cdot \min(b_1, b_2)$$

---

## 3. Tensor Mechanics & Memory Layout Architecture

```mermaid
flowchart TD
    subgraph L0["Level 0: Bayes Risk"]
        P["Posterior P(Y|X) [B, K]"] --> BR["Cost Matrix C [A, K]"]
        BR --> ER["Expected Risk [B, A]"]
        ER --> CR["Chow Reject Threshold"]
    end

    subgraph L1["Level 1: Contextual Bandits OPE"]
        D["Dataset (x, a, r, mu)"] --> W["IPS Weights w = pi/mu"]
        W --> DR["Doubly Robust = Q_hat + w*(r - Q_hat)"]
    end

    subgraph L2["Level 2: Closed-Loop MDP"]
        V["Value V(s) [S]"] --> BO["Bellman Backup Q(s, a) = R + gamma*P*V"]
        BO --> NV["V'(s) = max_a Q(s, a)"]
    end

    subgraph L3["Level 3: POMDP Belief Filter"]
        B["Belief b(s) [S]"] --> PRED["Predict: b_pred = b * T_a"]
        PRED --> UPD["Update: b' = O(o|s') * b_pred / Z"]
    end
```

---

## 4. Multi-Ecosystem Comparative Implementation Matrix

| Metric / Feature | PyTorch 2.x | JAX / Equinox | Candle Rust | Hardware Kernels |
| :--- | :--- | :--- | :--- | :--- |
| **L0 Bayes Risk** | `torch.matmul(p, C.t())` | `jnp.dot(p, C.T)` | `p.matmul(&c.t()?)` | Vectorized CPU / WGSL SIMD |
| **L1 Doubly Robust** | `q + w * (r - q)` | `q + w * (r - q)` | Elementwise arithmetic | Fused kernel loop |
| **L2 Bellman Backup** | `torch.einsum('sak,k->sa', P, V)` | `jnp.einsum('sak,k->sa', P, V)` | Tensor contraction | Parallel state-action reduction |
| **L3 POMDP Update** | Matmul + Hadamard + Norm | Pure functional scan | In-place normalisation | Fused probability filter |
| **Deterministic Parity** | $\|Y - Y^*\|_\infty < 10^{-6}$ | Match $< 10^{-6}$ | Match $< 10^{-6}$ | Transitive parity verified |

---

## 5. Pathological Failure Modes, Anti-Patterns & Diagnostic Guide

| Failure Mode | Mathematical Cause | Symptom | Production Diagnostic & Fix |
| :--- | :--- | :--- | :--- |
| **IPS Extreme Variance** | Propensity $\mu(a\|x) \to 0 \implies w \to \infty$ | Infinite variance in OPE value | Clip weights $w_i \le M$ or use Doubly Robust estimation. |
| **POMDP Information Neglect** | Greedy QMDP ignores belief entropy | State ambiguity remains unreduced | Incorporate active inference epistemic term $\mathbb{E}[D_{\text{KL}}]$. |
| **Misaligned 0-1 Cost** | Using argmax when false negative is $100\times$ costlier | Catastrophic downstream loss | Apply explicit Bayes risk cost matrix $C(a, y)$. |
| **Uncalibrated Probabilities** | Overconfident soft predictions | Spurious Chow rejection / acceptance | Calibrate with temperature scaling or Platt scaling. |

---

## 6. Cross-Domain Generalisation & Mathematical Isomorphisms

1. **Medical Diagnostics (L0 Bayes Risk):** Diagnosing rare critical diseases where a false negative has $1000\times$ penalty over a false positive, invoking Chow's reject option for human specialist review.
2. **Clinical Trial Assignment (L1 Bandits):** Allocating experimental treatments using contextual bandits while preserving historical safety via Doubly Robust off-policy evaluation.
3. **Robotics & Spacecraft Navigation (L3 POMDP):** Navigating under noisy LIDAR and IMU sensors by maintaining belief distributions over pose and velocity, taking probing maneuvers to localize.

---

## 7. Verification, Parity & Benchmarking Protocol

- **Golden Reference Checkpoint:** `weights/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.safetensors`
- **Tolerances:** Absolute difference $\|Y - Y_{\text{ref}}\|_\infty < 10^{-5}$ across all tracks.
- **Budget:** Sub-second execution (< 3.0s ceiling) in both CPU SIMD and GPU WGSL modes.

---

## 8. Navigation & Synchronized Track Artifacts

- [Interactive Concept Explorer Notebook](../../notebooks/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.ipynb)
- [Production PyTorch 2.x Module](../../pytorch2/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.py)
- [Functional JAX Module](../../jax/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.py)
- [Native Rust Candle Module](../../candle/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.rs)
- [Hardware Native Kernel](../../kernels/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.rs)
- [Golden SafeTensors Weights](../../weights/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.safetensors)
