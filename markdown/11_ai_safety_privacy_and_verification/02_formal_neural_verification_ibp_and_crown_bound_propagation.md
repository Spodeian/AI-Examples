# Formal Neural Verification: IBP and CROWN Bound Propagation

## 1. Mathematical Foundations & Lineage

While empirical adversarial attacks and defenses (such as PGD and TRADES) measure sensitivity against specific gradient search heuristics, they cannot certify that **no** perturbation exists that could flip a decision boundary. Formal neural verification solves this by proving mathematical invariance over an entire input domain $\mathcal{C} = \{x : \|x - x_0\|_p \le \epsilon\}$:
1. **Interval Bound Propagation (IBP — Gowal et al. 2018; Mirman et al. 2018)**: Propagates bounding hypercubes layer-by-layer using interval arithmetic, running at standard forward-pass complexity $\mathcal{O}(L \cdot d^2)$ but suffering from severe over-approximation error that grows exponentially with network depth.
2. **CROWN (Zhang et al. 2018)**: Replaces loose axis-aligned intervals with tight linear bounding relaxations for non-linear activations (e.g. ReLU, sigmoid, tanh), backward-propagating linear inequality forms to preserve inter-neuron and inter-layer correlations.
3. **auto_LiRPA (Xu et al. 2020)**: Generalized linear relaxation bound propagation across arbitrary computational graphs and multi-modal models.
4. **$\alpha$-CROWN & $\beta$-CROWN (Wang et al. 2021)**: Parameterized lower relaxation slopes $\alpha$ and branch-and-bound domain splitting splits $\beta$ optimized via gradient descent, winning the VNN-COMP neural verification competition.

---

## 2. Interval Bound Propagation (IBP)

For a feedforward network $z^{(l+1)} = W^{(l)} a^{(l)} + b^{(l)}$ and $a^{(l)} = \sigma(z^{(l)})$:

### Linear Layer
Given input interval $l^{(l)} \le a^{(l)} \le u^{(l)}$:
Decomposing into positive and negative weight components $W_+ = \max(W, 0)$ and $W_- = \min(W, 0)$, or equivalently via the center-radius representation:
$$\mu = \frac{u^{(l)} + l^{(l)}}{2}, \quad r = \frac{u^{(l)} - l^{(l)}}{2}$$
$$\mu_{\text{out}} = \mu W^{(l)} + b^{(l)}, \quad r_{\text{out}} = r |W^{(l)}|$$
$$l^{(l+1)} = \mu_{\text{out}} - r_{\text{out}}, \quad u^{(l+1)} = \mu_{\text{out}} + r_{\text{out}}$$

### Non-Linear Activations (ReLU)
For monotonically non-decreasing activations $\sigma(z)$:
$$l_a = \sigma(l_z) = \max(l_z, 0), \quad u_a = \sigma(u_z) = \max(u_z, 0)$$

---

## 3. CROWN Linear Relaxation Bound Propagation

For an activation $y = \text{ReLU}(x)$ on pre-activation interval $x \in [l, u]$:
1. **Strictly Active ($l \ge 0$)**: Exact linear equality $y = x \implies s_L = 1, b_L = 0, s_U = 1, b_U = 0$.
2. **Strictly Inactive ($u \le 0$)**: Exact zero $y = 0 \implies s_L = 0, b_L = 0, s_U = 0, b_U = 0$.
3. **Crossing Neuron ($l < 0 < u$)**:
   - **Upper Secant Line**: Connects $(l, 0)$ and $(u, u)$:
     $$s_U = \frac{u}{u - l}, \quad b_U = -\frac{u \cdot l}{u - l}$$
   - **Lower Bounding Line**: Adaptive slope passing through origin:
     $$s_L = \alpha \in [0, 1] \quad (\text{heuristic: } \mathbb{I}(u > |l|)), \quad b_L = 0$$

### Backward Propagation
Given a scalar specification $c^\top y$, CROWN propagates coefficients backward:
$$\Lambda_L = \text{where}(v \ge 0, s_L, s_U) \odot v, \quad \Lambda_U = \text{where}(v \ge 0, s_U, s_L) \odot v$$
resulting in verified linear bounding forms over input $x$:
$$A_L x + \gamma_L \le c^\top y \le A_U x + \gamma_U$$
The verified lower bound on $c^\top y$ is then computed analytically over $x \in [x_l, x_u]$:
$$\min_{x \in [x_l, x_u]} A_L x + \gamma_L = A_L \mu_x - |A_L| r_x + \gamma_L$$

---

## 4. Branch-and-Bound ($\beta$-CROWN) & Completeness

When incomplete verification via convex relaxation cannot verify a property ($l_{\text{crown}} < 0$), complete verification is achieved via **Branch-and-Bound (BaB)**:
1. **Neuron Selection**: Select an unstable crossing neuron with highest bound looseness $(u_i - l_i) \cdot |v_i|$.
2. **Domain Split**: Split the domain into two subproblems: $\mathcal{D}_1: z_i \ge 0$ (active) and $\mathcal{D}_2: z_i \le 0$ (inactive).
3. **Pruning**: If $l(\mathcal{D}_k) > 0$, the subproblem is certified safe and pruned.
4. **Counter-Example**: If a concrete adversarial input is discovered during bound search, unsafe is proved.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`ibp_linear`** | Center-radius matmul | Functional center-radius | Fast CPU vector iterator | Center-radius vectorizer |
| **`ibp_relu`** | `torch.clamp(min=0)` | `jnp.clip(a_min=0)` | `val.max(0.0)` | WGSL clamp kernel |
| **`crown_relu_relaxation`** | Vectorized slope/bias | `jnp.where` masking | Vector branch evaluator | `CROWN_RELU_RELAXATION_WGSL` |
| **`crown_2layer_bounds`** | Backward linear form | Functional backward pass | Dual-bound linear builder | SIMD bound propagator |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Exponential Bound Explosion in Pure IBP**:
   - *Symptom*: Verified lower bound is $-\infty$ or thousands of orders of magnitude away from true network outputs on 5+ layer models.
   - *Cause*: Axis-aligned bounding boxes lose orientation; independent intervals on correlated neurons multiply error by $\| |W| \|_\infty^L$.
   - *Fix*: Use CROWN or auto_LiRPA linear relaxation which maintains linear dependencies all the way to the input.
2. **Suboptimal Lower Slope in Crossing Neurons**:
   - *Symptom*: Loose certified bounds when crossing neurons have large negative ranges ($|l| \gg u$).
   - *Cause*: Fixed slope $\alpha = 0.5$ or static heuristics.
   - *Fix*: Optimize $\alpha$ parameters dynamically via projected gradient ascent ($\alpha$-CROWN).
3. **Combinatorial State Space in Branch-and-Bound**:
   - *Symptom*: BaB verification times out when attempting complete verification on large models.
   - *Cause*: $2^U$ branching possibilities for $U$ unstable crossing neurons.
   - *Fix*: Combine IBP-regularized robust training (which drives $U \to 0$) with fast GPU-parallelized BaB.

---

## 7. Real-World Engineering Applications

1. **Airborne Collision Avoidance (ACAS Xu)**: Proving formal neural safety guarantees that aircraft advisory networks never output "Turn Left" when an intruder is approaching from the right.
2. **Autonomous Vehicle Lane Keeping**: Verifying that vision and control MLPs never command unsafe steering angles within bounded sensor noise tolerances.
3. **Medical Closed-Loop Artificial Pancreas**: Certifying that continuous glucose monitoring RL controllers never deliver an insulin bolus exceeding safety thresholds.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation.safetensors`:
1. `z1_l_ref`, `z1_u_ref`: IBP pre-activation hidden bounds.
2. `a1_l_ref`, `a1_u_ref`: IBP post-activation ReLU bounds.
3. `s_L_ref`, `b_L_ref`, `s_U_ref`, `b_U_ref`: CROWN ReLU linear relaxation parameters.
4. `crown_lower_ref`, `crown_upper_ref`: Final verified lower and upper bounds on target margin specification.
