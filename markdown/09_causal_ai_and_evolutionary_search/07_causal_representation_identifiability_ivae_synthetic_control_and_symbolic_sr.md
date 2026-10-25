# Causal Representation Identifiability: iVAE, Synthetic Control, and Symbolic Regression

## 1. Mathematical Foundations & Lineage

A foundational goal of modern artificial intelligence is discovering the true underlying physical and causal generative factors from high-dimensional observational data. However, unsupervised representation learning faces a fundamental mathematical barrier:

### Locatello et al.'s Impossibility Theorem (2019)
For any unsupervised generative model (such as a standard VAE or $\beta$-VAE) with independent factorial prior $p(z) = \prod_i p(z_i)$, there exist an uncountably infinite number of non-trivial, measure-preserving homeomorphisms $h: \mathcal{Z} \to \mathcal{Z}$ such that $p(h(z)) = p(z)$ and $p_g(x) = p_{g \circ h^{-1}}(x)$. Consequently, **unsupervised disentanglement without inductive biases or auxiliary conditioning is mathematically impossible** — models produce arbitrary rotations and mixtures of causal factors that appear visually appealing but completely scramble the true causal mechanisms.

The lineage of identifiable causal representation learning comprises:
1. **Abadie's Convex Synthetic Control (Abadie et al. 2003, 2010)**: Revolutionized causal econometric policy evaluation for $N=1$ treated unit by estimating counterfactual trajectories as a simplex-constrained convex combination of donor units.
2. **Time-Contrastive Learning & Nonlinear ICA (Hyvärinen & Morioka 2016)**: Proved that conditioning latent distributions on non-stationary auxiliary time segments renders nonlinear independent components identifiable.
3. **Identifiable VAE (iVAE — Khemakhem et al. 2020)**: Extended nonlinear ICA to deep variational autoencoders by conditioning an exponential-family latent prior $p(z \mid u)$ on auxiliary observable variables $u$ (e.g. environment, time, intervention indices), proving identifiability up to permutation and scaling.
4. **AI Feynman & PySR (Udrescu & Tegmark 2020, Cranmer 2023)**: Automated discovery of closed-form physical laws using Pareto multi-objective genetic programming.

---

## 2. Identifiable VAE (iVAE) Formulation

In an iVAE, the latent prior is conditioned on an auxiliary observable variable $u \in \mathcal{U}$:
$$p_\theta(x, z \mid u) = p_f(x \mid z) p_{\mathbf{T}, \lambda}(z \mid u)$$
where the prior belongs to a conditionally factorial exponential family:
$$p_{\mathbf{T}, \lambda}(z \mid u) = \prod_{j=1}^d \frac{1}{Z(u)} \exp\left( \sum_{k=1}^K T_{j, k}(z_j) \lambda_{j, k}(u) \right)$$
For conditionally factorial Gaussians with mean $\mu_u(u)$ and variance $\sigma_u^2(u)$:
$$p_\theta(z \mid u) = \mathcal{N}\left(z; \boldsymbol{\mu}_u(u), \text{diag}(\boldsymbol{\sigma}_u^2(u))\right)$$
The variational encoder outputs posterior parameters $q_\phi(z \mid x, u) = \mathcal{N}\left(z; \boldsymbol{\mu}_z, \text{diag}(\boldsymbol{\sigma}_z^2)\right)$.

### Closed-Form Conditional KL Divergence
The KL divergence between posterior and conditional prior decomposes across latent dimensions $j=1, \dots, d$:
$$D_{\text{KL}}(q_\phi(z \mid x, u) \parallel p_\theta(z \mid u)) = \frac{1}{2} \sum_{j=1}^d \left[ \frac{\sigma_{z, j}^2 + (\mu_{z, j} - \mu_{u, j})^2}{\sigma_{u, j}^2} - 1 - \left(\log \sigma_{z, j}^2 - \log \sigma_{u, j}^2\right) \right]$$
Under mild linear independence assumptions on the auxiliary parameter matrix $\mathbf{L}$, the true causal latents $z^*$ are mathematically guaranteed to be recovered up to coordinate permutation and component-wise monotonic transformation.

---

## 3. Abadie's Convex Synthetic Control Method

In comparative case studies and econometric policy evaluation, a policy intervention occurs in a single treated unit (e.g. California's Proposition 99 tobacco law, German reunification), precluding conventional randomized controlled trials.

### Convex Simplex Optimization
Let $Y_1 \in \mathbb{R}^T$ denote the treated unit trajectory and $Y_0 \in \mathbb{R}^{T \times J}$ denote an untreated donor pool of $J$ control units. The pre-intervention periods are $t \in [1, T_0]$ with feature vectors $X_1 \in \mathbb{R}^K, X_0 \in \mathbb{R}^{K \times J}$. Synthetic control finds donor weights $\mathbf{w}^*$:
$$\min_{\mathbf{w} \in \Delta^J} \|X_1 - X_0 \mathbf{w}\|_2^2 \quad \text{s.t.} \quad w_j \ge 0, \; \sum_{j=1}^J w_j = 1$$

### Why the Simplex Constraint $\mathbf{w} \in \Delta^J$ is Mandatory
1. **Prevents Extrapolation**: Constraining $\mathbf{w}$ to the probability simplex guarantees that the synthetic counterfactual $\hat{X}_1 = X_0 \mathbf{w}$ lies strictly within the convex hull of donor units, eliminating out-of-hull extrapolation bias.
2. **Interpretability & Sparsity**: The non-negativity constraint produces sparse donor assignments, explicitly showing which specific comparison units construct the counterfactual.

### Counterfactual Trajectory & Causal Treatment Effect
For post-intervention periods $t > T_0$:
$$\hat{Y}_{1, t}^{\text{synthetic}} = \sum_{j=1}^J w_j^* Y_{0, t, j}, \quad \hat{\tau}_t = Y_{1, t} - \hat{Y}_{1, t}^{\text{synthetic}}$$

---

## 4. Multi-Objective Pareto Symbolic Regression

Symbolic regression discovers analytic mathematical formulas $f(x)$ balancing empirical goodness-of-fit against structural formula complexity $C(f)$ (node count / AST depth):
$$\mathcal{L}_{\text{Pareto}}(f) = \text{MSE}(y, f(x)) + \lambda \cdot C(f)$$
By tracing the Pareto frontier of non-dominated expressions, symbolic regression avoids overparameterized black-box approximations and surfaces exact scientific laws.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`ivae_kl_divergence`** | Vectorized Gaussian KL sum | Elementwise closed-form JAX | Exp, ratio, and affine reduction | `IVAE_KL_WGSL` |
| **`project_simplex`** | Duchi sort & threshold clamp | `jnp.sort` & `jnp.cumsum` | Descending sort & cumulative threshold | Parallel simplex projector |
| **`synthetic_control_effect`**| `y0 @ w`, difference $Y_1 - \hat{Y}_1$ | Matrix-vector product | Matrix multiplication & sub | Fused dot-product subtraction |
| **`symbolic_pareto_loss`** | `mean((y - y_hat)^2) + lambda * C`| Mean squared error + penalty | MSE reduction + affine scalar add | Vectorized MSE accumulator |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Unsupervised Disentanglement Failure in Standard $\beta$-VAEs**:
   - *Symptom*: High latent factor traversal correlation ($r > 0.85$); latents change arbitrarily between different random seeds.
   - *Cause*: Locatello's theorem: unsupervised VAEs have rotational symmetry in their latent prior $p(z) = \mathcal{N}(0, I)$, making causal factor orientation unidentifiable.
   - *Fix*: Condition prior on auxiliary supervision $u$ (iVAE) or incorporate interventional targets.
2. **Out-of-Hull Extrapolation in Unconstrained Synthetic Control**:
   - *Symptom*: Synthetic counterfactual predicts impossible negative values or extreme volatile spikes.
   - *Cause*: Using unconstrained Ordinary Least Squares (OLS) regression instead of simplex-constrained weights ($w_j < 0$ or $\sum w_j \neq 1$), allowing extreme extrapolation outside the convex hull of observed donor units.
   - *Fix*: Strictly enforce simplex projection $\mathbf{w} \in \Delta^J$ via Duchi's algorithm.
3. **Symbolic Regression Expression Bloat**:
   - *Symptom*: Expression AST grows to hundreds of nested polynomial and trigonometric terms with negligible training error but extreme test error.
   - *Cause*: Insufficient Pareto complexity penalty $\lambda$, allowing genetic mutations to insert redundant algebraic operations.
   - *Fix*: Enforce parsimony pressure and prune dominated equations from the Pareto archive.

---

## 7. Real-World Engineering Applications

1. **Healthcare Policy & Epidemiology**: Estimating unit-level counterfactual infection trajectories for regions implementing public health mandates using unaffected neighboring jurisdictions as synthetic donors.
2. **Robotics Causal State Disentanglement**: iVAE disentangles true underlying physical properties (mass, friction, joint angles) from raw sensor streams using action / time step conditioning.
3. **Automated Scientific Discovery**: Re-deriving classical physics equations (Kepler's laws, Lorentz factor, Navier-Stokes relations) from raw laboratory observations.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `ivae_kl_ref`: Conditional factorial Gaussian KL divergence vector.
3. `sc_w_simplex_ref`: Euclidean projection onto the probability simplex.
4. `sc_y1_synth_ref` & `sc_tau_ref`: Synthetic control counterfactual trajectory and causal treatment effect.
5. `sr_pareto_loss_ref`: Multi-objective Pareto complexity loss.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 9. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.py`](../../pytorch2/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.py)
- **JAX Reference**: [`jax/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.py`](../../jax/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.py)
- **Candle Reference**: [`candle/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.rs`](../../candle/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.rs)
- **Kernels**: [`kernels/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.rs`](../../kernels/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.rs)
- **Interactive Notebook**: [`notebooks/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.ipynb`](../../notebooks/09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../INDEX_BY_GOAL.md)
