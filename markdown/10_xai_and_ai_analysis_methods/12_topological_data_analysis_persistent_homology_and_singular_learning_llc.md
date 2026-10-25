# Topological Data Analysis: Persistent Homology and Singular Learning Theory LLC

## 1. Mathematical Foundations & Lineage

Analyzing complex geometry, multi-scale manifold structures, and singular parameter spaces requires modern algebraic topology and singular algebraic geometry:
1. **Topological Data Analysis & Persistent Homology (TDA — Edelsbrunner & Harer 2008; Carlsson 2009)**: Quantifies shape, clustering, and multi-scale voids in point clouds. By growing $\epsilon$-balls around points in a Vietoris-Rips filtration, features across topological homology dimensions $H_0$ (connected components), $H_1$ (1D loops/holes), and $H_2$ (voids) are recorded as birth-death pairs $(b_i, d_i)$.
2. **Persistence Landscapes (Bubenik 2015)**: Maps irregular persistence diagrams into a separable Banach vector space $L^p(\mathbb{R})$, enabling standard vector operations, statistical averages, and kernel methods on topological representations.
3. **Singular Learning Theory & Local Learning Coefficient (SLT & LLC — Watanabe 2001, 2009; Lau et al. 2023)**: Deep neural networks are strictly *singular* models (non-identifiable; true parameter fiber is an algebraic variety where Fisher information is degenerate). Classical regular statistical measures (AIC, BIC, parameter count $d/2$) fail. Watanabe proved that true generalization and free energy are governed by the Real Log Canonical Threshold (RLCT) $\lambda(w^*) \in (0, d/2]$, estimated locally via Stochastic Gradient Langevin Dynamics (SGLD).

---

## 2. Vietoris-Rips Persistent Homology

Given a finite metric space $(X, d_X)$ with $N$ points, the Vietoris-Rips complex $\text{VR}_\epsilon(X)$ contains simplex $\sigma = [x_0, \dots, x_k]$ if pairwise distances satisfy $d_X(x_i, x_j) \le \epsilon$ for all $0 \le i < j \le k$.

As scale $\epsilon$ increases from $0$ to $\infty$:
- **Betti-0 ($\beta_0$)**: Counts the number of connected components. At $\epsilon = 0$, $\beta_0 = N$; as edges form, components merge until $\beta_0 = 1$.
- **Birth-Death Lifespan**: A topological feature born at $\epsilon = b$ and dying (merging into an older component) at $\epsilon = d$ has lifespan $\text{pers} = d - b$. High-lifespan features reflect genuine geometric structure, while short-lived features represent sampling noise.

---

## 3. Persistence Landscapes

To compute vector statistics on persistence diagrams $\mathcal{D} = \{(b_i, d_i)\}_{i=1}^M$, Bubenik defined triangular functions for each feature:
$$f_{(b_i, d_i)}(t) = \begin{cases} t - b_i & \text{if } t \in [b_i, \frac{b_i + d_i}{2}] \\ d_i - t & \text{if } t \in (\frac{b_i + d_i}{2}, d_i] \\ 0 & \text{otherwise} \end{cases} = \max(0, \min(t - b_i, d_i - t))$$
The $k$-th level persistence landscape is:
$$\lambda_k(t) = \text{k-th largest value of } \{ f_{(b_i, d_i)}(t) \}_{i=1}^M$$
For $k=1$, $\lambda_1(t) = \max_{i=1}^M \max(0, \min(t - b_i, d_i - t))$. Persistence landscapes are 1-Lipschitz stable under the bottleneck and Wasserstein distance metrics.

---

## 4. Singular Learning Theory & Local Learning Coefficient (LLC)

In singular statistical models, the asymptotic expansion of the marginal likelihood / free energy is:
$$F_n = n L_n(w^*) + \lambda \ln n - (m - 1) \ln \ln n + \mathcal{O}_p(1)$$
where $\lambda$ is the RLCT / Local Learning Coefficient and $m$ is the multiplicity.
In a regular non-singular model, $\lambda = d/2$ (half the parameter count). In deep neural networks, $\lambda \ll d/2$, reflecting immense parameter redundancy and flatter, more generalizable singularities.

### Localized SGLD Estimator (Lau et al. 2023)
Given local empirical minimum $w^*$ on $n$ samples, draw posterior samples $w \sim \pi(w \mid \mathcal{D})$ via localized SGLD with inverse temperature $\beta$:
$$w_{t+1} = w_t - \frac{\epsilon}{2} \left( n \nabla L_n(w_t) + \gamma n (w_t - w^*) \right) + \sqrt{\epsilon} \xi_t, \quad \xi_t \sim \mathcal{N}(0, I)$$
The empirical LLC $\hat{\lambda}$ is estimated from the energy difference:
$$\hat{\lambda}(w^*) = \frac{n \beta}{2} \mathbb{E}_{w} [ L_n(w) - L_n(w^*) ]$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`persistence_diagram_lifespans`** | Vector subtraction $d - b$ | Pure array subtraction | Vector mapping iterator | Vector subtraction kernel |
| **`persistence_landscape_level1`** | Triangle clamp + max reduction | `jnp.maximum` + `jnp.max` | 1D grid loop + triangular max | `PERSISTENCE_LANDSCAPE_WGSL` |
| **`singular_learning_llc_energy`** | Mean loss difference + scale | `jnp.mean` + scalar scale | Vector mean + scalar product | Parallel loss accumulator |
| **`betti_0_filtration_counts`** | Threshold adjacency + BFS loop | Sequential graph traversal | Adjacency test + BFS queue | Graph connectivity reducer |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Dimensional Curse in Higher-Order VR Complexes**:
   - *Symptom*: Memory explodes when attempting to compute Betti-2 or Betti-3 on point clouds ($N > 500$).
   - *Cause*: The number of $k$-simplices scales as $\binom{N}{k+1} = \mathcal{O}(N^{k+1})$.
   - *Fix*: Subsample landmark points or construct sparse witness / alpha complexes.
2. **SGLD Temperature & Localization Drift**:
   - *Symptom*: LLC estimate $\hat{\lambda}$ becomes negative or diverges to infinity.
   - *Cause*: Weak localization $\gamma$ allows SGLD to escape the basin of $w^*$, or step size $\epsilon$ violates stability limits.
   - *Fix*: Calibrate $\gamma = 1.0$ and verify that SGLD samples stay within $\|w - w^*\|_2 < 0.1$.
3. **Metric Scale Distortion in Persistence**:
   - *Symptom*: Latent space TDA finds no topological loops despite circular ground truth.
   - *Cause*: Non-isotropic activation scales stretch circles into thin ellipsoids, causing death $\approx$ birth.
   - *Fix*: Standardize latent dimensions to unit variance before computing pairwise Euclidean distances.

---

## 7. Real-World Engineering Applications

1. **Phase Transition & Generalization Auditing**: Monitoring $\hat{\lambda}$ throughout training to detect algorithmic transitions (e.g., grokking in modular arithmetic).
2. **Manifold Topology Quality Control**: Using persistence landscapes to verify that generative models (diffusion, VAE) preserve the topology of data manifolds without introducing artificial holes.
3. **Complexity Regularization**: Using the LLC as an objective metric to select between competing neural architectures of different parameter counts.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `lifespans_ref`: Feature persistence lifespan vector $d - b$.
3. `landscape_ref`: Level-1 persistence landscape vector across grid points.
4. `llc_ref`: Singular Learning Theory Local Learning Coefficient scalar $\hat{\lambda}$.
5. `betti0_ref`: Betti-0 connected component counts across filtration thresholds.
