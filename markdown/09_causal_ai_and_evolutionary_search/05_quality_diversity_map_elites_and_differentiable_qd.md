# Quality Diversity: MAP-Elites and Differentiable Quality Diversity (DQD)

## 1. Mathematical Foundations & Lineage

Conventional optimization algorithms maximize a single scalar objective function $\max_\theta f(\theta)$. In complex, deceptive, or non-convex fitness landscapes, single-objective hill climbers frequently suffer premature convergence into suboptimal local attractors because every step toward the global optimum may temporarily decrease fitness. Quality Diversity (QD) algorithms discard single-solution convergence in favor of *illumination*: discovering an archive of diverse, high-performing solutions that span the entire behavioral or phenotypic space.

The conceptual lineage comprises:
1. **Novelty Search (Lehman & Stanley 2011)**: Abandoned the objective function entirely, searching exclusively for solutions with novel behavioral characteristics measured by nearest-neighbor distance in behavior space.
2. **MAP-Elites (Multi-dimensional Archive of Phenotypic Elites — Mouret & Clune 2015)**: Unified quality and diversity by partitioning the continuous behavioral descriptor space $\mathcal{B} \subset \mathbb{R}^K$ into a discrete grid of niches and maintaining the highest-performing "elite" individual in every niche.
3. **Centroidal Voronoi Tessellation MAP-Elites (CVT-MAP-Elites — Vassiliades et al. 2018)**: Scaled MAP-Elites to higher-dimensional descriptor spaces using k-means / Voronoi partitioning rather than rectilinear grid discretization.
4. **Differentiable Quality Diversity (DQD / OMG-MEGA / CMA-MEGA — Fontaine & Nikolaidis 2021)**: Exploited first-order gradients of both the objective $\nabla_\theta f(\theta)$ and behavioral descriptors $\nabla_\theta b_k(\theta)$ via auto-differentiation, accelerating archive illumination by orders of magnitude compared to derivative-free mutation.

---

## 2. MAP-Elites Archive Illumination & Metrics

MAP-Elites maps parameter vectors $\theta \in \mathbb{R}^D$ to both a scalar fitness $f(\theta) \in \mathbb{R}$ and a behavioral descriptor vector $b(\theta) = [b_1(\theta), \dots, b_K(\theta)]^T \in \mathbb{R}^K$.

### Grid Discretization & Binning
The continuous behavioral space $[\mathbf{b}_{\min}, \mathbf{b}_{\max}]$ is discretized into a grid with resolution $\mathbf{B} = [B_1, \dots, B_K]$. For each descriptor dimension $k$:
$$\text{idx}_k = \text{clamp}\left( \left\lfloor \frac{b_k - b_{\min, k}}{b_{\max, k} - b_{\min, k} + \epsilon} \cdot B_k \right\rfloor, 0, B_k - 1 \right)$$
For a 2D behavioral space ($K=2$), the flattened 1D cell index is:
$$\text{cell\_idx} = \text{idx}_0 \cdot B_1 + \text{idx}_1$$

### Archive Occupancy & Elite Replacement Rule
Let $\mathcal{A}_{\text{fit}} \in (\mathbb{R} \cup \{-\infty\})^{\prod B_k}$ denote the archive fitness table. When evaluating a candidate individual $(\theta, f, b)$:
$$\mathcal{A}_{\text{fit}}[\text{cell\_idx}] \leftarrow \max\left( \mathcal{A}_{\text{fit}}[\text{cell\_idx}], \, f \right)$$
If $f > \mathcal{A}_{\text{fit}}[\text{cell\_idx}]$, the occupant's parameter vector $\theta$ replaces the incumbent in that niche.

### Key Quality-Diversity Metrics
1. **Archive Coverage**: Fraction of discovered niches:
   $$\text{Coverage} = \frac{1}{\prod_{k=1}^K B_k} \sum_{c} \mathbb{I}(\mathcal{A}_{\text{fit}}[c] > -\infty)$$
2. **QD-Score**: Sum of fitness values across all occupied niches:
   $$\text{QD-Score} = \sum_{c \in \text{occupied}} \mathcal{A}_{\text{fit}}[c]$$

---

## 3. Differentiable Quality Diversity (DQD) Formulations

In differentiable simulators or computational graphs (e.g. PyTorch, JAX), analytical gradients are available for both the objective and each behavioral descriptor:
- Objective gradient: $g_0 = \nabla_\theta f(\theta) \in \mathbb{R}^D$
- Behavioral descriptor Jacobian: $J_b = \nabla_\theta b(\theta) = [g_1, \dots, g_K]^T \in \mathbb{R}^{K \times D}$

### Gradient Arborescence / Multi-Objective Directional Stepping
Instead of isotropic Gaussian mutations, DQD explores along linear combinations of the objective and measure gradients:
$$\Delta \theta = c_0 \nabla_\theta f(\theta) + \sum_{k=1}^K c_k \nabla_\theta b_k(\theta) = [g_0, g_1, \dots, g_K] \cdot \mathbf{c}$$
where coefficients $\mathbf{c} = [c_0, c_1, \dots, c_K]^T$ determine the trade-off between maximizing performance ($c_0 > 0$) and traversing along specific behavioral axes ($c_k \neq 0$).

### Gram-Schmidt Orthogonalization of Behavioral Gradients
To discover novel behavioral phenotypes without degrading objective quality, the behavioral gradients are projected orthogonal to the objective gradient $g_0$:
$$g_k^{\perp} = g_k - \frac{\langle g_k, g_0 \rangle}{\|g_0\|_2^2 + \epsilon} g_0 \quad \forall k \in \{1, \dots, K\}$$
Stepping along $g_k^{\perp}$ shifts the system's behavioral phenotype while preserving first-order fitness stationarity.

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`map_elites_grid_indices`** | Vectorized clamp & floor | Functional `jnp.clip` & `jnp.floor` | Vectorized coordinate binning | Parallel grid binning |
| **`map_elites_archive_update`** | Cell-wise max reduction | `jax.numpy.at[c].set` | In-place elite max update | Fused atomic max cell update |
| **`dqd_gradient_arborescence_step`** | `c0 * g0 + sum(c_k * g_k)` | Vectorized contraction | Chained affine and accumulator ops | `DQD_ARBORESCENCE_WGSL` |
| **`dqd_orthogonalize_behavior_gradients`** | Batch projection `jac - proj * g0` | In-place Gram-Schmidt projector | Outer-product projection sub | Fused Gram-Schmidt projector |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Curse of Dimensionality in Grid Archives**:
   - *Symptom*: Archive size explodes exponentially ($\prod_{k=1}^K B_k = 10^K$), leaving $99.9\%$ of cells unilluminated when $K > 5$.
   - *Cause*: Rectilinear grid discretization scales exponentially with descriptor dimension $K$.
   - *Fix*: Replace rectilinear grids with Centroidal Voronoi Tessellations (CVT-MAP-Elites) or unbounded nearest-neighbor archives.
2. **Objective/Behavior Gradient Collinearity**:
   - *Symptom*: Orthogonalized behavioral gradients collapse to zero $\|g_k^{\perp}\| \approx 0$.
   - *Cause*: The chosen behavioral descriptor is functionally dependent on the objective, making diversity collinear with quality.
   - *Fix*: Select behavioral descriptors that are statistically and geometrically disentangled from the fitness metric.
3. **Deceptive Local Optima in Single-Objective Baselines**:
   - *Symptom*: Single-objective gradient ascent hits a plateau at fitness 40.0 while MAP-Elites discovers fitness 120.0 in an adjacent niche.
   - *Cause*: Stepping directly toward fitness requires passing through a low-fitness valley ("stepping-stone" problem).
   - *Fix*: MAP-Elites illuminates stepping stones in alternative behavioral niches that circumvent the fitness valley.

---

## 6. Real-World Engineering Applications

1. **Robotic Damage Recovery**: Pre-computing thousands of walking gaits with diverse leg duty cycles; when a physical motor breaks, the robot looks up the best surviving gait from the archive in $< 10$ milliseconds without retraining.
2. **Aerodynamic Airfoil Design**: Generating a spectrum of wing profiles maximizing lift across different drag and structural thickness trade-offs.
3. **Procedural Game Content & Drug Molecule Illumination**: Exploring chemical molecular structures that exhibit high binding affinity across diverse logP / molecular weight niches.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `grid_indices_ref` & `cell_indices_ref`: Continuous behavioral descriptor grid binning.
3. `archive_fit_updated_ref`: Elite replacement archive update table.
4. `dqd_step_ref`: Gradient arborescence directional step vector.
5. `dqd_ortho_jac_ref`: Gram-Schmidt orthogonalized behavioral descriptor gradients.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.py`](../../pytorch2/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.py)
- **JAX Reference**: [`jax/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.py`](../../jax/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.py)
- **Candle Reference**: [`candle/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.rs`](../../candle/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.rs)
- **Kernels**: [`kernels/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.rs`](../../kernels/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.rs)
- **Interactive Notebook**: [`notebooks/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.ipynb`](../../notebooks/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../INDEX_BY_GOAL.md)
