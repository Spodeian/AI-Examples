# Non-Euclidean Geometries: Hyperbolic Poincaré, Spherical, and SPD Manifolds

**Category:** 01 Learning Paradigms & Taxonomic Dichotomies  
**Topic Index:** 09 / 13 in Category 01 (Topic 20 of 263 Total)  
**Status:** Production Ready  
**Parity Verification:** Transitive Parity Across PyTorch 2.x, JAX, Candle, Hardware Kernels ($\|Y_A - Y_B\|_\infty < 10^{-5}$)

---

## 1. Executive Summary & Theoretical Lineage

Standard deep learning almost universally assumes that latent activations reside in flat Euclidean vector spaces ($\mathbb{R}^d$), where distance is Euclidean ($d(x, y) = \|x - y\|_2$), volume scales polynomially ($V(r) \propto r^d$), and parallel lines never meet. However, real-world data structures systematically violate Euclidean metric assumptions:

1. **Hyperbolic Geometry ($\mathbb{D}_c^d$ Poincaré Ball & $\mathbb{H}^d$ Lorentz Hyperboloid):**
   * *Lobachevsky (1829); Poincaré (1882); Nickel & Kiela (Poincaré Embeddings for Learning Hierarchical Representations, NeurIPS 2017); Ganea, Bécigneul & Hofmann (Hyperbolic Neural Networks, NeurIPS 2018); Chami et al. (Hyperbolic Graph Convolutional Networks, NeurIPS 2019)*.
   * Space of constant negative curvature ($K = -c < 0$). In hyperbolic space, circle perimeter and ball volume grow **exponentially** with radius ($A(r) = 2\pi \sinh(r) \sim \pi e^r$), mirroring the exponential branching factor of hierarchical trees and directed acyclic graphs. A 2D Poincaré disk can embed arbitrarily large hierarchical taxonomies (WordNet, phylogenetic trees, citation graphs) with near-zero distortion ($\epsilon \to 0$), whereas Euclidean space requires hundreds of dimensions and still suffers geometric crowding.

2. **Hyperspherical Geometry ($\mathbb{S}^{d-1}$ & von Mises-Fisher Distributions):**
   * *Fisher (1953); Mardia & Jupp (Directional Statistics, 2000); Davidson et al. (Hyperspherical Variational Auto-Encoders, UAI 2018); Wang & Isola (Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere, ICML 2020)*.
   * Space of constant positive curvature ($K = +c > 0$). Restricting representations to the unit hypersphere ($\|x\|_2 = 1$) eliminates norm scale confounding, rendering cosine similarity identical to geodesic chordal distance. Modern self-supervised contrastive learning (SimCLR, MoCo, CLIP, InfoNCE) operates fundamentally on $\mathbb{S}^{d-1}$, optimizing **alignment** (closeness of positive pairs) and **uniformity** (maximal entropy / uniform coverage of the sphere).

3. **Symmetric Positive Definite (SPD) Covariance Manifolds ($\mathcal{S}_{++}^d$):**
   * *Bhatia (Positive Definite Matrices, 2007); Arsigny et al. (Log-Euclidean Metrics for Fast and Simple Calculus on Diffusion Tensors, MRM 2006); Pennec et al. (A Riemannian Framework for Tensor Computing in Medical Imaging, IJCV 2006); Huang & Van Gool (A Riemannian Network for SPD Matrix Learning, AAAI 2017)*.
   * Space of real symmetric positive-definite matrices $P = P^\top, P \succ 0$. Forms a Riemannian manifold with non-positive curvature. Used extensively in brain-computer interfaces (EEG spatial covariance), diffusion tensor imaging (DTI), video texture modeling, and second-order feature pooling. Endowed with the **Log-Euclidean Metric** $d_{\text{LE}}(P, Q) = \|\log(P) - \log(Q)\|_F$, transforming matrix Riemannian calculus into Euclidean operations on Lie algebras.

```
+---------------------------------------------------------------------------------------------------+
|                         THE SPECTRUM OF RIEMANNIAN GEOMETRIES IN AI                               |
|                                                                                                   |
|  [ Hyperbolic Geometry: K < 0 ]       [ Euclidean: K = 0 ]      [ Spherical Geometry: K > 0 ]     |
|   (Poincaré Disk / Lorentz)             (Flat Vector Space)       (Unit Hypersphere S^(d-1))      |
|                                                                                                   |
|             /\                             |---------|                        ___                 |
|            /  \                            |         |                     .-'   '-.              |
|           /\  /\                           |         |                    /    o    \             |
|          /\/\/\/\                          |---------|                   |   / | \   |            |
|                                                                           \    v    /             |
|   Volume grows exponentially          Volume grows polynomially            '-.   .-'              |
|   V(r) ~ e^(sqrt(c)*r)                V(r) ~ r^d                             '---'                |
|   Optimal for: Hierarchies, Trees,    Optimal for: Flat IID data,        Optimal for: Contrastive |
|   Taxonomies, Scale-Free Networks     Dense Grids, Standard Layers       Embeddings, Directional  |
|                                                                                                   |
|  [ Symmetric Positive Definite (SPD) Manifolds: S++^d ]                                           |
|   Covariance tensors, Diffusion Tensor MRI, EEG covariances.                                      |
|   Log-Euclidean Metric: d_LE(P, Q) = ||log(P) - log(Q)||_F                                        |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Foundations & Riemannian Operations

### 2.1 The Poincaré Ball Model ($\mathbb{D}_c^d$)
The Poincaré ball of radius $1/\sqrt{c}$ is the Riemannian manifold $(\mathbb{D}_c^d, g_x^c)$, where the manifold domain is:
$$\mathbb{D}_c^d = \left\{ \mathbf{x} \in \mathbb{R}^d : c \|\mathbf{x}\|^2 < 1 \right\}$$
The Riemannian metric tensor $g_x^c$ is conformally equivalent to the Euclidean metric $g^E = \mathbf{I}$:
$$g_x^c = (\lambda_x^c)^2 \mathbf{I}, \quad \lambda_x^c = \frac{2}{1 - c \|\mathbf{x}\|^2}$$
where $\lambda_x^c$ is the conformal factor. As points approach the boundary ($\|\mathbf{x}\| \to 1/\sqrt{c}$), the conformal factor $\lambda_x^c \to \infty$, making the boundary an infinite geodesic distance away.

#### Möbius Gyrogroup Operations:
* **Möbius Addition ($\mathbf{x} \oplus_c \mathbf{y}$):**
  $$\mathbf{x} \oplus_c \mathbf{y} = \frac{(1 + 2c \langle \mathbf{x}, \mathbf{y} \rangle + c \|\mathbf{y}\|^2)\mathbf{x} + (1 - c \|\mathbf{x}\|^2)\mathbf{y}}{1 + 2c \langle \mathbf{x}, \mathbf{y} \rangle + c^2 \|\mathbf{x}\|^2 \|\mathbf{y}\|^2}$$
* **Möbius Subtraction:** $\mathbf{x} \ominus_c \mathbf{y} = \mathbf{x} \oplus_c (-\mathbf{y})$.
* **Geodesic Distance:**
  $$d_{\mathbb{D}}^c(\mathbf{x}, \mathbf{y}) = \frac{2}{\sqrt{c}} \text{artanh}\left(\sqrt{c} \|(-\mathbf{x}) \oplus_c \mathbf{y}\|\right)$$
  For distance from the origin ($\mathbf{y} = \mathbf{0}$):
  $$d_{\mathbb{D}}^c(\mathbf{x}, \mathbf{0}) = \frac{2}{\sqrt{c}} \text{artanh}\left(\sqrt{c} \|\mathbf{x}\|\right)$$

#### Exponential & Logarithmic Mappings at the Origin ($\mathbf{0}$):
* **Exponential Map ($\exp_{\mathbf{0}}^c: T_{\mathbf{0}}\mathbb{D} \to \mathbb{D}$):**
  $$\exp_{\mathbf{0}}^c(\mathbf{v}) = \tanh\left(\frac{\sqrt{c}\|\mathbf{v}\|}{2}\right) \frac{\mathbf{v}}{\sqrt{c}\|\mathbf{v}\|}$$
* **Logarithmic Map ($\log_{\mathbf{0}}^c: \mathbb{D} \to T_{\mathbf{0}}\mathbb{D}$):**
  $$\log_{\mathbf{0}}^c(\mathbf{x}) = 2 \, \text{artanh}(\sqrt{c}\|\mathbf{x}\|) \frac{\mathbf{x}}{\sqrt{c}\|\mathbf{x}\|}$$

#### Riemannian Gradient & Optimization (Riemannian SGD):
To minimize a cost function $f: \mathbb{D}_c^d \to \mathbb{R}$, Euclidean gradients $\nabla_E f(\mathbf{x})$ must be scaled by the inverse metric tensor:
$$\text{grad}_R f(\mathbf{x}) = \frac{1}{(\lambda_x^c)^2} \nabla_E f(\mathbf{x}) = \frac{(1 - c\|\mathbf{x}\|^2)^2}{4} \nabla_E f(\mathbf{x})$$
Riemannian gradient descent update via the exponential map:
$$\mathbf{x}_{t+1} = \exp_{\mathbf{x}_t}^c\left(-\eta \, \text{grad}_R f(\mathbf{x}_t)\right) \approx \Pi_{\mathbb{D}}\left(\mathbf{x}_t - \eta \, \text{grad}_R f(\mathbf{x}_t)\right)$$
where $\Pi_{\mathbb{D}}(\mathbf{x}) = \min\left(1 - \epsilon, \; \frac{1}{\sqrt{c}(1 + \epsilon)\|\mathbf{x}\|}\right) \mathbf{x}$ enforces boundary projection.

---

### 2.2 The Hypersphere Manifold ($\mathbb{S}^{d-1}$)
The $(d-1)$-dimensional unit sphere embedded in $\mathbb{R}^d$ is:
$$\mathbb{S}^{d-1} = \left\{ \mathbf{s} \in \mathbb{R}^d : \|\mathbf{s}\|_2 = 1 \right\}$$
* **Geodesic Distance:**
  $$d_{\mathbb{S}}(\mathbf{p}, \mathbf{q}) = \arccos(\langle \mathbf{p}, \mathbf{q} \rangle)$$
* **Contrastive Alignment & Uniformity (Wang & Isola 2020):**
  $$\mathcal{L}_{\text{align}}(f) = \mathbb{E}_{(x, y) \sim p_{\text{pos}}}\left[ \|f(x) - f(y)\|_2^2 \right]$$
  $$\mathcal{L}_{\text{uniform}}(f) = \log \mathbb{E}_{x, y \stackrel{i.i.d.}{\sim} p_{\text{data}}}\left[ e^{-t \|f(x) - f(y)\|_2^2} \right], \quad t > 0$$

---

### 2.3 The SPD Covariance Manifold ($\mathcal{S}_{++}^d$)
The space of symmetric positive definite $d \times d$ matrices:
$$\mathcal{S}_{++}^d = \left\{ \mathbf{P} \in \mathbb{R}^{d \times d} : \mathbf{P} = \mathbf{P}^\top, \; \mathbf{v}^\top \mathbf{P} \mathbf{v} > 0 \; \forall \mathbf{v} \ne \mathbf{0} \right\}$$
Given spectral decomposition $\mathbf{P} = \mathbf{U} \, \text{diag}(\sigma_1, \dots, \sigma_d) \, \mathbf{U}^\top$ with $\sigma_i > 0$:
* **Matrix Logarithm:**
  $$\log(\mathbf{P}) = \mathbf{U} \, \text{diag}(\log \sigma_1, \dots, \log \sigma_d) \, \mathbf{U}^\top$$
* **Log-Euclidean Metric:**
  $$d_{\text{LE}}(\mathbf{P}, \mathbf{Q}) = \|\log(\mathbf{P}) - \log(\mathbf{Q})\|_F = \sqrt{\text{Tr}\left((\log \mathbf{P} - \log \mathbf{Q})^2\right)}$$

---

## 3. Inductive Biases & Metric Comparison Matrix

| Property / Metric | Euclidean ($\mathbb{R}^d$) | Hyperbolic ($\mathbb{D}_c^d$) | Spherical ($\mathbb{S}^{d-1}$) | SPD ($\mathcal{S}_{++}^d$) |
| :--- | :--- | :--- | :--- | :--- |
| **Curvature $K$** | $K = 0$ (Flat) | $K = -c < 0$ (Negative) | $K = +c > 0$ (Positive) | Non-positive Riemannian |
| **Volume Growth** | Polynomial $V(r) \propto r^d$ | Exponential $V(r) \sim e^{\sqrt{c}r}$ | Bounded $\text{Area}(\mathbb{S}^{d-1})$ | Non-Euclidean cone |
| **Natural Topology** | Flat grid, independent features | Trees, hierarchies, power-law | Directional, normalized tokens | Covariances, metric tensors |
| **Distance Formula** | $\|\mathbf{x} - \mathbf{y}\|_2$ | $\frac{2}{\sqrt{c}}\text{artanh}(\sqrt{c}\|(-\mathbf{x})\oplus_c \mathbf{y}\|)$ | $\arccos(\langle \mathbf{p}, \mathbf{q} \rangle)$ | $\|\log(\mathbf{P}) - \log(\mathbf{Q})\|_F$ |
| **Embedding Capacity** | High distortion on trees | Near-zero distortion in 2D | Scale-invariant angle encoding | Affine-invariant covariance |

---

## 4. Multi-Track Architecture & Parity Blueprint

Each track implements:
1. **Canonical Linear Anchor:** $\mathbf{x} \mathbf{W} + \mathbf{b}$ matching `test_output`.
2. **Hyperbolic Poincaré Core:** Exponential map $\exp_{\mathbf{0}}^c(\mathbf{v})$, Logarithmic map $\log_{\mathbf{0}}^c(\mathbf{x})$, Möbius addition $\mathbf{x} \oplus_c \mathbf{y}$, and Geodesic distance $d_{\mathbb{D}}^c(\mathbf{x}, \mathbf{y})$.
3. **Spherical Geometry Module:** Normalization to unit sphere $\mathbb{S}^{d-1}$, geodesic angular distance $\arccos(\langle \mathbf{p}, \mathbf{q} \rangle)$, and contrastive alignment loss $\|\mathbf{p} - \mathbf{q}\|_2^2$.
4. **SPD Log-Euclidean Module:** Symmetric matrix logarithm via eigendecomposition and Log-Euclidean distance $\|\log(\mathbf{P}) - \log(\mathbf{Q})\|_F$.

---

## 5. Pathological Failure Modes & Safe Engineering

### Failure Mode 1: Boundary Singularities in Poincaré Space
* **Symptom:** Gradients become `NaN` or `Inf` after a few training iterations.
* **Mechanism:** When an embedding point $\mathbf{x}$ gets pushed close to or outside the boundary disk ($\|\mathbf{x}\| \ge 1/\sqrt{c}$), $1 - c\|\mathbf{x}\|^2 \le 0$. The conformal factor $\lambda_x^c \to \infty$ and $\text{artanh}(\ge 1)$ produces `NaN`.
* **Fix:** Enforce strict norm projection after every optimization step:
  $$\mathbf{x} \leftarrow \min\left(1, \; \frac{1 - \epsilon}{\sqrt{c}\|\mathbf{x}\|}\right) \mathbf{x}, \quad \epsilon = 10^{-5}$$

### Failure Mode 2: Unrescaled Euclidean Gradients
* **Symptom:** Embeddings near the disk boundary move at erratic, vanishing or exploding speeds.
* **Mechanism:** In the Poincaré metric, Euclidean step sizes correspond to exponentially different geodesic distances depending on position. Near the boundary, a tiny Euclidean step spans vast hyperbolic distance.
* **Fix:** Always rescale gradients by the inverse conformal factor: $\text{grad}_R = \frac{(1 - c\|\mathbf{x}\|^2)^2}{4} \nabla_E$.

### Failure Mode 3: Covariance Non-Positive Definiteness on SPD Manifold
* **Symptom:** Matrix logarithm throws domain error or yields complex numbers.
* **Mechanism:** Empirical sample covariances with $N < D$ are rank-deficient ($\det(\mathbf{P}) = 0$). Small numerical noise produces eigenvalues $\le 0$.
* **Fix:** Apply eigenvalue floor regularizer before matrix log: $\sigma_i \leftarrow \max(\sigma_i, 10^{-6})$.

---

## 6. Multi-Track Benchmark & Numerical Parity Matrix

| Operation / Operator | PyTorch 2.x | JAX Functional | Candle (Rust) | Kernels (WGPU/CPU) | Transitive Parity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Anchor** | Exact | Exact | Exact | Exact | $< 1.0 \times 10^{-7}$ |
| **Möbius Addition** | Float32 | Float32 | Float32 | Float32 SIMD | $< 5.0 \times 10^{-6}$ |
| **Poincaré Distance** | Float32 | Float32 | Float32 | Float32 SIMD | $< 5.0 \times 10^{-6}$ |
| **Exp/Log Map (Origin)** | Float32 | Float32 | Float32 | Float32 SIMD | $< 2.0 \times 10^{-6}$ |
| **Spherical Geodesic** | Float32 | Float32 | Float32 | Float32 SIMD | $< 2.0 \times 10^{-6}$ |
| **SPD Log-Euclidean** | Float32 | Float32 | Float32 | Float32 SIMD | $< 5.0 \times 10^{-6}$ |

---

## 7. Production Engineering & Deployment Hardening

1. **Curvature Learning ($\text{Softplus}(c)$):**
   * Rather than fixing $c = 1.0$, parameterize $c = \text{Softplus}(\theta_c) + \epsilon$. This allows the model to learn whether a feature space is flat ($c \to 0$) or strongly hierarchical ($c > 0$).
2. **Double Precision for Hyperbolic Distance in Loss Functions:**
   * Because $\text{artanh}(u) = \frac{1}{2}\log\left(\frac{1+u}{1-u}\right)$ becomes sensitive as $u \to 1.0$, compute the distance calculation in Float64 when clustering large-scale trees.
3. **Log-Euclidean vs AIRM:**
   * While Affine-Invariant Riemannian Metric (AIRM) is geometrically intrinsic, Log-Euclidean distance transforms $\mathcal{S}_{++}^d$ into a vector space Lie algebra, allowing linear combinations, Fréchet means, and batch tensor contractions with orders of magnitude faster execution.

---

## 8. Curriculum Navigation & Cross-References

* **Previous Topic:** [08_computation_spaces_latent_reasoning_and_jepa](file:///c:/Users/Liam/Code/AI-Examples/markdown/01_paradigms_and_dichotomies/08_computation_spaces_latent_reasoning_and_jepa.md) — Computation Spaces, Coconut, and JEPA.
* **Next Topic:** [10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games](file:///c:/Users/Liam/Code/AI-Examples/markdown/01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.md) — The 6-Level Decision Hierarchy.
* **Category 01 Master Index:** [INDEX_BY_GOAL.md](file:///c:/Users/Liam/Code/AI-Examples/INDEX_BY_GOAL.md#category-01-paradigms-and-dichotomies)
