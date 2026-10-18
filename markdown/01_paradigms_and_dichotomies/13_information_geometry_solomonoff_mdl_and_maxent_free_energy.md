# Information Geometry, Solomonoff MDL, and MaxEnt Free Energy (Topic 13)

## 1. Executive Summary
Information Geometry (*Amari 1985, 2016*) provides the differential geometric foundation for modern statistical learning and machine perception. Rather than treating probability distributions as elements of an arbitrary Euclidean vector space, Information Geometry formalizes families of probability densities as Riemannian manifolds equipped with the unique invariant **Fisher-Rao metric tensor** $g_{ij}(\theta)$. In exponential families, this manifold possesses a **dually flat structure** governed by Legendre-Fenchel conjugacy between primal natural parameters $\theta$ and dual expectation parameters $\eta = \nabla_\theta \psi(\theta)$.

Complementing this Riemannian geometric framework, **Algorithmic Information Theory** (*Solomonoff 1964; Kolmogorov 1965*) and the **Minimum Description Length (`MDL`) principle** (*Rissanen 1978; Grünwald 2007*) eliminate subjective priors by formalizing inductive inference as data compression. Minimax optimal universal codes, achieved through **Shtarkov's Normalized Maximum Likelihood (`NML`)**, establish finite-sample generalization bounds through parametric complexity. Finally, **Friston's Variational Free Energy Principle** (*Friston 2010*) unifies perceptual state estimation and motor action as the variational minimization of a single information-theoretic upper bound on surprise $\mathcal{F}(q) \ge -\log p(x)$.

This document details the mathematical architecture, coordinate mechanics, and cross-framework implementation of Information Geometry, MDL, and Free Energy across PyTorch 2.x, JAX, Rust Candle, and WebGPU/CPU hardware kernels.

---

## 2. Mathematical Foundations

### 2.1 Exponential Families and Amari's Dually Flat Manifold
An exponential family on $\mathcal{X} \subseteq \mathbb{R}^D$ is defined by:
$$p(x; \theta) = h(x) \exp\left( \langle \theta, T(x) \rangle - \psi(\theta) \right)$$
where $\theta \in \Theta \subseteq \mathbb{R}^K$ denotes the natural (canonical) parameter vector, $T(x) \in \mathbb{R}^K$ is the sufficient statistic vector, $h(x)$ is the base carrier measure, and $\psi(\theta)$ is the log-partition function (cumulant generating function):
$$\psi(\theta) = \log \int_{\mathcal{X}} h(x) \exp\left( \langle \theta, T(x) \rangle \right) dx$$

The gradient of $\psi(\theta)$ maps natural coordinates $\theta$ directly to **dual expectation coordinates** $\eta$:
$$\eta = \nabla_\theta \psi(\theta) = \mathbb{E}_{p(\cdot; \theta)}[T(X)]$$

The dual potential function $\varphi(\eta)$ is obtained via the **Legendre-Fenchel transformation**:
$$\varphi(\eta) = \sup_{\theta \in \Theta} \left( \langle \theta, \eta \rangle - \psi(\theta) \right) = \langle \theta(\eta), \eta \rangle - \psi(\theta(\eta))$$
For exponential families, $\varphi(\eta) = -H(p_\eta)$ is the **negative Shannon differential entropy**. The dually flat manifold satisfies exact zero Legendre gap:
$$\psi(\theta) + \varphi(\eta) - \langle \theta, \eta \rangle = 0$$

### 2.2 The Fisher-Rao Metric and Natural Gradients
The Riemannian metric tensor on the statistical manifold $\mathcal{M}$ is uniquely given by the Fisher Information Matrix (Chentsov's Theorem):
$$g_{ij}(\theta) = \mathbb{E}_{p_\theta}\left[ \frac{\partial \log p(X; \theta)}{\partial \theta^i} \frac{\partial \log p(X; \theta)}{\partial \theta^j} \right] = \frac{\partial^2 \psi}{\partial \theta^i \partial \theta^j}$$
In dual expectation coordinates $\eta$, the dual Riemannian metric is the inverse Fisher Information Matrix:
$$g^{ij}(\eta) = \frac{\partial^2 \varphi}{\partial \eta^i \partial \eta^j} = \left[ g_{ij}(\theta) \right]^{-1}$$

When optimizing an objective loss $\mathcal{L}(\theta)$ over a probability manifold, the steepest descent direction invariant to reparameterization is the **Natural Gradient** (*Amari 1998*):
$$\tilde{\nabla}_\theta \mathcal{L} = G(\theta)^{-1} \nabla_\theta \mathcal{L} = \nabla_\eta \mathcal{L}$$
Natural gradient descent directly corresponds to standard Euclidean gradient descent in the dual expectation coordinate system.

### 2.3 Dual $\alpha$-Connections, Geodesics, and Bregman Divergence
Amari's one-parameter family of affine connections $\nabla^{(\alpha)}$ defines dual flat geometries:
1. **$e$-Connection ($\alpha = 1$, Exponential):** Flat in primal natural coordinates $\theta$. Geodesics are straight lines in parameter space:
   $$\theta(t) = (1 - t) \theta_1 + t \theta_2$$
   The $e$-projection onto a sub-manifold minimizes the **Reverse KL divergence** (mode-seeking, Variational Inference):
   $$\theta^* = \arg\min_{\theta \in \mathcal{S}} D_{\text{KL}}(q \| p_\theta)$$
2. **$m$-Connection ($\alpha = -1$, Mixture):** Flat in dual expectation coordinates $\eta$. Geodesics are straight lines in expectation space:
   $$\eta(t) = (1 - t) \eta_1 + t \eta_2$$
   The $m$-projection onto an exponential family minimizes the **Forward KL divergence** (mean-seeking, Maximum Likelihood / Moment Matching):
   $$\eta^* = \arg\min_{\theta \in \mathcal{S}} D_{\text{KL}}(p_\theta \| q) \implies \eta^* = \mathbb{E}_q[T(X)]$$

The canonical divergence between two distributions on a dually flat manifold coincides with the **Bregman Divergence** generated by $\psi$:
$$D_\psi(\theta_2, \theta_1) = \psi(\theta_2) - \psi(\theta_1) - \langle \nabla\psi(\theta_1), \theta_2 - \theta_1 \rangle = D_{\text{KL}}(p_{\theta_1} \| p_{\theta_2})$$

### 2.4 Algorithmic Information, Shtarkov NML & Prequential MDL
The Minimum Description Length principle selects hypotheses that minimize the combined description length of the model and the data given the model. The **Normalized Maximum Likelihood (`NML`)** distribution (*Shtarkov 1987; Rissanen 1996*) achieves exact minimax regret:
$$p_{\text{NML}}(x^n) = \frac{p(x^n; \hat{\theta}(x^n))}{\sum_{y^n \in \mathcal{X}^n} p(y^n; \hat{\theta}(y^n))}$$
where $\text{COMP}(n) = \sum_{y^n} p(y^n; \hat{\theta}(y^n))$ represents the **parametric complexity** of the model class $\mathcal{M}$. Asymptotically, for a regular $d$-dimensional model:
$$\log \text{COMP}(n) = \frac{d}{2} \log\left(\frac{n}{2\pi}\right) + \log \int_{\Theta} \sqrt{\det G(\theta)} d\theta + o(1)$$

In online streaming settings, **Prequential (Predictive) MDL** computes cumulative sequential log-loss under predictive distributions:
$$L_{\text{preq}}(x^n) = -\sum_{t=1}^n \log p(x_t \mid x^{t-1}; \hat{\theta}_{t-1})$$
For Bernoulli sequences under the Krichevsky-Trofimov estimator $\hat{p}_t = \frac{s_{t-1} + 1/2}{t}$, prequential coding matches the NML minimax bound up to an $O(1)$ constant without computing full combinatorial sums.

### 2.5 Variational Free Energy and Active Inference
Friston's Variational Free Energy $\mathcal{F}(q)$ establishes an upper bound on surprise (negative marginal log-likelihood $-\log p(x)$) for an agent with internal representation $q(z)$ observing external sensory data $x$:
$$\mathcal{F}(q, x) = \mathbb{E}_{q(z)}\left[ \log q(z) - \log p(x, z) \right] = D_{\text{KL}}(q(z) \| p(z)) + \mathbb{E}_{q(z)}\left[ -\log p(x \mid z) \right]$$
- **Perception:** Minimizes $\mathcal{F}$ with respect to the recognition density $q(z)$, driving $q(z) \to p(z \mid x)$.
- **Action:** Modifies the external environment to generate observations $x$ that minimize expected free energy under prior preferences $p(x)$.

---

## 3. Tensor Mechanics & Dual-Engine Architecture

### 3.1 Tensor Dimensions and Layout Contract
The implementation operates across 4 core tensors:
- Primal Parameter: $\theta \in \mathbb{R}^K$ (float32).
- Dual Parameter: $\eta = \text{Softmax}(\theta) \in \Delta^{K-1}$ (float32).
- Fisher-Rao Metric Tensor: $G(\theta) \in \mathbb{R}^{K \times K}$, where $G_{ij} = \eta_i \delta_{ij} - \eta_i \eta_j$.
- Linear Anchor: $W \in \mathbb{R}^{8 \times 4}$, $b \in \mathbb{R}^4$, $X \in \mathbb{R}^{4 \times 8} \to Y \in \mathbb{R}^{4 \times 4}$.

```
  Primal Natural Coordinate theta [K] 
           │
           ▼ (Gradient of Log-Partition nabla psi = Softmax)
  Dual Expectation Coordinate eta [K] on Simplex
           │
           ├───────────────────────────────┐
           ▼                               ▼
  Fisher-Rao Metric G = diag(eta) - eta*eta^T    Dual Potential phi(eta) = sum eta*log(eta)
           │                               │
           └───────────────┬───────────────┘
                           ▼
             Legendre Gap: |psi + phi - <theta, eta>| == 0.0
```

### 3.2 Dual-Engine Execution Pipeline (WebGPU & CPU)
1. **WGSL Compute Pipeline (`DUAL_GEODESIC_SHADER`):** Dispatches workgroups of size 64 to compute parallel dual $\alpha$-geodesic linear interpolations across batches of distributions on GPU hardware.
2. **Native CPU SIMD Engine:** Computes stable `logsumexp`, closed-form Fisher-Rao outer products, Shtarkov combinatorial sums, and Variational Free Energy covariance trace reductions on host CPU with zero external dependencies.

---

## 4. Multi-Ecosystem Parity & Cross-Track Verification
Transitive numerical parity $\|Y_A - Y_B\|_\infty < 10^{-5}$ is enforced across all four runtime tracks using golden checkpoint tensors stored in `weights/01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.safetensors`:

| Component Operator | PyTorch 2.x | JAX / NumPy | Candle (Rust) | Kernels (WGSL/CPU) | Parity Max Residual |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Anchor** | $4 \times 4$ GEMM | $4 \times 4$ Dot | Tensor Matmul | Vectorized GEMM | $< 1.0 \times 10^{-7}$ |
| **Log-Partition $\psi(\theta)$** | `torch.logsumexp` | `np.log(sum(exp))` | Stable max-shift | Scalar reduction | $< 1.0 \times 10^{-7}$ |
| **Expectation $\eta = \nabla\psi$** | `F.softmax` | `exp / sum` | `candle_nn::softmax` | Elementwise normalize | $< 1.0 \times 10^{-7}$ |
| **Legendre Gap** | $|\psi + \varphi - \theta^T\eta|$ | Vector dot | Tensor broadcast | Dot product | $< 2.4 \times 10^{-7}$ |
| **Fisher-Rao Metric $G$** | `diag - outer` | `diag - outer` | Dense $K \times K$ | Flattened matrix | $< 1.0 \times 10^{-7}$ |
| **$e$-Geodesic ($\alpha=1$)** | $(1-t)\theta_1 + t\theta_2$ | Vector lerp | Tensor affine | WGSL / SIMD lerp | $< 1.0 \times 10^{-7}$ |
| **$m$-Geodesic ($\alpha=-1$)** | $(1-t)\eta_1 + t\eta_2$ | Vector lerp | Tensor affine | WGSL / SIMD lerp | $< 1.0 \times 10^{-7}$ |
| **Bregman Divergence** | $\psi_2 - \psi_1 - \eta_1^T \Delta\theta$ | Vector dot | Tensor reduction | Dot reduction | $< 1.0 \times 10^{-7}$ |
| **Shtarkov NML Complexity** | Combinatorial sum | Combinatorial sum | Combinatorial sum | Dynamic factorials | $< 1.0 \times 10^{-7}$ |
| **Prequential MDL** | Sequential KT loss | Sequential KT loss | Sequential KT loss | Streaming loop | $< 1.0 \times 10^{-7}$ |
| **Free Energy $\mathcal{F}(q)$** | Gaussian KL + Recon | Exact scalar | Exact scalar | Exact scalar | $< 1.0 \times 10^{-7}$ |

---

## 5. Pathological Failure Modes & Anti-Patterns

### 5.1 Naive Euclidean Interpolation of Probability Distributions
Linearly interpolating distribution parameters in Euclidean space ($\theta_{\text{mid}} = \frac{1}{2}(\theta_1 + \theta_2)$) fails to respect the underlying Riemannian geometry. For Gaussians $\mathcal{N}(\mu, \sigma^2)$, Euclidean interpolation of variances yields intermediate distributions with artificially reduced entropy or incorrect variance manifolds. On the simplex, the Euclidean midpoint does not correspond to the geodesic in the Fisher-Rao metric, causing severe likelihood degradation.

### 5.2 Two-Part Crude MDL vs. Normalized Maximum Likelihood (`NML`)
Crude two-part MDL codes model parameters with an arbitrary fixed discretization grid $\frac{k}{2} \log N$, ignoring the parameter manifold's Riemannian volume $\int \sqrt{\det G(\theta)} d\theta$. In nested model comparison (e.g., polynomial degree selection), crude MDL either over-penalizes simple models or fails to detect parameter redundancies along singular Fisher directions, whereas Shtarkov NML automatically penalizes the true Riemannian volume of distinguishability.

### 5.3 Mode-Collapse in Reverse KL vs. Mean-Seeking Forward KL
Optimizing $D_{\text{KL}}(q \| p)$ ($e$-projection / Reverse KL) forces $q$ to zero wherever $p$ is zero, causing catastrophic mode-dropping when approximating multimodal target densities with unimodal representations. Conversely, $D_{\text{KL}}(p \| q)$ ($m$-projection / Forward KL) forces $q$ to cover the entire support of $p$, yielding overly diffuse approximations with low peak fidelity.

---

## 6. Cross-Domain Generalisation & Production Workflows

```
                           Information Geometry & MDL
                                       │
     ┌──────────────────┬──────────────┴─────┬──────────────────┐
     ▼                  ▼                    ▼                  ▼
Natural Gradient    Active Inference     Universal Coding    Hyperspherical
Deep Learning       Robotics & Control   Model Selection     Embeddings
(K-FAC / Shampoo)   (Free Energy)        (Shtarkov NML)      (Fisher Manifold)
```

1. **Natural Gradient Optimization (K-FAC & Shampoo):** Preconditioning parameter gradients with block-diagonal or Kronecker-factored Fisher information matrices accelerates optimization in deep vision and transformer models, navigating flat saddle regions without plateau stalling.
2. **Active Inference & Robotics:** Robots minimize variational free energy over sensory-motor predictive loops, combining state estimation with epistemic exploration to actively resolve hidden obstacle uncertainties.
3. **Model Selection in Financial & Scientific ML:** Shtarkov NML complexity metrics enable exact, unbiased model selection across non-nested neural architectures without requiring cross-validation splits.

---

## 7. Verification & Benchmarking Suite
All components are validated through automated test harnesses:
- **Pytest:** `pytest tests/test_cross_track_parity.py` verifies cross-track parity within $10^{-5}$.
- **Cargo Nextest:** `cargo nextest run --workspace` executes 70 integration tests across `ai-candle` and `ai-kernel`.
- **Repository Validation:** `python scripts/verify_repo.py --run` validates all 23 educational notebooks under a strict 3.0s runtime budget.

---

## 8. Navigation & Sibling Track Links
- [Track 1: Synchronized Interactive Notebook](../../notebooks/01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.ipynb)
- [Track 2: Production PyTorch 2.x Module](../../pytorch2/01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.py)
- [Track 3: Pure Functional JAX / NumPy Reference](../../jax/01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.py)
- [Track 4: Native Rust Candle Implementation](../../candle/01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.rs)
- [Track 5: Dual-Engine Hardware Kernels (WGSL / CPU)](../../kernels/01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.rs)
- [Track 6: Golden Reference Weights](../../weights/01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.safetensors)
