# Energy-Based Models, RBM, and Hopfield Networks

## Executive Summary & ELI5

Most modern deep learning models are directed feedforward predictors that map an input directly to an output. In contrast, **Energy-Based Models (EBMs)** assign an unnormalized scalar score—an "energy" $E(x)$—to configurations of variables. Desirable configurations (realistic images, valid protein conformations, correct logical proofs) are assigned low energy, while unnatural or conflicting configurations receive high energy.

This topic explores the lineage connecting classical statistical mechanics to modern transformer attention:
- **Energy-Based Models & Langevin MCMC**: Formulate probability as a Boltzmann distribution $p(x) \propto e^{-E(x)}$. Sampling is executed by following the negative energy gradient $\nabla_x E(x)$ with injected Gaussian noise via Stochastic Gradient Langevin Dynamics (SGLD).
- **Restricted Boltzmann Machines (RBMs)**: Bipartite Markov Random Fields with visible and hidden units. Because visible units are conditionally independent given hidden units, exact Gibbs sampling and Contrastive Divergence (CD-k) learning are tractable. Marginalizing hidden variables yields closed-form free energy.
- **Modern / Dense Hopfield Networks**: Expand classical associative memories (which possessed limited capacity $\approx 0.14 d$) to continuous continuous state spaces with exponential storage capacity ($C \sim 2^{d/2}$). Crucially, Krotov, Hopfield, and Ramsauer et al. proved that the single-step retrieval update rule of modern continuous Hopfield networks is mathematically identical to the **Transformer Dot-Product Attention mechanism**.

---

## Mathematical Foundations & Governing Equations

### 1. Boltzmann Distribution and Free Energy
Given energy function $E_\theta(x)$, the probability density is given by the Gibbs-Boltzmann distribution:
$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}, \quad Z_\theta = \int e^{-E_\theta(x)} dx$$
In a Restricted Boltzmann Machine with visible units $v \in \mathbb{R}^V$ and binary hidden units $h \in \{0, 1\}^H$, joint energy is:
$$E(v, h) = -v^T W h - a^T v - b^T h$$
Marginalizing over all $2^H$ hidden configurations yields the **Free Energy** $\mathcal{F}(v)$ such that $p(v) \propto e^{-\mathcal{F}(v)}$:
$$\mathcal{F}(v) = -a^T v - \sum_{j=1}^H \log\left(1 + e^{W_{\cdot j}^T v + b_j}\right) = -a^T v - \sum_{j=1}^H \text{softplus}\left(W_{\cdot j}^T v + b_j\right)$$

### 2. Overdamped Langevin Dynamics
To sample from $p(x) \propto e^{-E(x)}$ without calculating the intractable partition function $Z$, overdamped Langevin dynamics integrates the stochastic differential equation:
$$d x_t = -\frac{1}{2} \nabla_x E(x_t) dt + d W_t$$
Discretized with step size $\epsilon > 0$ and standard Gaussian noise $z_k \sim \mathcal{N}(0, I)$:
$$x_{k+1} = x_k - \frac{\epsilon}{2} \nabla_x E(x_k) + \sqrt{\epsilon} z_k$$
As $\epsilon \to 0$ and $k \to \infty$, the stationary distribution of $x_k$ converges exactly to $p(x)$.

### 3. Modern Continuous Hopfield Networks
Given a set of $M$ stored memory patterns $X = [x_1, \dots, x_M]^T \in \mathbb{R}^{M \times D}$, the continuous modern Hopfield energy function is:
$$E(\xi) = -\frac{1}{\beta} \text{lse}(\beta, X \xi) + \frac{1}{2} \|\xi\|^2 + C$$
where $\text{lse}(\beta, z) = \log\left(\sum_{m=1}^M e^{\beta z_m}\right)$.
By taking the concave-convex procedure (CCCP) or gradient descent on this energy surface, the fixed-point state update is:
$$\xi^{\text{new}} = X^T \text{softmax}\left(\beta X \xi\right)$$
Setting queries $Q = \xi$, keys $K = X$, and values $V = X$ with $\beta = \frac{1}{\sqrt{D}}$ proves that **self-attention is a modern continuous Hopfield memory retrieval step**.

---

## Tensor Memory Layout & Implementation Details

### RBM Free Energy Computation
- Visible state: $v \in \mathbb{R}^{B \times V}$
- Weights: $W \in \mathbb{R}^{V \times H}$
- Visible bias: $a \in \mathbb{R}^V$, Hidden bias: $b \in \mathbb{R}^H$
- Pre-activation: $Z = v W + b \in \mathbb{R}^{B \times H}$
- Hidden energy reduction: $H_{\text{term}} = \sum_{j=1}^H \text{softplus}(Z_{ij})$
- Output Free Energy: $\mathcal{F}(v) = -(v \cdot a) - H_{\text{term}} \in \mathbb{R}^{B \times 1}$

### Hopfield Attention Equivalence
- Query token: $q \in \mathbb{R}^{B \times D}$
- Memory pattern matrix: $M \in \mathbb{R}^{M \times D}$
- Dot-product similarity logits: $S = \beta \cdot q M^T \in \mathbb{R}^{B \times M}$
- Associative softmax attention weights: $P = \text{Softmax}(S, \text{dim}=-1) \in \mathbb{R}^{B \times M}$
- Retrieved associative pattern: $q^{\text{retrieved}} = P M \in \mathbb{R}^{B \times D}$

---

## Multi-Ecosystem Implementation Matrix

| Operation / Feature | PyTorch 2.x | JAX (vmap + jit) | Candle (Rust) | WGSL Compute Shader |
| :--- | :--- | :--- | :--- | :--- |
| **RBM Free Energy** | `softplus` + `sum(keepdim=True)` | `jax.nn.softplus` + `jnp.sum` | `exp()?.affine(1,1)?.log()` | Fused loop per visible row |
| **Langevin Step** | Vectorized linear combination | Functional scalar-broadcast | `affine` chaining | Thread per coordinate |
| **LogSumExp Energy** | `torch.logsumexp` | `jax.nn.logsumexp` | Shifted max trick + `sum_keepdim` | Workgroup parallel reduction |
| **Hopfield Retrieval** | `F.softmax(beta*q@M.T) @ M` | `jax.nn.softmax` + `@` | `softmax_last_dim` + `matmul` | Fused GEMM + Softmax pipeline |
| **Parity Tolerance** | Native float32 | $< 10^{-6}$ vs golden | $< 10^{-6}$ vs golden | $< 10^{-5}$ vs golden |

---

## Pathological Misuse & Failure Modes

1. **Langevin Step Size Instability**:
   - *Anti-Pattern*: Choosing an overly aggressive Langevin step size ($\epsilon > 0.1$) without gradient clipping.
   - *Consequence*: The MCMC chain diverges exponentially into high-energy sparse regions, causing exploding numerical infinities (`NaN`).
   - *Remedy*: Clip energy gradients $\|\nabla_x E(x)\|$ and use annealed Langevin dynamics with step sizes decreasing along noise levels.

2. **Inverse Temperature Degeneracy in Hopfield Retrieval**:
   - *Anti-Pattern*: Setting $\beta \to 0$ or failing to scale $\beta$ proportionally to inverse feature dimension ($1/\sqrt{D}$).
   - *Consequence*: When $\beta \to 0$, retrieval collapses to an uninformative uniform average over all stored memories. When $\beta \to \infty$, gradient flow ceases due to one-hot hard-max saturation.
   - *Remedy*: Scale $\beta = \frac{1}{\sqrt{D}}$ or tune $\beta$ based on memory density.

3. **Spurious Energy Minima**:
   - *Anti-Pattern*: Training EBMs solely with Maximum Likelihood without negative phase regularization.
   - *Consequence*: The model creates deep artificial local energy wells far from the training distribution, capturing Langevin chains in nonsensical states.
   - *Remedy*: Use Contrastive Divergence with persistent replay buffers or Score Matching objectives.

---

## Cross-Domain Generalisation Matrix

| Domain | Energy Function $E(x)$ | Visible / Hidden Variables | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **Associative Memory (Neuroscience)** | Hopfield Hamiltonian | Neuronal spike patterns | Pattern completion from partial cues |
| **Transformer Architectures** | Continuous Modern Hopfield Energy | Query tokens / Key-Value memories | Attention-based contextual retrieval |
| **Molecular Structure & Docking** | Interatomic potential energy | 3D coordinate atoms / conformational states | Energy minimization to equilibrium |
| **Robotic Trajectory Generation** | Obstacle avoidance potential field | Trajectory waypoints / control actions | Langevin trajectory sampling |

---

## Verification, Parity & Empirical Benchmarks

The multi-ecosystem implementations across PyTorch 2.x, JAX, Candle, and WGSL are validated against `weights/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.safetensors` initialized with deterministic seed 42.

- **Parity Verification**:
  - `linear_anchor`: Parity error $< 10^{-6}$ across CPU and GPU.
  - `rbm_free_energy`: Exact numerical parity on marginal softplus reductions.
  - `langevin_dynamics_step`: Exact float32 match for stochastic gradient updates.
  - `modern_hopfield_energy`: Log-sum-exp shift invariance validated across platforms.
  - `modern_hopfield_retrieve`: Exact equivalence to Transformer dot-product attention.

---

## Synchronized Track Artifacts & Navigation

- **PyTorch 2.x Reference**: [`pytorch2/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.py)
- **Functional JAX Track**: [`jax/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.py)
- **Candle Engine Implementation**: [`candle/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.rs)
- **Hardware Kernels & WGSL**: [`kernels/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.rs)
- **Integration Tests**: [`crates/ai-candle/tests/05_architectures/test_33_energy_based_models_rbm_and_hopfield_networks.rs`](file:///c:/Users/Liam/Code/AI-Examples/crates/ai-candle/tests/05_architectures/test_33_energy_based_models_rbm_and_hopfield_networks.rs)
- **Interactive Jupyter Notebook**: [`notebooks/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.ipynb)
- **Golden Weights**: [`weights/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.safetensors)
