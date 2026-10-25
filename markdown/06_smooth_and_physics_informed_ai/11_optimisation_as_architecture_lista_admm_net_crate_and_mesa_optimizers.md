# Optimisation as Architecture: LISTA, ADMM-Net, CRATE & Mesa-Optimizers

## 1. Algorithm Unrolling & Neural Optimization Hybrids

Classical optimization algorithms (ISTA, FISTA, ADMM, Primal-Dual splitting) provide mathematically grounded convergence guarantees and interpretable parameters. However, on practical tasks such as sparse coding, compressive sensing, and magnetic resonance imaging (MRI) reconstruction, their convergence can be painfully slow, often requiring thousands of iterations.

Conversely, standard black-box deep neural networks (CNNs, Transformers) achieve rapid feedforward inference and high empirical accuracy, but act as uninterpretable black boxes that lack physical domain inductive biases and structural safety bounds.

**Algorithm Unrolling (Deep Unfolding)** systematically bridges this dichotomy:
1. Truncate an iterative mathematical optimization algorithm to a fixed number of steps $K$ (e.g., $K=3 \dots 10$).
2. Map each optimization step $k$ directly to a layer in a deep feedforward network.
3. Relax rigid algorithm matrices (e.g., dictionary projections $D^T D$ or measurement filters) into **learnable weight parameters**, trained end-to-end via gradient descent.

The resulting network architecture inherits the structural interpretability and inductive biases of the underlying optimization scheme while achieving orders-of-magnitude acceleration in convergence speed.

---

## 2. LISTA: Learned Iterative Shrinkage-Thresholding

Consider the classic $\ell_1$-regularized sparse coding / LASSO problem:
$$\min_{z \in \mathbb{R}^N} \frac{1}{2} \|y - D z\|_2^2 + \lambda \|z\|_1,$$
where $y \in \mathbb{R}^M$ is a noisy measurement and $D \in \mathbb{R}^{M \times N}$ ($M < N$) is an overcomplete dictionary.

Standard Iterative Shrinkage-Thresholding Algorithm (ISTA) updates:
$$z_{k+1} = \eta_{\lambda / L}\left( z_k - \frac{1}{L} D^T (D z_k - y) \right) = \eta_{\theta}\left( \left(I - \frac{1}{L} D^T D\right) z_k + \frac{1}{L} D^T y \right),$$
where $\eta_\theta(u) = \operatorname{sign}(u) \odot \max(|u| - \theta, 0)$ is the soft-thresholding proximal operator.

Gregor & LeCun (2010) introduced **Learned ISTA (LISTA)** by parametrizing the iterative step as a recurrent neural layer:
$$z_{k+1} = \eta_{\theta_k}(z_k W_s + y W_e),$$
where:
- $W_e \approx \frac{1}{L} D^T$ ($M \times N$) lifts the raw measurements into the sparse latent space.
- $W_s \approx I - \frac{1}{L} D^T D$ ($N \times N$) acts as a lateral inhibition / Gram matrix.
- $\theta_k \in \mathbb{R}^N$ is a learned per-layer threshold vector.

While standard ISTA requires hundreds of iterations to achieve an $\epsilon$-accurate sparse code, LISTA learns adapted dictionary metrics that converge in as few as $3$ to $5$ unrolled layers.

---

## 3. ADMM-Net: Unrolling the Alternating Direction Method of Multipliers

For constrained or composite inverse problems:
$$\min_{x, z} f(x) + g(z) \quad \text{s.t.} \quad A x + B z = c,$$
the **Alternating Direction Method of Multipliers (ADMM)** decouples $f$ and $g$ by introducing an augmented Lagrangian:
$$\mathcal{L}_\rho(x, z, u) = f(x) + g(z) + u^T (A x + B z - c) + \frac{\rho}{2} \|A x + B z - c\|_2^2.$$

In compressive sensing MRI reconstruction ($f(x) = \frac{1}{2} \|A x - y\|_2^2$, $g(z) = \lambda \|z\|_1$, with constraint $x - z = 0$):
1. **Primal Step 1**:
   $$x_{k+1} = \arg\min_x \mathcal{L}_\rho(x, z_k, u_k) = (A^T A + \rho I)^{-1} \bigl( A^T y + \rho (z_k - u_k) \bigr).$$
2. **Primal Step 2 (Proximal Shrinkage)**:
   $$z_{k+1} = \arg\min_z \mathcal{L}_\rho(x_{k+1}, z, u_k) = \operatorname{Prox}_{\lambda / \rho}(x_{k+1} + u_k).$$
3. **Dual Multiplier Update**:
   $$u_{k+1} = u_k + x_{k+1} - z_{k+1}.$$

**ADMM-Net** unrolls these three updates into a structured multi-stage neural network where:
- The inversion matrix $(A^T A + \rho I)^{-1}$ is replaced by a learned convolutional inverse filter $W_{\text{inv}}$.
- The penalty parameter $\rho$ and threshold $\lambda$ are learned scalar parameters.
- The dual state $u_k$ is propagated as an internal recurrent memory feature.

---

## 4. CRATE: White-Box Transformers from Coding Rate Reduction

Standard Vision Transformers (ViT) and language models are composed of alternating multi-head self-attention (MSA) and multi-layer perceptron (MLP) blocks whose architectural rationale was historically empirical.

**CRATE (Coding Rate Reduction Transform Architecture)** proves that canonical Transformer blocks mathematically emerge from **unrolling an optimization algorithm for maximal coding rate reduction**.

The optimization objective maximizes the rate reduction $\Delta R(Z)$:
$$\max_{Z} \Delta R(Z) = R(Z) - R_c(Z),$$
where $R(Z) = \frac{1}{2} \log \det(I + \frac{d}{m \epsilon^2} Z Z^T)$ measures the compactness of representations across all tokens, and $R_c(Z) = \sum_{j=1}^K \frac{m_j}{2m} \log \det(I + \frac{d}{m_j \epsilon^2} Z_j Z_j^T)$ measures diversity across sub-clusters.

When optimizing this objective via alternating proximal gradient steps:
1. **Multi-Head Self-Attention (MSA)** emerges as the gradient step maximizing global subspace volume expansion:
   $$Z^{(k + 1/2)} = Z^{(k)} + \eta \operatorname{Attention}(Z^{(k)}).$$
2. **MLP Blocks with Soft-Thresholding** emerge as the proximal sparse coding step compressing token features onto low-dimensional subspaces:
   $$Z^{(k+1)} = \operatorname{Prox}_{\lambda} \bigl( Z^{(k + 1/2)} - \eta Z^{(k + 1/2)} (D^T D - I) \bigr).$$

CRATE demonstrates that Transformers are not arbitrary heuristic black boxes, but unrolled optimization algorithms seeking structured low-dimensional subspace arrangements.

---

## 5. Mesa-Optimizers & Meta-Learning

When deep networks or reinforcement learning agents are trained by gradient descent (the **base optimizer**), their inner representations can self-organize into an internal optimization algorithm that acts at runtime (**mesa-optimization**):

$$\min_\theta \mathbb{E}_{\tau} \left[ \mathcal{L}_{\text{outer}}\bigl( z^\star(x; \theta) \bigr) \right] \quad \text{s.t.} \quad z^\star(x; \theta) = \arg\min_z \mathcal{L}_{\text{inner}}(z; x, \theta).$$

Key properties of mesa-optimizers:
- **In-Context Learning as Gradient Descent**: Von Oswald et al. (2023) and Dai et al. (2023) demonstrated that causal autoregressive Transformers simulate iterative gradient descent steps on an implicit linear regression loss within their forward activation flow.
- **Model-Agnostic Meta-Learning (MAML)**: Explicitly unrolls $T$ steps of inner-loop gradient descent:
  $$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{train}}(\theta),$$
  and differentiates through the inner unrolled trajectory to optimize the outer initialization $\theta$.

---

## 6. Asymptotic Equivalence & Weight Tying

When an unrolled architecture increases its depth $K \to \infty$, three distinct architectural regimes emerge:

1. **Untied Unrolling (Standard Feedforward)**: Each layer has independent weights $\{W_e^{(k)}, W_s^{(k)}, \theta^{(k)}\}_{k=1}^K$. This provides maximal capacity but increases parameter count linearly with depth.
2. **Weight-Tied Deep Equilibrium (DEQ)**: Constraining weights to be identical across layers ($W_s^{(k)} \equiv W_s$) transforms the unrolled network into a **Deep Equilibrium Model**:
   $$z^\star = \operatorname{Prox}_\theta(z^\star W_s + y W_e).$$
   The network can be trained via the Implicit Function Theorem in $\mathcal{O}(1)$ memory.
3. **Averaged & Accelerated Schemes**: Introducing Nesterov momentum into unrolled layers yields **Learned FISTA**, accelerating convergence from $\mathcal{O}(1/K)$ to $\mathcal{O}(1/K^2)$.

---

## 7. Numerical Verification & Structural Stability

To guarantee that unrolled optimization architectures remain numerically stable during training and deployment:
1. **Contraction Certified Weight Matrices**:
   In LISTA, the state transition matrix $W_s$ must satisfy $\|W_s\|_2 < 1$. Normalizing $W_s \leftarrow W_s / (\rho(W_s) + \epsilon)$ prevents iterative state explosion across unrolled layers.
2. **Subgradient Stability in Soft-Thresholding**:
   The derivative of soft-thresholding $\eta_\theta(x)$ is discontinuous at $|x| = \theta$:
   $$\eta_\theta'(x) = \mathbb{I}(|x| > \theta).$$
   In deep backward passes, smooth approximations (e.g., smoothed Huber or GELU-proximal relaxations) prevent gradient vanishing.

---

## 8. Cross-Track Verification & Architectural Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every unrolled optimization operator obeys the following strict invariants:

| Operator | Verification Specification | Tolerance |
| :--- | :--- | :--- |
| **Soft-Thresholding Parity** | $\operatorname{sign}(x) \max(|x| - \theta, 0)$ matches exactly across backends | $< 10^{-6}$ |
| **LISTA Unrolling Parity** | $K=3$ layer recurrent state $z_3$ matches golden checkpoint | $< 10^{-5}$ |
| **ADMM-Net State Invariants** | Primal states $(x_1, z_1)$ and dual multiplier $u_1$ match golden | $< 10^{-5}$ |
| **CRATE Gram Reduction** | Proximal gradient step with dictionary $D$ matches reference | $< 10^{-5}$ |
| **Memory & Parameter Parity** | Weight-tied operators achieve identical outputs with stationary weights | $< 10^{-5}$ |
