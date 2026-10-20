# Orthogonal and Preconditioned Matrix Optimisers (Topic 04)

## 1. Executive Summary & Intuitive Mental Model
- **Core Problem & Intuition:** Standard first-order optimisers (SGD, AdamW, Lion) flatten weight tensors into 1D vectors, ignoring the underlying 2D linear operator structure, matrix rank, and singular value spectra of deep neural network layers. Conversely, full second-order methods (Natural Gradient, Newton's method) compute curvature across all $N$ parameters, requiring an intractable $O(N^2)$ memory footprint and $O(N^3)$ matrix inversion. Orthogonal and preconditioned matrix optimisers exploit matrix geometry directly:
  1. **Muon (Momentum Orthogonalized by Newton-Schulz):** Projects momentum matrices onto the nearest semi-orthogonal matrix ($O = U V^T$ on the Stiefel manifold) using fast quintic Newton-Schulz iterations. Every update matrix has a flat singular spectrum ($\sigma_i = 1$), eliminating ill-conditioned coordinate updates and achieving target validation loss in ~35% fewer steps via pure BF16 tensor core GEMMs.
  2. **Shampoo (Kronecker-Factored 2D Curvature):** Accumulates left ($L \in \mathbb{R}^{m \times m}$) and right ($R \in \mathbb{R}^{n \times n}$) second-moment covariances, preconditioning gradients via $L^{-1/4} G R^{-1/4}$ to approximate full Fisher information in $O(m^2 + n^2)$ space.
  3. **GaLore (Gradient Low-Rank Projection):** Projects full dense gradients into a low-rank subspace ($G_{\text{low}} = P^T G$), updates Adam momentum buffers in low rank ($O(rn)$ memory instead of $O(mn)$), and projects back ($P \cdot \text{Adam}(G_{\text{low}})$), slashing optimizer VRAM by up to 65%.
  4. **SOAP (Second-Order Preconditioner in Eigenbasis):** Performs curvature preconditioning in the eigenbasis of the gradient covariance matrix.
- **Historical Lineage & Seminal Citations:**
  - *Gupta, Shamir, Sridharan (2018)*: **Shampoo**, introducing Kronecker-factored left/right second-moment preconditioning for tensor parameters.
  - *Anil et al. (2020)*: **Scalable Second Order Optimization for Deep Learning**, scaling distributed Shampoo across multi-node clusters.
  - *Zhao et al. (2024)*: **GaLore**, demonstrating full pretraining of 7B LLMs on single consumer GPUs using gradient low-rank projection.
  - *Vyas et al. (2024)*: **SOAP**, accelerating LLM pretraining via second-order eigenspace preconditioning.
  - *Jordan et al. (2024)*: **Muon**, introducing Newton-Schulz momentum orthogonalization as a high-performance optimizer for internal hidden layers in neural networks.
- **Decision Boundary:**
  - *When to use Muon:* Internal 2D hidden matrix layers ($W \in \mathbb{R}^{m \times n}$ with $m, n \ge 16$) in Transformers and modern vision/language models.
  - *When NOT to use Muon:* Never apply Muon to 1D vectors, LayerNorm weights, biases, token embeddings, or the output language model head. Always deploy a hybrid schedule: Muon optimizes internal 2D hidden weights while AdamW optimizes embeddings, normalization, and heads.
  - *When to use Shampoo / SOAP:* Frontier pretraining runs where accelerated wall-clock convergence justifies periodic $O(m^3 + n^3)$ matrix root calculations.
  - *When to use GaLore:* Memory-constrained pretraining or fine-tuning of large models where standard AdamW states exceed GPU memory limits.

## 2. Mathematical Foundations & First-Principles Derivations
- **Polar Decomposition & The Stiefel Manifold:**
  Given a 2D matrix $M \in \mathbb{R}^{m \times n}$ with singular value decomposition $M = U \Sigma V^T$, the polar decomposition decomposes $M$ into:
  $$M = O S, \quad \text{where } O = U V^T, \quad S = V \Sigma V^T$$
  The orthogonal factor $O$ is the exact metric projection of $M$ onto the Stiefel manifold $\text{St}(m, n) = \{Q \in \mathbb{R}^{m \times n} \mid Q^T Q = I_n\}$:
  $$O = \arg\min_{Q \in \text{St}(m, n)} \|M - Q\|_F$$
  $O$ equalizes all singular values to $\sigma_i(O) = 1$, eliminating directional stretching and coordinate imbalance.
- **Newton-Schulz Quintic Iteration:**
  Computing SVD on accelerators is slow and non-vectorized. Newton-Schulz iteratively orthogonalizes $M$ using matrix multiplications.
  Normalizing $X_0 = \frac{M}{\|M\|_F + \epsilon}$, assuming $m \le n$:
  $$A_k = X_k X_k^T \in \mathbb{R}^{m \times m}$$
  $$X_{k+1} = a X_k + b A_k X_k + c A_k^2 X_k$$
  Where the quintic polynomial coefficients are:
  $$a = 3.4445, \quad b = -4.7750, \quad c = 2.0315$$
  In $K=5$ iterations, singular values converge to $1.0 \pm 10^{-4}$ with high precision.
- **Muon Optimization Step:**
  For internal 2D weight matrix $W \in \mathbb{R}^{m \times n}$:
  $$M_t = \mu M_{t-1} + G_t$$
  $$V_t = \begin{cases} \mu M_t + G_t & \text{if Nesterov} \\ M_t & \text{otherwise} \end{cases}$$
  $$O_t = \text{NewtonSchulz}(V_t, \text{steps}=5)$$
  $$W_{t+1} = W_t (1 - \eta \lambda) - \eta O_t$$
- **Shampoo Kronecker-Factored Preconditioning:**
  For weight matrix $W \in \mathbb{R}^{m \times n}$:
  $$L_t = \beta_2 L_{t-1} + (1 - \beta_2) G_t G_t^T \in \mathbb{R}^{m \times m}$$
  $$R_t = \beta_2 R_{t-1} + (1 - \beta_2) G_t^T G_t \in \mathbb{R}^{n \times n}$$
  The preconditioned gradient update is:
  $$\tilde{G}_t = L_t^{-1/4} G_t R_t^{-1/4}$$
  Where $M^{-1/4}$ is computed via symmetric eigendecomposition $M = Q \Lambda Q^T \implies M^{-1/4} = Q (\max(\Lambda, \epsilon))^{-1/4} Q^T$.
  $$W_{t+1} = W_t (1 - \eta \lambda) - \eta \tilde{G}_t$$
- **GaLore (Gradient Low-Rank Projection):**
  Given projection matrix $P \in \mathbb{R}^{m \times r}$ with $P^T P = I_r$:
  $$G_{\text{low}} = P^T G_t \in \mathbb{R}^{r \times n}$$
  Standard Adam update on compact representation:
  $$M_t = \beta_1 M_{t-1} + (1 - \beta_1) G_{\text{low}}$$
  $$V_t = \beta_2 V_{t-1} + (1 - \beta_2) (G_{\text{low}} \odot G_{\text{low}})$$
  $$U_{\text{low}} = \frac{M_t / (1 - \beta_1^t)}{\sqrt{V_t / (1 - \beta_2^t)} + \epsilon}$$
  Project back to full dimension:
  $$U_t = P U_{\text{low}} \in \mathbb{R}^{m \times n}$$
  $$W_{t+1} = W_t (1 - \eta \lambda) - \eta U_t$$

## 3. Tensor Mechanics & Memory Layout Architecture
- **Memory Footprint Comparison for $4096 \times 4096$ Hidden Matrix ($d = 16.7\text{M}$ parameters):**
  - *Standard AdamW:* $2 \times 16.7\text{M} = 33.5\text{M}$ floats ($134.2\text{ MB}$).
  - *Muon:* $1 \times 16.7\text{M} = 16.7\text{M}$ floats ($67.1\text{ MB}$, 50% state savings; no variance buffers!).
  - *Shampoo:* $16.7\text{M} + 4096^2 + 4096^2 = 50.3\text{M}$ floats ($201.3\text{ MB}$).
  - *GaLore (rank $r=128$):* $2 \times (128 \times 4096) = 1.05\text{M}$ floats ($4.2\text{ MB}$, **97% state memory reduction!**).

```
Dense AdamW State:                      GaLore Low-Rank State:
┌───────────────────────────┐         P: [m x r]     M_low: [r x n]
│ m: [m x n]   v: [m x n]   │         ┌───┐          ┌───────────────┐
│                           │         │   │          │ r x n floats  │
│ 33.5M floats (134 MB)     │         └───┘          └───────────────┘
└───────────────────────────┘         1.05M floats (4.2 MB)
```

## 4. Multi-Ecosystem Comparative Implementation Matrix
| Feature / Characteristic | PyTorch 2.x | JAX / Equinox | Candle (Rust) | WGSL / Hardware Kernel |
| :--- | :--- | :--- | :--- | :--- |
| Muon Newton-Schulz | Fused matmul iteration | Functional `jax.lax` | `newton_schulz_quintic` | Fused `MUON_UPDATE_SHADER` |
| Shampoo Matrix Root | `torch.linalg.eigh` | `jax.scipy.linalg.eigh` | Jacobi diagonalization | CPU fallback / Jacobi |
| GaLore Subspace Projection | Tensor slice matmul | Functional projection | `step_galore` Candle | `galore_cpu` |
| Execution Speed | Tensor core GEMM | XLA JIT | Zero-alloc Candle | WGPU dispatch |

## 5. Pathological Failure Modes, Anti-Patterns & Diagnostic Guide
- **Applying Muon to 1D Parameters, Embeddings, or Heads:**
  - *Symptom:* Language model pretraining collapses with high perplexity or gradient NaN when applying Muon universally across all model parameters.
  - *Root Cause:* Muon assumes 2D matrix geometry with balanced aspect ratio. On token embedding tables ($V \times D$ where $V=128\text{k}, D=4096$) or 1D LayerNorm scales and biases, Newton-Schulz equalizes singular values across vocab tokens, destroying token frequency distributions and blowing up norms.
  - *Remedy:* Restrict Muon strictly to 2D hidden matrices ($W \in \mathbb{R}^{m \times n}$ where $m, n \ge 16$). Route embeddings, norms, and the final projection head to decoupled AdamW.
- **Ill-Conditioned Eigendecomposition in Shampoo:**
  - *Symptom:* Negative or infinite values inside $L^{-1/4}$ during Shampoo step.
  - *Root Cause:* Covariance matrices $L$ or $R$ accumulate near-zero eigenvalues along unvisited subspace directions. Without eigenvalue clamping, $\lambda_i^{-1/4} \to \infty$.
  - *Remedy:* Clamp eigenvalues $\lambda_i \ge \epsilon = 10^{-6}$ prior to negative fourth root power.

## 6. Cross-Domain Generalisation & Mathematical Isomorphisms
- **Differential Geometry:** Muon optimizes along Riemannian trajectories on the Stiefel manifold of semi-orthogonal frames, minimizing geodesic curvature distortion.
- **Quantum Mechanics:** Newton-Schulz polar decomposition is isomorphic to finding the closest unitary operator that diagonalizes density matrices in quantum state tomography.
- **Kronecker Product Signal Processing:** Shampoo's separable covariance decomposition mirrors 2D separable spatial filter design in image restoration and computer vision.

## 7. Verification, Parity & Benchmarking Protocol
- **Parity Tolerance:** Strict $\|Y_A - Y_B\|_\infty < 10^{-5}$ across PyTorch, JAX, Candle, and Hardware Kernels.
- **Reference Vectors:** Verified against `weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.safetensors`.
- **Target Invariants:**
  1. $p_{\text{muon}}$ and $m_{\text{muon}}$ match reference within $10^{-5}$.
  2. $p_{\text{shampoo}}$, $l_{\text{shampoo}}$, and $r_{\text{shampoo}}$ match reference within $10^{-5}$.
  3. $p_{\text{galore}}$, $m_{\text{galore}}$, and $v_{\text{galore}}$ match reference within $10^{-5}$.
  4. Linear Anchor forward matches reference within $10^{-5}$.

## 8. Navigation & Synchronized Track Artifacts
- [Interactive Concept Explorer Notebook](../../../notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.ipynb)
- [Production PyTorch 2.x Module](../../../pytorch2/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.py)
- [Functional JAX / Equinox Module](../../../jax/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.py)
- [Native Rust Candle Module](../../../candle/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.rs)
- [Hardware Kernels (WGSL + Native Fallback)](../../../kernels/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.rs)
