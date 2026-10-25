# AI Analysis: Representation Similarity (CKA, RSA) and TwoNN Intrinsic Dimension

## 1. Mathematical Foundations & Lineage

Comparing neural network representations across distinct layers, random seeds, training checkpoints, and architectures is essential for understanding representation learning and functional convergence.

The lineage of representational similarity and geometry metrics comprises:
1. **Representational Similarity Analysis (RSA — Kriegeskorte et al. 2008)**: Originating in cognitive neuroscience to align fMRI brain responses with computational models, RSA characterizes geometry via pairwise Representational Dissimilarity Matrices (RDMs).
2. **Canonical Correlation Analysis (CCA & SVCCA — Raghu et al. 2017; Morcos et al. 2018)**: Applied linear projection to maximize correlation between latent subspaces, but struggled with sensitivity to small singular values and non-isotropic noise.
3. **Centered Kernel Alignment (CKA — Kornblith et al. 2019)**: Proved that Linear CKA based on the Hilbert-Schmidt Independence Criterion (HSIC) is strictly invariant to orthogonal rotations and isotropic scaling while robustly matching layers across diverse architectures.
4. **TwoNN Intrinsic Dimension Estimation (Facco et al. 2017)**: Discovered that the ratio of distances to the second and first nearest neighbors $\mu = r_2 / r_1$ follows a Pareto distribution governed solely by the local intrinsic manifold dimension $d$, enabling training trajectory manifold compression audits.

---

## 2. Centered Kernel Alignment (Linear CKA)

Given representations $X \in \mathbb{R}^{N \times D_X}$ and $Y \in \mathbb{R}^{N \times D_Y}$ for $N$ shared evaluation instances, linear Gram matrices are:
$$K = X X^T \in \mathbb{R}^{N \times N}, \quad L = Y Y^T \in \mathbb{R}^{N \times N}$$
Let $H = I_N - \frac{1}{N} \mathbf{1}\mathbf{1}^T$ be the centering projector. The centered Gram matrices are:
$$K_c = H K H, \quad L_c = H L H$$
The empirical Hilbert-Schmidt Independence Criterion (HSIC) is the matrix inner product:
$$\text{HSIC}(K, L) = \frac{1}{(N-1)^2} \text{Tr}(K_c L_c) = \frac{1}{(N-1)^2} \sum_{i, j} K_{c, ij} L_{c, ij}$$
Linear CKA normalizes HSIC to $[0, 1]$:
$$\text{CKA}(X, Y) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$
CKA is invariant to orthogonal rotations ($X Q$ with $Q^T Q = I$) and uniform isotropic scaling ($c X$), ensuring identical scores regardless of internal coordinate bases.

---

## 3. Representational Similarity Analysis (RSA)

RSA computes pairwise dissimilarity across stimuli rather than aligning individual feature dimensions directly.
Given representation $X \in \mathbb{R}^{N \times D}$:
$$\text{RDM}_{X}(i, j) = \| x_i - x_j \|_2 \quad \text{or} \quad 1 - \frac{(x_i - \bar{x}_i) \cdot (x_j - \bar{x}_j)}{\|x_i - \bar{x}_i\|_2 \|x_j - \bar{x}_j\|_2}$$
To compare two models $A$ and $B$, the upper-triangular off-diagonal entries $\mathbf{u}_A, \mathbf{u}_B \in \mathbb{R}^{N(N-1)/2}$ are correlated:
$$\text{RSA}(A, B) = \frac{(\mathbf{u}_A - \bar{\mathbf{u}}_A) \cdot (\mathbf{u}_B - \bar{\mathbf{u}}_B)}{\|\mathbf{u}_A - \bar{\mathbf{u}}_A\|_2 \|\mathbf{u}_B - \bar{\mathbf{u}}_B\|_2}$$

---

## 4. TwoNN Intrinsic Manifold Dimension Estimation

High-dimensional neural representations often lie on low-dimensional curved submanifolds. The TwoNN estimator (*Facco et al. 2017*) estimates intrinsic dimension $d$ using only the first two nearest neighbors of each point, eliminating sensitivity to inhomogeneous global densities.

For each point $i$, let $r_1(i)$ be the Euclidean distance to its nearest neighbor, and $r_2(i)$ be the distance to its second nearest neighbor ($r_2 > r_1$).
The ratio $\mu_i = \frac{r_2(i)}{r_1(i)} \ge 1$ follows a Pareto distribution:
$$P(\mu) = 1 - \mu^{-d}, \quad \mu \in [1, \infty)$$
The closed-form Maximum Likelihood Estimator (MLE) for intrinsic dimension is:
$$\hat{d} = \frac{N}{\sum_{i=1}^N \ln(\mu_i)}$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`linear_cka`** | Gram centering + HSIC normalized dot | Array outer products + HSIC | Double centering + scalar norm | `CENTER_GRAM_WGSL` |
| **`rsa_correlation`** | Upper-triangle index extract + Pearson | `jnp.triu_indices` + Pearson | Upper-triangular iterator | Vectorized Pearson reducer |
| **`twonn_intrinsic_dimension`** | $\mu$ ratio log sum quotient | Log ratio reduction | Vectorized log accumulator | Parallel ratio reducer |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Uncentered Gram Artifacts in CKA**:
   - *Symptom*: CKA reports 0.99 similarity between two completely independent representations.
   - *Cause*: Large shared mean activation shifts dominate raw inner products $K \cdot L$ without double-centering $H K H$.
   - *Fix*: Double-center all Gram matrices: $K_c = K - \text{mean}(K, 0) - \text{mean}(K, 1) + \text{mean}(K)$.
2. **TwoNN Boundary Effect Distortions**:
   - *Symptom*: Intrinsic dimension estimates diverge when sample size $N$ is too small ($N < 50$).
   - *Cause*: Sample sparsity causes the 2nd nearest neighbor to reach beyond the local tangent plane.
   - *Fix*: Ensure $N \ge 500$ points and trim the top 10% highest-$\mu$ outliers.
3. **CCA Invertibility Failure on Wide Layers**:
   - *Symptom*: Classical CCA produces `NaN`s when comparing wide over-parameterized layers ($D > N$).
   - *Cause*: Covariance matrices are rank-deficient and non-invertible.
   - *Fix*: Use Linear CKA, which operates entirely on $N \times N$ Gram matrices and requires zero matrix inversions.

---

## 7. Real-World Engineering Applications

1. **Cross-Architecture Representation Alignment**: Comparing representations learned by Vision Transformers (ViT) vs. Convolutional Networks (ConvNeXt) to identify structural differences in inductive biases.
2. **Phase-Transition & Grokking Diagnostics**: Tracking intrinsic dimension contraction over training epochs to detect the onset of algorithmic generalization.
3. **Model Compression Fidelity Auditing**: Verifying whether pruned or 4-bit quantized models preserve the representational geometry of full-precision teacher checkpoints.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `cka_score_ref`: Linear CKA similarity scalar between two representations.
3. `rsa_corr_ref`: RSA Pearson correlation between representational dissimilarity matrices.
4. `twonn_id_ref`: TwoNN estimated intrinsic manifold dimension.
