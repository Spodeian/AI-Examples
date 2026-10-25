# Quantile and Information-Theoretic Quantization: NF4, SF4, Lloyd-Max, and Dynamic Tree

Quantization can be formulated as a lossy source coding problem governed by Shannon's rate-distortion theory. Uniform quantization maps continuous intervals of equal width $\Delta$ to discrete indices, which is asymptotically optimal for uniformly distributed data. However, pretrained deep neural network weights and activations follow heavy-tailed or bell-shaped continuous distributions (such as Gaussian $\mathcal{N}(0, \sigma^2)$ or Student-$t$). Uniform quantization of such distributions wastes representation budget on low-probability tail regions or incurs severe saturation clipping distortion. Information-theoretic and quantile quantization techniques construct non-uniform codebooks where each quantization bin either contains an equal amount of probability mass (maximizing mutual information) or directly minimizes expected mean squared error (MSE).

---

## 1. Rate-Distortion Theory & Optimal Quantizer Conditions

For a continuous random variable $X \sim p_X(x)$ quantized to $K$ discrete levels $\mathcal{C} = \{c_0, c_1, \dots, c_{K-1}\}$ with decision boundaries $\{b_0, b_1, \dots, b_K\}$:

$$
D = \mathbb{E}[(X - Q(X))^2] = \sum_{k=0}^{K-1} \int_{b_k}^{b_{k+1}} (x - c_k)^2 p_X(x) \, dx
$$

Subject to fixed codebook cardinality $K = 2^b$ (e.g., $b=4$ bits, $K=16$), Shannon's rate-distortion theorem dictates that the theoretical minimum distortion $D(R)$ is bounded by the differential entropy $h(X) = -\int p_X(x) \log_2 p_X(x) \, dx$. An information-maximising (equal-probability) quantizer enforces:

$$
\mathbb{P}(b_k \le X < b_{k+1}) = \int_{b_k}^{b_{k+1}} p_X(x) \, dx = \frac{1}{K} \quad \forall k \in \{0, \dots, K-1\}
$$

Such quantile quantizers maximize the discrete entropy $H(Q(X)) = \log_2 K = b$, preventing underutilized bit states and eliminating information redundancy.

---

## 2. NormalFloat 4 (NF4) & QLoRA Quantile Mapping

NormalFloat 4 (NF4), introduced in QLoRA (Dettmers et al., 2023), is an information-theoretically optimal quantile quantization data type for zero-mean normally distributed weights $\mathcal{N}(0, 1)$. Standard uniform INT4 allocates discrete integers with uniform spacing, leaving the Gaussian tails underrepresented or clipped.

NF4 derives 16 quantization bins by computing the quantiles of a standard normal distribution:

$$
q_i = \Phi^{-1}\left(\frac{i}{2 \cdot 16}\right) \quad \text{for negative quantiles and symmetry adjustments}
$$

Because exact quantiles of $\mathcal{N}(0, 1)$ do not have an exact zero point (essential for padding and zero-valued sparse weights), NF4 computes separate quantiles for the negative and positive half-planes and normalizes the resulting codebook such that $c_7 = 0.0$ and the maximum absolute entry is $1.0$:

$$\mathcal{C}_{\text{NF4}} = \begin{bmatrix}
-1.0000000, & -0.6961928, & -0.5250731, & -0.3949175, \\
-0.2844414, & -0.1847734, & -0.0910500, & 0.0000000, \\
0.0795803, & 0.1609302, & 0.2461123, & 0.3379152, \\
0.4407098, & 0.5626170, & 0.7229568, & 1.0000000
\end{bmatrix}$$

Block quantization scales each block of $B=64$ elements by $s = \max_{j} |w_j|$, normalizes $w / s \in [-1, 1]$, maps to the nearest entry in $\mathcal{C}_{\text{NF4}}$, and reconstructs during forward pass via $\hat{w} = c_{\text{opt}} \cdot s$.

---

## 3. Student-t Float 4 (SF4) for Heavy-Tailed Distributions

While pretrained weights frequently exhibit Gaussian statistics, intermediate activation states and specific transformer layer matrices (e.g., down-projection in MLP blocks) frequently exhibit excess kurtosis and heavy tails, closely following a Student-$t$ distribution with degrees of freedom $\nu \in [2, 5]$:

$$
p_{\nu}(x) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\nu \pi} \, \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}
$$

Student-t Float 4 (SF4) computes quantile intervals from the cumulative distribution function $F_{\nu}(x)$. The resulting levels reserve wider dynamic range for extreme outliers while retaining dense sampling near zero:

$$\mathcal{C}_{\text{SF4}} = \begin{bmatrix}
-1.0000, & -0.7303, & -0.5512, & -0.4072, \\
-0.2885, & -0.1852, & -0.0905, & 0.0000, \\
0.0792, & 0.1614, & 0.2486, & 0.3448, \\
0.4571, & 0.5982, & 0.7681, & 1.0000
\end{bmatrix}$$

When evaluated on tensors with high kurtosis, SF4 yields an improvement of $+1.5$ to $+3.0$ dB in Signal-to-Quantization-Noise Ratio (SQNR) over standard NF4.

---

## 4. The Lloyd-Max Iterative Scalar Optimization

When the underlying data distribution $p_X(x)$ is empirical or does not fit a closed-form analytical density, the Lloyd-Max algorithm iteratively converges to the local minimum MSE quantizer. It satisfies two necessary optimality conditions:

1. **Nearest-Neighbor Condition (Optimal Partitions):**
   Given centroids $c_k$, the decision boundaries must be the midpoints:
   $$b_k = \frac{c_{k-1} + c_k}{2}$$
2. **Centroid Condition (Optimal Reconstruction):**
   Given partition intervals $[b_k, b_{k+1})$, the reconstruction level $c_k$ must be the conditional expectation (center of mass):
   $$c_k = \mathbb{E}[X \mid b_k \le X < b_{k+1}] = \frac{\int_{b_k}^{b_{k+1}} x \, p_X(x) \, dx}{\int_{b_k}^{b_{k+1}} p_X(x) \, dx}$$

In finite discrete sample sets, this corresponds to $1\text{D}$ $K$-means clustering, alternating between assignment to the nearest centroid and updating centroids to the mean of each cluster.

---

## 5. Dynamic Tree Quantization

The Dynamic Tree Quantizer constructs a hierarchical adaptive binary tree over continuous scalar values. At each tree level up to depth $d$ (yielding $K = 2^d$ leaves):
1. Compute the empirical median of the current node data: $m = \text{median}(S)$.
2. Split samples into left child $S_{\le} = \{x \in S : x \le m\}$ and right child $S_{>} = \{x \in S : x > m\}$.
3. Recurse until depth $d$ is reached.
4. Set leaf reconstruction values to the conditional sample mean: $c_j = \frac{1}{|S_j|} \sum_{x \in S_j} x$.

Because splits occur at empirical medians, every leaf interval contains equal sample counts, directly approximating an empirical quantile codebook in $\mathcal{O}(N \log K)$ time without iterative clustering loops.

---

## 6. Precision & Distortion Metrics

Quantization performance across non-uniform formats is quantified through:

- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2$$
- **Signal-to-Quantization-Noise Ratio (SQNR):**
  $$\text{SQNR (dB)} = 10 \log_{10} \left( \frac{\sum_{i=1}^N x_i^2}{\sum_{i=1}^N (x_i - \hat{x}_i)^2} \right)$$
- **Entropy Utilization:**
  $$H(Q) = -\sum_{k=0}^{K-1} p_k \log_2 p_k \quad \text{where } p_k = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(q_i = k)$$

For Gaussian tensors, standard INT4 achieves $\sim 18.2\text{ dB}$, while NF4 achieves $\sim 21.4\text{ dB}$, representing a $2.1\times$ reduction in variance error at identical 4-bit storage.

---

## 7. Cross-Framework Numerical Parity

To ensure reproducible deployment across training and inference backends, all four quantizers (NF4, SF4, Lloyd-Max, and Dynamic Tree) are verified with deterministic seed-42 golden references:

| Framework | Implementation File | Precision Parity |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/01_foundations_formats_and_observers/03_...py` | Reference |
| **JAX** | `jax/14_quantisation/01_foundations_formats_and_observers/03_...py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/01_foundations_formats_and_observers/03_...rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/01_foundations_formats_and_observers/03_...rs` | WGSL / CPU |

Consistent tie-breaking and indexing conventions (e.g. median index $(N-1)/2$) guarantee zero drift across CPU and GPU hardware platforms.

---

## 8. Production Applications & QLoRA Deployment

In production parameter-efficient fine-tuning (PEFT):
1. **Base Model Freezing:** 70B+ parameter LLMs have their weight matrices quantized into NF4 blocks ($B=64$), reducing memory footprint from 140 GB (FP16) to $\sim 35\text{ GB}$.
2. **Double Quantization:** The block scales $s$ are themselves quantized into FP8, saving an additional 0.37 bits per parameter.
3. **On-the-Fly Dequantization:** During the forward and backward passes, WGSL or Triton kernels dequantize NF4 indices back to BF16 registers, compute GEMM with LoRA adapters in full precision, and discard uncompressed activations immediately.
