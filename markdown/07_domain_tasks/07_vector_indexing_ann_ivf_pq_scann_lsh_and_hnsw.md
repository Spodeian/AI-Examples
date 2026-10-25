# Vector Indexing & ANN: IVF, Product Quantization, ScaNN, LSH & HNSW

## 1. The Billion-Scale Vector Search Challenge: Exact vs. Approximate

Dense embeddings from modern foundation models (LLMs, CLIP, Audio Spectrograms) require searching billions of $D$-dimensional vectors ($D \in [128, 1536]$):
- **Exact Flat Search (Brute Force)**: Evaluates $N$ Euclidean distances or inner products. Time complexity $\mathcal{O}(N \cdot D)$ becomes prohibitive ($> 1\text{ s}$ per query for $N = 10^7$).
- **The Curse of Dimensionality**: Traditional spatial search trees ($k\text{-d}$ trees, R-trees) degenerate to brute-force linear scans when $D > 20$.
- **Approximate Nearest Neighbor (ANN) Search**: Sacrifices a negligible fraction of recall ($< 1\text{--}5\%$) to achieve sub-millisecond retrieval latencies $\mathcal{O}(\log N)$ or $\mathcal{O}(1)$ via space partitioning, vector quantization, and graph indexing.

---

## 2. Inverted File Index (IVF): Voronoi Space Partitioning

IVF partitions high-dimensional vector space into $K$ Voronoi cells using k-means clustering:
1. **Offline Centroid Clustering**: Learn centroids $\{c_1, \dots, c_K\} \subset \mathbb{R}^D$ on representative corpus vectors.
2. **Inverted List Ingestion**: Each vector $x_i$ is assigned to its nearest centroid $c^\star = \arg\min_k \|x_i - c_k\|_2$ and appended to inverted list $\mathcal{I}[c^\star]$.
3. **Multi-Probe Querying ($n_{\text{probe}}$)**: At query time, the query $q$ finds its $n_{\text{probe}} \ll K$ nearest centroids. Only vectors within those $n_{\text{probe}}$ inverted lists are scanned.

Increasing $n_{\text{probe}}$ smoothly navigates the Pareto frontier between retrieval recall and query latency.

---

## 3. Product Quantization (PQ) & Asymmetric Distance Computation (ADC)

Jégou et al. (2011) introduced **Product Quantization (PQ)** to compress high-dimensional vectors by $16\times\text{--}64\times$ while enabling blazing-fast distance computations.

### 3.1 Subspace Decomposition
A vector $x \in \mathbb{R}^D$ is split into $M$ orthogonal subvectors of dimension $d = D / M$:
$$x = [x^{(1)}, x^{(2)}, \dots, x^{(M)}], \quad x^{(m)} \in \mathbb{R}^d.$$
For each subspace $m \in \{1, \dots, M\}$, a separate codebook of $K$ centroids (typically $K = 256 = 2^8$, requiring $1$ byte per subvector) is trained via k-means.

### 3.2 Asymmetric Distance Computation (ADC)
At query time, the query $q$ is *not* quantized. We construct a precomputed distance lookup table $T \in \mathbb{R}^{M \times K}$:
$$T[m, k] = \|q^{(m)} - c_k^{(m)}\|_2^2.$$
For any quantized document $x$ represented by code tuple $(k_1, \dots, k_M) \in \{0, \dots, 255\}^M$, its asymmetric squared distance is computed via $M$ fast table lookups and additions without floating-point multiplies:
$$d(q, x)^2 \approx \sum_{m=1}^M T[m, k_m].$$

---

## 4. ScaNN: Anisotropic Vector Quantization

Guo et al. (2020) demonstrated that standard PQ minimizes reconstruction error $\|x - \tilde{x}\|_2^2$, which is suboptimal for **Maximum Inner Product Search (MIPS)**.

### 4.1 Orthogonal vs. Parallel Quantization Error
In MIPS, the inner product error is:
$$\langle q, x \rangle - \langle q, \tilde{x} \rangle = \langle q, x - \tilde{x} \rangle.$$
Decomposing residual error $e = x - \tilde{x}$ into components parallel and orthogonal to $x$:
$$e_\parallel = \frac{\langle e, x \rangle}{\|x\|_2^2} x, \quad e_\perp = e - e_\parallel.$$
Since high-dimensional nearest neighbors tend to be directionally aligned with query $q$, error in the parallel direction $e_\parallel$ directly distorts vector rank orderings, whereas orthogonal error $e_\perp$ is largely cancelled out in expectation.

### 4.2 Anisotropic Loss Objective
ScaNN heavily penalizes parallel quantization error via anisotropic weighting parameter $\omega > 0$:
$$\mathcal{L}_{\text{anisotropic}}(x, \tilde{x}) = \|e_\perp\|_2^2 + (1 + \omega) \|e_\parallel\|_2^2.$$
This anisotropic distortion metric yields up to $2\times$ higher QPS at $95\%$ recall compared to classical isotropic PQ.

---

## 5. Locality-Sensitive Hashing (LSH)

Locality-Sensitive Hashing hashes vectors into buckets such that similar vectors collide with high probability:
- **Random Hyperplane Hashing (SimHash)**: Given random normal hyperplane normals $r_1, \dots, r_B \sim \mathcal{N}(0, I_D)$:
  $$h_b(x) = \operatorname{sign}(\langle x, r_b \rangle) \in \{-1, +1\}.$$
- **Angular Collision Probability**: For unit vectors $u, v$:
  $$P(h(u) = h(v)) = 1 - \frac{\theta(u, v)}{\pi}, \quad \theta(u, v) = \arccos(\langle u, v \rangle).$$
Comparing bit vectors using hardware POPCNT and XOR yields sub-linear candidate filtering.

---

## 6. Hierarchical Navigable Small World (HNSW)

Malkov & Yashunin (2018) introduced **HNSW**, the state-of-the-art graph-based indexing algorithm:
1. **Multi-Layer Skip-List Structure**: Builds a hierarchy of proximity graphs with exponentially sparser density at upper layers.
2. **Logarithmic Greedy Traversal**: Search initiates at the top sparse layer, performing greedy best-first routing until a local minimum is reached, then descends to the next layer.
3. **Small-World Navigability**: Every node maintains short-range local edges (high clustering) and long-range shortcuts (small diameter), achieving $\mathcal{O}(\log N)$ query complexity.

---

## 7. Comparative ANN Taxonomy & System Benchmarks

| Algorithm | Index Structure | Memory Footprint | Query Latency | Recall @ 10 | Construction Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flat L2 / IP** | Contiguous Array | $1.0\times$ (Raw FP32) | High ($\mathcal{O}(N)$) | $100\%$ (Exact) | $\mathcal{O}(1)$ |
| **IVF-Flat** | Voronoi Centroid Lists| $1.05\times$ | Low ($\mathcal{O}(n_{\text{probe}} \frac{N}{K})$) | $90\text{--}98\%$ | Fast (k-means) |
| **IVF-PQ** | Quantized Codes | **$0.03\text{--}0.06\times$** | **Lowest (ADC)** | $80\text{--}95\%$ | Moderate |
| **ScaNN** | Anisotropic PQ + IVF | $0.05\text{--}0.10\times$ | **Ultra-Low** | **$92\text{--}98\%$** | Moderate |
| **HNSW** | Multi-Layer Graph | $1.5\text{--}2.5\times$ (High RAM) | Lowest (Graph Walk) | **$98\text{--}99.9\%$** | Slowest |

---

## 8. Cross-Track Verification & Systems-Level Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every ANN indexing operator complies with strict numerical standards:

| Operator | Verification Metric | Tolerance | Numerical Invariant |
| :--- | :--- | :--- | :--- |
| **PQ Distance Table** | $\|q^m - c_k^m\|_2^2$ squared Euclidean distances | $< 10^{-5}$ | Strictly non-negative: $T[m, k] \ge 0$ |
| **ADC Distance Sum** | Exact table lookup accumulation $\sum_m T[m, I_m]$ | $< 10^{-5}$ | Matches analytical sum of looked-up values |
| **ScaNN Anisotropic Loss** | $\|e_\perp\|^2 + (1 + \omega) \|e_\parallel\|^2$ | $< 10^{-5}$ | Upper bounded by $(1+\omega) \|e\|^2$ |
| **LSH Random Projections** | Hyperplane sign dot product $\frac{1}{B} \sum b_q \cdot b_d$ | $< 10^{-5}$ | Strictly bounded in $[-1.0, +1.0]$ |
| **Linear Anchor** | Shared cross-track regression anchor | $< 10^{-5}$ | Deterministic match against golden tensor |


