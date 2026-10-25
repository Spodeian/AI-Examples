# Heterophilic Graphs, Temporal Networks, Cellular Sheaves, and Knowledge Graph Embeddings

## Executive Summary & ELI5

Classical Graph Neural Networks make a foundational assumption known as **homophily**: that connected nodes share similar attributes, labels, or functional classes ("birds of a feather flock together"). However, many real-world systems exhibit extreme **heterophily** (disassortativity): fraudsters systematically interact with legitimate accounts in financial graphs; chemical amino acids bond with complementary rather than identical residues; and predator species connect to prey. Under heterophily, standard low-pass GCN filters act as destructive smoothers, averaging conflicting features into useless uniform distributions.

This topic unifies advanced topological, dynamic, and relational graph architectures designed to excel beyond homophily:
- **Heterophilic Spatial Filtering**: Decouples graph message passing into separate self, low-pass ($A_{\text{low}} = \hat{A}$), and high-pass ($A_{\text{high}} = I - \hat{A}$) spectral channels, allowing the model to learn signed differences across dissimilar connected neighbors.
- **Cellular Sheaf Neural Networks**: Generalizes graphs to algebraic topology using **cellular sheaves**. By assigning vector spaces (stalks) to each node and learnable linear transport maps (restriction maps) to edges, the **Sheaf Laplacian** allows parallel transport across non-trivial topological bundles, enabling diffusion to preserve antipodal features without collapsing.
- **Temporal Graph Networks (TGN)**: Generalizes static graphs to continuous-time dynamic event streams using continuous time-delta harmonic Fourier encodings.
- **Knowledge Graph Embeddings (KGE: TransE & RotatE)**: Embeds multi-relational graphs consisting of entity-relation triplets $(h, r, t)$. **RotatE** maps relations to rotations in complex vector space $\mathbb{C}^d$, elegantly modeling symmetry, antisymmetry, inversion, and composition.

---

## Mathematical Foundations & Governing Equations

### 1. Heterophilic Decoupled Filtering
In heterophilic networks, high-frequency signals carry the primary discriminative signal. The graph Laplacian is decomposed into low-pass and high-pass operators:
$$\hat{A}_{\text{low}} = \hat{A}, \quad \hat{A}_{\text{high}} = I_N - \hat{A}$$
The decoupled heterophilic layer projects self, low-pass, and high-pass signals independently:
$$H^{(l+1)} = \text{ReLU}\left(H^{(l)} W_{\text{self}} + \left(\hat{A}_{\text{low}} H^{(l)}\right) W_{\text{low}} + \left(\hat{A}_{\text{high}} H^{(l)}\right) W_{\text{high}}\right)$$

### 2. Cellular Sheaves and the Sheaf Laplacian
A cellular sheaf $\mathcal{F}$ over a graph $G = (V, E)$ assigns a vector space $\mathcal{F}(v) = \mathbb{R}^{d_v}$ to each node $v$ and $\mathcal{F}(e) = \mathbb{R}^{d_e}$ to each edge $e = (u, v)$, along with linear restriction maps $\mathcal{F}_{v \trianglelefteq e}: \mathcal{F}(v) \to \mathcal{F}(e)$.
The **Sheaf Laplacian** $L_{\mathcal{F}}$ acts on the direct sum space of node stalks:
$$(L_{\mathcal{F}} x)_v = \sum_{e=(u, v)} \mathcal{F}_{v \trianglelefteq e}^T \left(\mathcal{F}_{v \trianglelefteq e} x_v - \mathcal{F}_{u \trianglelefteq e} x_u\right)$$
For a 1D orthogonal sheaf where $\mathcal{F}_{v \trianglelefteq e} \in \{-1, +1\}$, the Sheaf Laplacian simplifies to:
$$L_{\mathcal{F}} = D - S \odot A$$
where $S_{ij} \in \{-1, +1\}$ denotes the connection sign (parallel vs antipodal transport).
Continuous sheaf diffusion evolves via:
$$\frac{d H(t)}{dt} = -L_{\mathcal{F}} H(t) \implies H^{(l+1)} = H^{(l)} - \Delta t \cdot L_{\mathcal{F}} H^{(l)}$$

### 3. RotatE Complex Relational Geometry
Given head entity $h \in \mathbb{C}^d$, tail entity $t \in \mathbb{C}^d$, and relation $r \in \mathbb{C}^d$ constrained to unit modulus $|r_i| = 1$ ($r_i = e^{i \theta_i}$):
$$h \circ r \approx t \iff (h_{\text{re}} + i h_{\text{im}}) \circ (\cos \theta + i \sin \theta) \approx (t_{\text{re}} + i t_{\text{im}})$$
The distance score is the complex Euclidean metric:
$$d_r(h, t) = -\|h \circ r - t\|_2 = -\sqrt{\sum_{k=1}^d \left( (h_{\text{re}} \cos \theta - h_{\text{im}} \sin \theta - t_{\text{re}})^2 + (h_{\text{re}} \sin \theta + h_{\text{im}} \cos \theta - t_{\text{im}})^2 \right)}$$

---

## Tensor Memory Layout & Implementation Details

### Sheaf Laplacian Buffer Matrix
For $N$ nodes and 1D stalks:
- Adjacency matrix: $A \in \{0, 1\}^{N \times N}$
- Edge signs: $S \in \{-1, 1\}^{N \times N}$
- Degree matrix: $D = \text{diag}(\sum_j A_{ij}) \in \mathbb{R}^{N \times N}$
- Sheaf Laplacian: $L_{\mathcal{F}} = D - (S \odot A) \in \mathbb{R}^{N \times N}$
- Diffusion update: $H_{\text{next}} = H - \Delta t \cdot (L_{\mathcal{F}} \cdot H) \in \mathbb{R}^{N \times D}$

### RotatE Complex Embedding Layout
- Head entity: $h = [h_{\text{re}} \in \mathbb{R}^{B \times D}, h_{\text{im}} \in \mathbb{R}^{B \times D}]$
- Relation angle: $\theta \in \mathbb{R}^{B \times D}$
- Tail entity: $t = [t_{\text{re}} \in \mathbb{R}^{B \times D}, t_{\text{im}} \in \mathbb{R}^{B \times D}]$
- Output score: $s \in \mathbb{R}^{B \times 1}$

---

## Multi-Ecosystem Implementation Matrix

| Operation / Feature | PyTorch 2.x | JAX (vmap + jit) | Candle (Rust) | WGSL Compute Shader |
| :--- | :--- | :--- | :--- | :--- |
| **High-Pass Graph Filter** | `(I - A_hat) @ H @ W_high` | Vectorized `jnp.matmul` | `(eye - norm_adj).matmul(h)` | Fused kernel aggregation |
| **1D Sheaf Laplacian** | `D - S * A` | `jnp.diag(deg) - S * A` | Explicit diagonal + matrix sub | Parallel edge difference pass |
| **Sheaf Diffusion** | `H - dt * (L @ H)` | Pure functional update | `h - diff` tensor arithmetic | Elementwise state update |
| **RotatE Complex Score** | Complex multiplication + norm | Pure JAX complex ops | Explicit Cartesian components | Trigonometric rotation + reduction |
| **Parity Tolerance** | Native float32 | $< 10^{-6}$ vs golden | $< 10^{-6}$ vs golden | $< 10^{-5}$ vs golden |

---

## Pathological Misuse & Failure Modes

1. **Applying Standard GCN to Disassortative Graphs**:
   - *Anti-Pattern*: Using classical GCN or mean-pooling GraphSAGE on fraud or bipartite dating networks where heterophily ratio $\mathcal{H} < 0.2$.
   - *Consequence*: Neighbors of opposing classes are forcefully averaged together, destroying linear separability and producing worse performance than an MLP ignoring graph structure entirely.
   - *Remedy*: Decouple high-pass filters ($I - \hat{A}$) or utilize cellular sheaf Laplacians.

2. **Unconstrained Modulus in RotatE**:
   - *Anti-Pattern*: Allowing relation embeddings $r \in \mathbb{C}^d$ to have unconstrained scaling factors ($|r_i| \neq 1$).
   - *Consequence*: The model scales embeddings rather than rotating them, losing the ability to model symmetry, inversion, and composition patterns.
   - *Remedy*: Parameterize relation embeddings strictly as phase angles $\theta \in [-\pi, \pi]$ and evaluate unit modulus components $(\cos \theta, \sin \theta)$.

3. **Time Leakage in Dynamic Temporal Graphs**:
   - *Anti-Pattern*: Aggregating future temporal edges $(u, v, t_2)$ when predicting an event at earlier timestamp $t_1 < t_2$.
   - *Consequence*: Causal temporal leakage inflates training evaluation while failing completely in production streaming environments.
   - *Remedy*: Strictly mask all edges $t > t_{\text{eval}}$ in temporal attention buffers.

---

## Cross-Domain Generalisation Matrix

| Graph Domain | Homophily Level | Geometric Structure | Architectural Paradigm |
| :--- | :--- | :--- | :--- |
| **Financial Anti-Fraud Networks** | Strongly Heterophilic | Bipartite Transaction Graph | Decoupled high-pass filtering / Sheaf Laplacians |
| **Biomedical Knowledge Graphs** | Multi-Relational | Directed Knowledge Base $(h, r, t)$ | RotatE complex relational geometry |
| **Continuous Streaming Social Feeds** | Dynamic Homophilic | Timestamped Event Graph | Temporal Graph Networks (TGN) with time Fourier bias |
| **Molecular Conformation Bundles** | Non-trivial Symmetry | Lie Group G-bundles | Cellular Sheaves with orthogonal restriction maps |

---

## Verification, Parity & Empirical Benchmarks

The multi-ecosystem implementations across PyTorch 2.x, JAX, Candle, and WGSL are validated against `weights/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.safetensors` initialized with deterministic seed 42.

- **Parity Metrics**:
  - `linear_anchor`: Parity error $< 10^{-6}$ across all tracks.
  - `heterophilic_filter`: Exact numerical agreement across self, low-pass, and high-pass spatial channels.
  - `cellular_sheaf_laplacian_1d`: Signed adjacency and diagonal degree matrix conservation.
  - `sheaf_diffusion_step`: Exact float32 match for continuous diffusion Euler integration.
  - `rotate_kge_score`: Precision match for complex trigonometric rotation and Euclidean norm.

---

## Synchronized Track Artifacts & Navigation

- **PyTorch 2.x Reference**: [`pytorch2/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.py)
- **Functional JAX Track**: [`jax/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.py)
- **Candle Engine Implementation**: [`candle/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.rs)
- **Hardware Kernels & WGSL**: [`kernels/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.rs)
- **Integration Tests**: [`crates/ai-candle/tests/05_architectures/test_35_heterophilic_temporal_cellular_sheaves_and_kge.rs`](file:///c:/Users/Liam/Code/AI-Examples/crates/ai-candle/tests/05_architectures/test_35_heterophilic_temporal_cellular_sheaves_and_kge.rs)
- **Interactive Jupyter Notebook**: [`notebooks/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.ipynb)
- **Golden Weights**: [`weights/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.safetensors)
