# Graph Neural Networks: GCN, GAT, GIN, Graphormer, and SDRF Rewiring

## Executive Summary & ELI5

Relational and non-Euclidean data—ranging from molecular chemical graphs and social networks to abstract knowledge graphs and 3D meshes—cannot be processed using standard 2D convolutional grids or 1D sequential transformers without discarding permutation invariance and connectivity topology.

**Graph Neural Networks (GNNs)** and **Graph Transformers** generalize deep learning to topological graph spaces:
- **GCN (Graph Convolutional Networks)**: A localized first-order spectral approximation of ChebNet that aggregates localized neighbor representations weighted by symmetric degree normalization $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.
- **GAT (Graph Attention Networks)**: Replaces static topological degree weighting with learnable masked self-attention, dynamically weighting neighbors by semantic feature relevance.
- **GIN (Graph Isomorphism Networks)**: Establishes the theoretical upper bound of message-passing GNNs, proving that injective sum multiset aggregation achieves the exact discriminative power of the 1-Weisfeiler-Lehman (1-WL) graph isomorphism test.
- **Graphormer**: An all-pairs spatial Transformer for graphs that breaks the localized message-passing bottleneck by injecting shortest path distance encodings and degree centrality biases into global attention matrices.
- **SDRF Rewiring (Stochastic Discrete Ricci Flow)**: Diagnoses the fundamental vulnerability of deep GNNs—**oversquashing**—arising from negative Ricci curvature bottlenecks, and geometrically rewires graph edges to alleviate information bottlenecks.

---

## Mathematical Foundations & Governing Equations

### 1. GCN Spectral Normalization
For adjacency matrix $A \in \{0, 1\}^{N \times N}$, self-loops are added: $\tilde{A} = A + I_N$. The diagonal degree matrix is $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$. The layer-wise spectral propagation rule is:
$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

### 2. GIN and Weisfeiler-Lehman Isomorphism Equivalence
Xu et al. proved that standard mean and max aggregators fail to distinguish simple non-isomorphic multiset neighborhoods. GIN parameterizes an injective multiset aggregation function via:
$$h_i^{(l+1)} = \text{MLP}^{(l)}\left(\left(1 + \epsilon^{(l)}\right) h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} h_j^{(l)}\right)$$
where $\epsilon$ is a learnable scalar or fixed parameter.

### 3. Graphormer Spatial Structural Encoding
Graphormer computes dense all-pairs attention between every pair of nodes $(i, j)$ in the graph, injecting inductive bias via shortest path distance (SPD) $\phi(i, j)$:
$$A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(i, j)} + c_{\text{deg}^-}(i) + c_{\text{deg}^+}(j)$$
where $b_{\phi(i, j)}$ is an indexed learnable bias encoding spatial geodesic distance.

### 4. SDRF and Balanced Forman-Ricci Curvature
Information oversquashing occurs on edges with negative discrete Ricci curvature where neighborhood volumes grow exponentially. Balanced Forman-Ricci curvature on an edge $e = (i, j)$ is defined as:
$$\text{Ric}(i, j) = 4 - d_i - d_j + 3 \cdot \Delta(i, j)$$
where $d_i, d_j$ are node degrees and $\Delta(i, j) = \sum_k A_{ik} A_{jk}$ is the number of triangles supporting edge $e$.
Edges with $\text{Ric}(i, j) \ll 0$ act as information bottlenecks; SDRF rewires the graph by adding edges that triangulate negative-curvature bridges.

---

## Tensor Memory Layout & Implementation Details

### GCN Degree-Normalized Propagation
1. Adjacency with self-loops: $\tilde{A} = A + I_N \in \mathbb{R}^{N \times N}$
2. Degree inverse square root: $D^{-1/2} = \text{diag}((\sum_j \tilde{A}_{ij})^{-1/2}) \in \mathbb{R}^{N \times N}$
3. Normalized operator: $\hat{A} = D^{-1/2} \tilde{A} D^{-1/2} \in \mathbb{R}^{N \times N}$
4. Linear node projection: $H_{\text{proj}} = H \cdot W \in \mathbb{R}^{N \times D_{\text{out}}}$
5. Spatial message convolution: $H_{\text{out}} = \text{ReLU}(\hat{A} \cdot H_{\text{proj}} + b) \in \mathbb{R}^{N \times D_{\text{out}}}$

### GIN Multiset Aggregation
1. Graph neighbor sum: $H_{\text{agg}} = A \cdot H \in \mathbb{R}^{N \times D}$
2. Injective combination: $Z = (1 + \epsilon) H + H_{\text{agg}} \in \mathbb{R}^{N \times D}$
3. Multi-layer perceptron: $H_{\text{gin}} = \text{MLP}(Z) \in \mathbb{R}^{N \times D}$

---

## Multi-Ecosystem Implementation Matrix

| Operation / Feature | PyTorch 2.x | JAX (vmap + jit) | Candle (Rust) | WGSL Compute Shader |
| :--- | :--- | :--- | :--- | :--- |
| **Adjacency Normalization** | `diag(deg**-0.5)` + `@` | `jnp.diag` + pure matmul | `Tensor::eye` + explicit diagonal | Parallel degree reduction kernel |
| **GCN Aggregation** | Sparse / Dense `A_hat @ H` | Fused JIT sparse GEMM | `norm_adj.matmul(&h_proj)` | Fused row-scatter gather |
| **GIN Injective Update** | `(1+eps)*h + A@h` | Pure arithmetic vectorization | `affine` + `matmul` | Direct node thread accumulation |
| **Graphormer Attention** | Attention + SPD bias matrix | `jax.nn.softmax` + bias | `softmax_last_dim` + `spatial_bias` | Fused GEMM + Softmax pass |
| **Ricci Curvature Analysis** | `(4 - d_i - d_j + 3*tri)*A` | Vectorized outer sum | `sum_keepdim` + `broadcast_add` | Thread per edge index |

---

## Pathological Misuse & Failure Modes

1. **Mean/Max Aggregation in Discriminative Graph Classification**:
   - *Anti-Pattern*: Using GCN mean-aggregation or GraphSAGE max-pooling for molecular graph isomorphism tasks.
   - *Consequence*: Fails to distinguish basic non-isomorphic multiset neighborhoods (e.g. 2 neighbors of feature $x$ vs 4 neighbors of feature $x$), collapsing downstream property prediction.
   - *Remedy*: Use GIN sum-aggregation to guarantee 1-WL expressivity.

2. **Oversquashing in Deep Message Passing Networks**:
   - *Anti-Pattern*: Stacking 16–32 localized message-passing layers to expand receptive fields on tree-like or hierarchical graphs.
   - *Consequence*: Exponential information compression into fixed-size node vectors creates severe oversquashing bottlenecks.
   - *Remedy*: Perform SDRF Ricci curvature rewiring or employ Graphormer all-pairs global attention.

3. **Oversmoothing from Infinite GCN Depth**:
   - *Anti-Pattern*: Stacking unregularized GCN layers without skip connections.
   - *Consequence*: Normalized Laplacian powers $\hat{A}^L$ converge to the stationary distribution, rendering node features identical across the entire graph.
   - *Remedy*: Add residual skip connections ($H + H_{\text{next}}$) or initial residual connections (APPNP/GCNII).

---

## Cross-Domain Generalisation Matrix

| Domain | Node Entities $V$ | Edge Relations $E$ | Crucial Inductive Bias |
| :--- | :--- | :--- | :--- |
| **Small-Molecule Drug Discovery** | Atoms (C, N, O, H) | Covalent chemical bonds | GIN multiset isomorphism distinction |
| **Social & Citation Networks** | Users / Scientific papers | Follows / Citations | GCN degree normalization & homophily |
| **Protein Structure Modeling** | Amino acid residues | Spatial proximity (< 8Å) | Graphormer shortest-path spatial bias |
| **Power Grid & Traffic Flow** | Substations / Road junctions | Power lines / Highways | SDRF curvature bottleneck alleviation |

---

## Verification, Parity & Empirical Benchmarks

The cross-track implementation across PyTorch 2.x, JAX, Candle, and WGSL is validated against `weights/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.safetensors` initialized with deterministic seed 42.

- **Parity Metrics**:
  - `linear_anchor`: Parity error $< 10^{-6}$ across CPU and GPU.
  - `gcn_normalize_adjacency`: Exact degree matrix inversion and symmetric normalization.
  - `gcn_layer`: Numerical parity on linear transformation, neighborhood aggregation, and ReLU activation.
  - `gin_layer`: Exact alignment for $(1 + \epsilon)$ scaled multiset sum aggregation.
  - `graphormer_attention`: Structural spatial bias addition validated to $< 10^{-6}$.
  - `sdrf_ricci_curvature`: Exact discrete Forman curvature evaluation across all existing edges.

---

## Synchronized Track Artifacts & Navigation

- **PyTorch 2.x Reference**: [`pytorch2/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.py)
- **Functional JAX Track**: [`jax/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.py)
- **Candle Engine Implementation**: [`candle/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.rs)
- **Hardware Kernels & WGSL**: [`kernels/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.rs)
- **Integration Tests**: [`crates/ai-candle/tests/05_architectures/test_34_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.rs`](file:///c:/Users/Liam/Code/AI-Examples/crates/ai-candle/tests/05_architectures/test_34_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.rs)
- **Interactive Jupyter Notebook**: [`notebooks/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.ipynb)
- **Golden Weights**: [`weights/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.safetensors)
