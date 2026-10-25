# Hypergraph Neural Networks: AllSet, Combinatorial Complexes, and Exphormer

## Executive Summary & ELI5

Standard graphs restrict relationships to pairwise edges $(u, v)$ connecting exactly two nodes. However, higher-order group interactions are ubiquitously multi-way: co-authors on a paper collaborate as an un-ordered group of arbitrary size; co-purchased shopping baskets contain multiple products simultaneously; and biochemical protein complexes involve tens of molecules interacting together. Decomposing these group interactions into pairwise cliques artificially introduces redundant edges and distorts true multi-way correlations.

**Hypergraph Neural Networks** and **Higher-Order Topological Transformers** process multi-way relational structures natively:
- **Hypergraph Convolutions (HGNN)**: Models multi-way relations via bipartite incidence matrices $H \in \{0, 1\}^{V \times E}$, propagating messages from nodes to hyperedges ($D_e^{-1} H^T X$), and subsequently from hyperedges back to nodes ($D_v^{-1} H Z_e$).
- **AllSet Framework (AllSetTransformer / AllDeepSets)**: Establishes a universal theoretical framework for hypergraph neural networks by parameterizing message passing as two alternating permutation-invariant multiset functions, proving universality across all multi-set aggregations.
- **Combinatorial & Cell Complexes**: Extends beyond pairwise graphs and simplicial complexes to cell complexes of arbitrary topological rank, enabling exact algebraic topology boundary computations on non-triangulated geometric domains.
- **Exphormer**: Solves the quadratic $O(N^2)$ bottleneck of dense Graph Transformers by constructing sparse attention graphs based on **expander graphs** (e.g. Ramanujan expanders). With constant node degree $d_{\text{exp}}$, expanders guarantee logarithmic spectral diameter and rapid information mixing in linear $O(V + E)$ time.

---

## Mathematical Foundations & Governing Equations

### 1. Hypergraph Incidence and Degree Normalization
A hypergraph $G = (V, E)$ consists of vertex set $V$ and hyperedge set $E$, where each hyperedge $e \in E$ is a non-empty subset of $V$. The incidence matrix $H \in \{0, 1\}^{|V| \times |E|}$ satisfies:
$$H_{ve} = \begin{cases} 1 & \text{if } v \in e \\ 0 & \text{otherwise} \end{cases}$$
The diagonal node degree matrix $D_v \in \mathbb{R}^{|V| \times |V|}$ and edge degree matrix $D_e \in \mathbb{R}^{|E| \times |E|}$ are:
$$(D_v)_{vv} = \sum_{e \in E} H_{ve}, \quad (D_e)_{ee} = \sum_{v \in V} H_{ve}$$
The spectral hypergraph convolution operator is:
$$X^{(l+1)} = \sigma\left(D_v^{-1} H D_e^{-1} H^T X^{(l)} W^{(l)} + b^{(l)}\right)$$

### 2. AllSet Universal Multiset Aggregation
Chien et al. proved that any permutation-invariant hypergraph layer can be factored into two set functions:
1. **Node-to-Hyperedge**:
   $$Z_e = \psi_E\left(\{X_v : v \in e\}\right) = \text{MLP}_E\left(\sum_{v \in e} \text{MLP}_V(X_v)\right)$$
2. **Hyperedge-to-Node**:
   $$X_v^{(l+1)} = X_v^{(l)} + \psi_V\left(\{Z_e : e \ni v\}\right) = X_v^{(l)} + \text{MLP}_V^{\text{out}}\left(\sum_{e \ni v} Z_e\right)$$

### 3. Exphormer Sparse Expander Attention
Dense transformers require $O(N^2)$ all-pairs attention. Exphormer defines a sparse attention graph $A_{\text{attn}} = A_{\text{local}} \cup A_{\text{exp}} \cup A_{\text{virt}}$:
- $A_{\text{local}}$: Existing 1-hop graph edges.
- $A_{\text{exp}}$: $d$-regular expander graph edges (e.g. Hamiltonian random cycles). Expander graphs have Cheeger constant $h(G) \ge \epsilon > 0$ and spectral gap $\lambda_1 - \lambda_2 \ge \gamma > 0$, guaranteeing that information diffuses across the entire graph in $O(\log N)$ hops.
- $A_{\text{virt}}$: Virtual global nodes connecting to all vertices.
Total complexity scales as $O(d_{\text{exp}} |V| + |E|) = O(|V| + |E|)$.

---

## Tensor Memory Layout & Implementation Details

### Hypergraph Message Passing
For $V$ vertices, $E$ hyperedges, and feature dimension $D$:
- Node features: $X \in \mathbb{R}^{V \times D}$
- Incidence matrix: $H \in \{0, 1\}^{V \times E}$
- Hyperedge representations: $Z_e = D_e^{-1} H^T X \in \mathbb{R}^{E \times D}$
- Node aggregation: $X_{\text{agg}} = D_v^{-1} H Z_e \in \mathbb{R}^{V \times D}$
- Feature projection: $X_{\text{next}} = \text{GELU}(X_{\text{agg}} \cdot W + b) \in \mathbb{R}^{V \times D}$

### Expander Spectral Mixing
- Expander graph adjacency: $A_{\text{exp}} \in \{0, 1\}^{V \times V}$ (2-regular cycle or Ramanujan graph)
- Expander message: $H_{\text{exp}} = A_{\text{exp}} \cdot H \in \mathbb{R}^{V \times D}$
- Output: $H_{\text{next}} = \text{GELU}((H + H_{\text{exp}}) \cdot W_{\text{mix}} + b_{\text{mix}}) \in \mathbb{R}^{V \times D}$

---

## Multi-Ecosystem Implementation Matrix

| Operation / Feature | PyTorch 2.x | JAX (vmap + jit) | Candle (Rust) | WGSL Compute Shader |
| :--- | :--- | :--- | :--- | :--- |
| **Incidence Projection** | `(H.T @ X) / d_e.T` | Vectorized `jnp.matmul` | Reshaped `t()?.matmul()` + div | Workgroup edge accumulator |
| **Hyperedge-to-Node** | `(H @ Z) / d_v` | `jnp.clip` + broadcast div | Matrix multiply + div | Node-gather thread dispatch |
| **AllSet Nonlinearity** | Alternating MLP pass | Pure functional JIT | Chained `relu()` + matmul | Fused GEMM + activation |
| **Expander Graph Mixing** | `(H + A_exp @ H) @ W` | Sparse or dense `jnp.matmul` | `matmul` + `broadcast_add` | Sparse CSR / dense cycle step |
| **Numerical Parity** | Native float32 | $< 10^{-6}$ vs golden | $< 10^{-6}$ vs golden | $< 10^{-5}$ vs golden |

---

## Pathological Misuse & Failure Modes

1. **Clique Expansion Distortion**:
   - *Anti-Pattern*: Projecting an $N$-node hyperedge into an $\binom{N}{2}$ pairwise clique graph and applying standard GCN.
   - *Consequence*: Dense cliques introduce artificial false paths, quadratically blow up edge count ($O(N^2)$), and alter the spectral graph Laplacian spectrum.
   - *Remedy*: Process the native bipartite incidence matrix $H$ directly using hypergraph convolutions or AllSet.

2. **Zero Degree Division in Sparse Hypergraphs**:
   - *Anti-Pattern*: Computing $D_v^{-1}$ or $D_e^{-1}$ without clamping un-connected isolated nodes or empty hyperedges.
   - *Consequence*: Division by zero generates `NaN` and `Inf` across all node states.
   - *Remedy*: Clamp degree vectors strictly with $\max(d, 10^{-6})$ or add self-loops.

3. **Dense Global Attention on Massive Graphs**:
   - *Anti-Pattern*: Using full $N \times N$ Graphormer attention on graphs with $N > 100,000$ nodes.
   - *Consequence*: Out-of-memory (OOM) fatal crashes due to $O(N^2)$ memory footprint.
   - *Remedy*: Replace dense all-pairs attention with sparse Expander attention (Exphormer) to retain global receptive fields in linear $O(N)$ time.

---

## Cross-Domain Generalisation Matrix

| Relational Domain | Vertex Entities $V$ | Multi-Way Hyperedges $E$ | Core Advantage |
| :--- | :--- | :--- | :--- |
| **Academic Co-Authorship & Citations** | Researchers | Multi-author research publications | Unambiguous multi-way collaboration modeling |
| **E-Commerce & Market Baskets** | Products | User transactions / shopping carts | Captures multi-item complementary purchases |
| **Cell Biology & Metabolic Networks** | Enzymes / Metabolites | Chemical reaction complexes | True stoichiometry without clique distortion |
| **Large-Scale Knowledge Graphs** | Entities | Expander Graph sparse edges (Exphormer) | Linear $O(N)$ global attention with expander mixing |

---

## Verification, Parity & Empirical Benchmarks

The multi-ecosystem implementations across PyTorch 2.x, JAX, Candle, and WGSL are validated against `weights/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.safetensors` initialized with deterministic seed 42.

- **Parity Metrics**:
  - `linear_anchor`: Parity error $< 10^{-6}$ across all runtimes.
  - `hypergraph_conv`: Exact agreement on node-to-hyperedge and hyperedge-to-node normalized aggregation.
  - `allset_multiset_step`: Precision match across alternating universal set functions.
  - `exphormer_spectral_mixing`: Exact numerical parity on expander graph message propagation.

---

## Synchronized Track Artifacts & Navigation

- **PyTorch 2.x Reference**: [`pytorch2/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.py)
- **Functional JAX Track**: [`jax/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.py)
- **Candle Engine Implementation**: [`candle/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.rs)
- **Hardware Kernels & WGSL**: [`kernels/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.rs)
- **Integration Tests**: [`crates/ai-candle/tests/05_architectures/test_38_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.rs`](file:///c:/Users/Liam/Code/AI-Examples/crates/ai-candle/tests/05_architectures/test_38_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.rs)
- **Interactive Jupyter Notebook**: [`notebooks/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.ipynb)
- **Golden Weights**: [`weights/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.safetensors)
