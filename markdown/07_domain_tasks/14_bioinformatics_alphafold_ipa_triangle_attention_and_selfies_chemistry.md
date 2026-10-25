# Bioinformatics: AlphaFold, Invariant Point Attention, Triangle Attention, and SELFIES Chemistry

## 1. The Structural Biology Revolution: From 1D Sequence to 3D Conformation

Proteins are linear heteropolymers composed of 20 amino acid residues that fold spontaneously into complex 3D nanomachines. Determining protein structure historically required months of synchrotron X-ray crystallography or cryogenic electron microscopy (cryo-EM). Deep learning breakthroughs (AlphaFold 2, ESMFold, AlphaFold 3, RoseTTAFold) revolutionized structural biology by predicting atomic coordinates directly from amino acid sequences with sub-Angstrom root-mean-square deviation (RMSD).

## 2. Evoformer Backbone: MSA and Pair Representations

AlphaFold models structural context through two co-evolving representations:
- **Multiple Sequence Alignment (MSA) Representation $M \in \mathbb{R}^{S \times N \times D_{\text{msa}}}$:** Encodes evolutionary conservation and correlated mutations across homologous species.
- **Residue Pair Representation $Z \in \mathbb{R}^{N \times N \times D_{\text{pair}}}$:** Acts as a continuous, spatial contact map and distance matrix between residues $i$ and $j$.
- Outer product mean projections dynamically transfer evolutionary co-mutation signals from MSA into the 2D pair track:
  $$z_{ij} \leftarrow z_{ij} + \frac{1}{S} \sum_{s=1}^S m_{si} \otimes m_{sj}$$

## 3. Triangle Attention and Triangle Multiplicative Updates

Pairwise spatial relationships between residues must obey geometric physical consistency: if residue $i$ is near $k$, and $k$ is near $j$, then residue $i$ cannot be arbitrarily distant from $j$ (the triangle inequality). Triangle updates update edge $(i, j)$ by aggregating pairwise interactions through intermediate vertex $k$:

1. **Triangle Multiplicative Update (Outgoing Edges):**
   $$a_{ik} = z_{ik} W_a + b_a, \quad b_{jk} = z_{jk} W_b + b_b, \quad g_{ij} = \sigma(z_{ij} W_g + b_g)$$
   $$S_{ij} = \sum_{k=1}^N a_{ik} \odot b_{jk}$$
   $$z'_{ij} = (g_{ij} \odot S_{ij}) W_o + b_o$$

2. **Triangle Attention:** Operates along the edges of triangle $(i, j, k)$, using $z_{ik}$ as queries and $z_{jk}$ as keys with spatial relation bias $z_{ij}$.

## 4. Invariant Point Attention (IPA): SE(3)-Equivariant Geometric Attention

The AlphaFold Structure Module maintains a rigid body frame $T_i = (R_i, \vec{t}_i) \in \text{SE}(3)$ for each residue, representing the backbone $\text{N-C}_\alpha\text{-C}$ plane. Invariant Point Attention (IPA) couples standard scalar attention with 3D spatial points:
- Queries, keys, and values project both scalar representations and 3D points into the residue's local frame:
  $$\vec{q}_{i, h, p} = R_i \vec{q}^{\text{local}}_{i, h, p} + \vec{t}_i, \quad \vec{k}_{j, h, p} = R_j \vec{k}^{\text{local}}_{j, h, p} + \vec{t}_j$$
- The attention logit includes a spatial squared Euclidean distance penalty:
  $$a_{ij}^h = \frac{1}{\sqrt{D_k}} q_i^h k_j^h + b_{ij}^h - \frac{\gamma^h}{2} \sum_{p=1}^P \| \vec{q}_{i, h, p} - \vec{k}_{j, h, p} \|_2^2$$
- This construction is strictly invariant to global rotations and translations $\text{SE}(3)$, enforcing physical geometric reasoning.

## 5. Frame Aligned Point Error (FAPE) Loss

Evaluating predicted 3D structures via global RMSD suffers from sensitivity to hinge motions and flexible domain linkers: two rigidly identical domains connected by a flexible linker can have massive global RMSD despite perfect local geometry.

Frame Aligned Point Error (FAPE) solves this by projecting all atom coordinates into every local residue frame:
- For predicted frame $T_{x, i} = (R_{x, i}, \vec{t}_{x, i})$ and atom coordinate $\vec{x}_j$:
  $$\vec{x}^{\text{local}}_{i, j} = R_{x, i}^T (\vec{x}_j - \vec{t}_{x, i})$$
- For true experimental frame $T_{y, i}$ and true atom coordinate $\vec{y}_j$:
  $$\vec{y}^{\text{local}}_{i, j} = R_{y, i}^T (\vec{y}_j - \vec{t}_{y, i})$$
- The loss penalizes clamped coordinate discrepancy averaged across all pairs $(i, j)$:
  $$\mathcal{L}_{\text{FAPE}} = \frac{1}{N^2} \sum_{i, j} \min\left( \| \vec{x}^{\text{local}}_{i, j} - \vec{y}^{\text{local}}_{i, j} \|_2, d_{\text{clamp}} \right)$$

## 6. Molecular Chemistry: SMILES vs. 100% Robust SELFIES

Small-molecule drug discovery traditionally represented chemical structures as SMILES (Simplified Molecular Input Line Entry System) strings (e.g., `CC(=O)Oc1ccccc1C(=O)O` for aspirin). SMILES models frequently output syntactically invalid strings (unclosed rings, mismatched parentheses) or chemically nonsensical structures violating atomic valences.

SELFIES (Self-Referencing Embedded Strings) reformulates chemical strings as an execution grammar:
- Every token is a state transition that dynamically adapts to available valencies.
- **100% Syntactic and Chemical Validity:** Any random permutation of SELFIES tokens translates into a physically valid chemical graph, eliminating the need to discard invalid generative samples.

## 7. Valency-Constrained Molecular Graph Optimization

In de novo molecular generative models, continuous bond adjacency matrices $M \in \mathbb{R}^{B \times N \times N}$ are trained to satisfy physical octet valency constraints ($V_{\max} = 4$ for Carbon, $3$ for Nitrogen, $2$ for Oxygen, $1$ for Hydrogen):
- Bond probabilities: $P_{ij} = \sigma(M_{ij})$ for $i \ne j$.
- Valence penalty loss:
  $$\mathcal{L}_{\text{valence}} = \sum_{i} \max\left( \sum_{j \ne i} P_{ij} - V_{\max, i}, 0 \right)^2$$
- Differentiating through this objective regularizes molecular graphs toward valid chemical configurations.

## 8. Cross-Track Verification and Parity

All structural biology and molecular chemistry operators are verified across the 7-track curriculum against seed-42 golden safetensors:
- **PyTorch 2.x:** Reference implementations of Triangle Multiplicative Update, FAPE loss with local frame projection, IPA point distance penalty, and molecular valence loss.
- **JAX:** Functional implementation using `jnp.einsum`, `jnp.clip`, and vectorized valence penalty reductions, verified to $< 10^{-5}$ error.
- **Candle:** Native Rust implementation with batched frame alignments, 3D point Euclidean distance reductions, and clamped FAPE metric calculations, passing `cargo test -p ai-candle --test 07_domain_tasks test_14`.
- **WGSL Kernel:** Fused AlphaFold IPA 3D distance penalty compute shader calculating Euclidean point biases across attention heads in parallel.
