# AI Brain Surgery: Knowledge Editing (ROME, MEMIT), Model Stitching, and Surgical Pruning

## 1. Mathematical Foundations & Lineage

Deep neural networks store vast quantities of factual knowledge and algorithmic associations within their feedforward MLP weights. When facts become outdated or models hallucinate, fine-tuning risks catastrophic forgetting. Direct weight surgery modifies specific parameters algebraically in closed form without gradient descent.

The lineage of model brain surgery and knowledge editing comprises:
1. **Model Stitching (Lenc & Vedaldi 2015; Csiszárik et al. 2021)**: Tested the modularity and functional equivalence of representations across distinct networks by training linear stitching layers connecting early layers of Model A to later layers of Model B.
2. **MEND (Mitchell et al. 2022)**: Used hypernetworks to produce localized gradient updates for fast model editing.
3. **ROME (Rank-One Model Editing — Meng et al. 2022)**: Proved that factual associations (e.g. "The Eiffel Tower is in [Paris]") act as linear associative memories in mid-layer MLP projections $W_{\text{out}}$, allowing exact surgical factual insertion via rank-one updates $W' = W + u v^T$ while preserving the second-moment covariance of existing facts.
4. **MEMIT (Mass-Editing Memory in a Transformer — Meng et al. 2023)**: Extended ROME to scale from single-fact edits to thousands of concurrent edits distributed across multiple transformer layers.
5. **Surgical Neuron Pruning**: Identified and knocked out specific spurious or hallucinating neurons via binary projection masks without modifying remaining weights.

---

## 2. Rank-One Model Editing (ROME) Formulation

Consider an MLP layer operating as a key-value associative memory: $v = W k$.
We wish to insert a new association: $k_* \mapsto v_*$, such that:
$$(W + \Delta W) k_* = v_*$$
To prevent catastrophic forgetting of existing knowledge, we minimize the distortion on prior keys drawn from distribution $k \sim \mathcal{D}$ with uncentered covariance $C = \mathbb{E}[k k^T]$:
$$\min_{\Delta W} \mathbb{E}_{k \sim \mathcal{D}} [ \| \Delta W k \|_2^2 ] \quad \text{s.t.} \quad (W + \Delta W) k_* = v_*$$

### Closed-Form Solution
Using Lagrange multipliers, the unique optimal rank-one update is:
$$\Delta W = (v_* - W k_*) \frac{k_*^T C^{-1}}{k_*^T C^{-1} k_*}$$
Multiplying by $k_*$ confirms exact retrieval:
$$(W + \Delta W) k_* = W k_* + (v_* - W k_*) \frac{k_*^T C^{-1} k_*}{k_*^T C^{-1} k_*} = v_*$$

---

## 3. Mass-Editing Memory in a Transformer (MEMIT)

When updating $M$ facts concurrently with key matrix $K_* = [k_1, \dots, k_M] \in \mathbb{R}^{D_{\text{in}} \times M}$ and target values $V_* = [v_1, \dots, v_M] \in \mathbb{R}^{D_{\text{out}} \times M}$, ROME cannot be applied greedily without interference.

MEMIT computes the joint least-squares update with residual matrix $\Delta = V_* - W K_*$:
$$\Delta W = \Delta \left( K_*^T C^{-1} K_* + \lambda I \right)^{-1} K_*^T C^{-1}$$
This distributes the knowledge update across the subspace spanned by the new keys while minimizing interference with the historical covariance $C$.

---

## 4. Model Stitching & Functional Equivalence

Model stitching assesses whether intermediate layers of two distinct models (e.g., trained with different initializations or architectures) encode compatible representations.
Given activation matrices $A \in \mathbb{R}^{N \times D_A}$ and $B \in \mathbb{R}^{N \times D_B}$ on the same validation batch:
$$S = \arg\min_S \| A S - B \|_F^2 + \lambda \|S\|_F^2 = (A^T A + \lambda I)^{-1} A^T B$$
If stitching layer $S$ achieves high task accuracy, representations in Model A and Model B are functionally isomorphic up to a linear transformation.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`rome_rank_one_edit`** | Outer product of residual and $C^{-1} k$ | Vectorized rank-1 outer product | In-place rank-1 update | `ROME_RANK_ONE_UPDATE_WGSL` |
| **`memit_mass_edit`** | `torch.linalg.solve` on key Gram | `jnp.linalg.solve` on key Gram | Multi-column linear solver | Fused mass-edit GEMM |
| **`model_stitching_layer`** | Ridge regression closed form | `jnp.linalg.solve` Gram | Block Gauss-Jordan solver | Parallel cross-model Gram |
| **`neuron_knockout`** | Binary mask broadcast multiplication | Elementwise array mask mul | Broadcast vector mul | Elementwise knockout mask |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Covariance Singularity & Knowledge Bleed**:
   - *Symptom*: Editing a single fact (e.g. "The President is [Name]") corrupts unrelated knowledge across the model.
   - *Cause*: Key covariance matrix $C$ estimated with too few samples or without sufficient diagonal damping $\lambda I$, causing $C^{-1}$ to amplify noise in directions with low sample variance.
   - *Fix*: Estimate $C$ over $\ge 100,000$ text passages and add ridge regularization $C + \lambda I$.
2. **Locality Breakdown & Memorization Leak**:
   - *Symptom*: The model regurgitates the target value $v_*$ on every prompt containing the subject, even in unrelated contexts.
   - *Cause*: Key vector $k_*$ computed without prompt prefix averaging or noise injection, causing the rank-one update to overfit to the exact prompt template.
   - *Fix*: Average key vectors across multiple paraphrased prompt prefixes with Gaussian noise augmentation.
3. **Stitching Bottleneck Collapse**:
   - *Symptom*: Stitched model accuracy drops to random chance.
   - *Cause*: Dimensionality mismatch or attempting to stitch across incompatible representational regimes (e.g. early spatial feature maps to high-level semantic tokens).
   - *Fix*: Verify CKA similarity before stitching and apply $1\times 1$ conv or affine adaptors.

---

## 7. Real-World Engineering Applications

1. **Real-Time Factual Updates**: Updating knowledge bases in deployed LLMs (e.g., updating newly elected officials, corporate mergers) in milliseconds without expensive retraining.
2. **Defamation & Privacy Erasure (Right to be Forgotten)**: Surgically erasing specific personally identifiable information (PII) from neural checkpoints.
3. **Modular Model Assembly**: Stitching a lightweight edge visual encoder to a frozen cloud language reasoning model.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `rome_delta_ref` & `rome_wnew_ref`: ROME rank-one update matrix and modified weight tensor.
3. `memit_delta_ref` & `memit_wnew_ref`: MEMIT multi-fact mass editing updates.
4. `stitch_s_ref` & `stitch_bhat_ref`: Linear model stitching matrix and stitched representations.
5. `ko_pruned_ref`: Surgically pruned weight matrix.
