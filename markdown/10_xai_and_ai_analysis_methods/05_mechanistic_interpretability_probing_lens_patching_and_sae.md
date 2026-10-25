# Mechanistic Interpretability: Linear Probing, Logit Lens, Attribution Patching, and Sparse Autoencoders

## 1. Mathematical Foundations & Lineage

Mechanistic interpretability reverse-engineers the inner circuits, computational graphs, and representational dictionaries of deep neural networks into human-interpretable algorithms.

The lineage of mechanistic circuit analysis comprises:
1. **Linear Probing (Alain & Bengio 2016; Hewitt & Liang 2019)**: Evaluated whether intermediate representations linearly encode semantic properties, contrasting against control tasks to decouple probe memorization from true representation extraction.
2. **Logit Lens & Tuned Lens (nostalgebraist 2020; Belrose et al. 2023)**: Projected intermediate residual stream activations directly through the model's final unembedding layer $W_U$, revealing that transformers iteratively refine token predictions across layers.
3. **Activation Patching / Causal Mediation (Vig et al. 2020; Meng et al. 2022)**: Pinpointed causal circuits by replacing clean activations with corrupted counterparts and measuring the change in target logits.
4. **Attribution Patching (AtP — Syed et al. 2023)**: Approximated exhaustive activation patching via first-order Taylor expansion $(a_{\text{corrupt}} - a_{\text{clean}}) \cdot \nabla_a \mathcal{L}$, reducing $O(L \cdot N)$ forward passes to a single backward pass.
5. **Sparse Autoencoders (SAE — Bricken et al. 2023; Gao et al. 2024)**: Solved polysemanticity (neurons responding to multiple unrelated concepts) by learning overcomplete sparse dictionaries using Top-K and JumpReLU thresholding (*Rajamanoharan et al. 2024*).

---

## 2. The Logit Lens Formulation

In autoregressive transformers, the residual stream state $h_l \in \mathbb{R}^D$ at layer $l$ serves as a communication channel. The Logit Lens decodes intermediate states by projecting directly through the unembedding matrix $W_U \in \mathbb{R}^{D \times V}$:
$$\text{logits}_l = h_l W_U + b_U$$
The implied vocabulary distribution at layer $l$ is:
$$p_l = \text{softmax}(\text{logits}_l)$$
This enables tracking the exact layer where a factual association or grammatical resolution is formed.

---

## 3. Activation Patching & Fast Attribution Patching (AtP)

Activation patching evaluates whether a specific hidden node or attention head $a$ is causally necessary for a behavior.
Given clean input $x_{\text{clean}}$ and corrupted input $x_{\text{corrupt}}$, exact patching replaces activation $a_{\text{clean}}$ with $a_{\text{corrupt}}$ and evaluates the shift in metric $\mathcal{M}$:
$$\Delta \mathcal{M}_{\text{exact}}(a) = \mathcal{M}(x_{\text{clean}} \mid a \leftarrow a_{\text{corrupt}}) - \mathcal{M}(x_{\text{clean}})$$

### First-Order Linear Attribution Patching
Because evaluating every node independently requires thousands of model forward passes, Attribution Patching approximates $\Delta \mathcal{M}$ via first-order Taylor expansion:
$$\text{AtP}(a) \approx (a_{\text{corrupt}} - a_{\text{clean}}) \cdot \nabla_a \mathcal{M}(x_{\text{clean}})$$
A single forward and backward pass simultaneously estimates causal importance across all layers, heads, and MLP neurons.

---

## 4. Sparse Autoencoders: Top-K & JumpReLU Dictionaries

Due to superposition, neural networks represent more features than they have dimensions by packing non-orthogonal directions into polysemantic neurons. Sparse Autoencoders (SAEs) decompose residual states $x \in \mathbb{R}^D$ into an overcomplete sparse dictionary $D_{\text{sae}} \gg D$.

### Top-K SAE
Standard $L_1$ penalties suffer from feature shrinkage bias (penalizing true feature amplitudes). Top-K SAEs replace $L_1$ with an explicit top-$k$ selection rule:
$$z = \text{ReLU}\left( (x - b_{\text{dec}}) W_{\text{enc}} + b_{\text{enc}} \right)$$
$$f_z = \text{TopK}(z, k)$$
$$\hat{x} = f_z W_{\text{dec}} + b_{\text{dec}}$$

### JumpReLU SAE
JumpReLU applies discontinuous hard thresholding above threshold $\theta$, ensuring exact zero-activation below $\theta$ with zero shrinkage on active features:
$$\text{JumpReLU}(z, \theta) = z \odot \mathbb{I}(z > \theta)$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`logit_lens`** | Matrix multiplication + bias broadcast | Array matmul + broadcast | `matmul` + `broadcast_add` | Fused GEMM unembed |
| **`attribution_patching`** | $(a_{\text{corrupt}} - a_{\text{clean}}) \odot g$ | Elementwise subtraction and mul | Vectorized difference and dot | Parallel AtP reducer |
| **`topk_sparse_autoencoder`** | `torch.topk` + `scatter_` | `jax.lax.top_k` + `f_z.at` | Sort and take-$k$ projection | Parallel top-$k$ bitonic |
| **`jumprelu_activation`** | `torch.where` hard threshold | `jnp.where` hard threshold | In-place threshold iterator | `JUMPRELU_WGSL` |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Probe Memorization vs. Representation Trap**:
   - *Symptom*: A deep probe achieves 99% accuracy on syntactic tasks from random embeddings.
   - *Cause*: High-capacity non-linear probes memorize labels rather than measuring linearly accessible model representations (*Hewitt & Liang 2019*).
   - *Fix*: Use strictly linear probes and validate against control tasks with randomized target labels.
2. **Attribution Patching Cancellation & Non-Linearity Bias**:
   - *Symptom*: Attribution patching predicts zero causal impact for an MLP layer that activation patching proves is critical.
   - *Cause*: First-order Taylor approximation ignores second-order curvature and activation saturation across large corrupted deltas.
   - *Fix*: Use AtP as an initial screening filter, then validate top candidates with exact activation patching.
3. **Dead Latent Collapse in Sparse Autoencoders**:
   - *Symptom*: 80% of dictionary latents in an SAE have zero activations across the entire training corpus.
   - *Cause*: Poor encoder bias initialization and suboptimal learning rate warmups.
   - *Fix*: Resample dead features or apply ghost gradient regularization.

---

## 7. Real-World Engineering Applications

1. **Safety Auditing & Backdoor Detection**: Isolating monosemantic dictionary features corresponding to deception, malicious code execution, or sleeper agent triggers.
2. **Model Steering via Latent Clamping**: Clamping specific sparse autoencoder features during inference to force a model into helpful, non-hallucinatory regimes without retraining.
3. **Automated Circuit Discovery**: Mapping out the minimal subgraphs responsible for complex reasoning tasks (e.g. Indirect Object Identification, induction heads).

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `ll_logits_ref`: Logit Lens intermediate vocabulary logits.
3. `atp_scores_ref`: First-order attribution patching importance vector.
4. `sae_feat_ref` & `sae_xhat_ref`: Top-K sparse autoencoder features and reconstructions.
5. `jump_out_ref`: JumpReLU thresholded feature activations.
