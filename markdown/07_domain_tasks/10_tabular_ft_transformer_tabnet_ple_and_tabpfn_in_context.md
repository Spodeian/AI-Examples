# Tabular Deep Learning: FT-Transformer, TabNet, PLE, and In-Context TabPFN

## 1. The Tabular Learning Frontier: Why Tabular Data is Unique

Tabular datasets are ubiquitous across enterprise applications (financial fraud detection, credit scoring, click-through rate prediction, and clinical risk stratification). Unlike vision and NLP—where data exhibits continuous spatial or temporal translation invariance—tabular data features:
- **Heterogeneous Feature Semantics:** Extreme mix of continuous numerical distributions (power-law, multimodal) and discrete high-cardinality categorical variables.
- **Uncorrelated Coordinates:** Arbitrary column orderings with no natural neighborhood topology.
- **Axis-Aligned Decision Boundaries:** Decision boundaries are frequently step-functions along individual feature axes, historically making Gradient Boosted Decision Trees (GBDTs like XGBoost, LightGBM, CatBoost) challenging for standard MLPs to outperform.

## 2. Feature Tokenizer and the FT-Transformer Architecture

FT-Transformer (Feature Tokenizer Transformer) adapts the Transformer architecture to tabular data by projecting every individual feature into an embedding vector of dimension $D$:

1. **Numerical Feature Tokenization:** For each numerical feature $j \in \{1, \dots, N_{\text{num}}\}$ with scalar value $x_j \in \mathbb{R}$, dedicated weight and bias vectors $W_j, b_j \in \mathbb{R}^D$ compute an affine feature token:
   $$e_{\text{num}}^{(j)} = x_j \cdot W_j + b_j \in \mathbb{R}^D$$
2. **Categorical Feature Tokenization:** Embedding table lookup for discrete category $k$:
   $$e_{\text{cat}}^{(k)} = E^{(k)}[x_k] \in \mathbb{R}^D$$
3. **Sequence Construction:** A learnable `[CLS]` token is prepended:
   $$T_0 = \left[ e_{\text{cls}}, e_{\text{num}}^{(1)}, \dots, e_{\text{num}}^{(N_{\text{num}})}, e_{\text{cat}}^{(1)}, \dots, e_{\text{cat}}^{(N_{\text{cat}})} \right] \in \mathbb{R}^{B \times (1 + N_{\text{features}}) \times D}$$

Standard multi-head self-attention then models pairwise feature interactions dynamically for each tabular row.

## 3. TabNet: Sequential Attention and Instance-Wise Feature Selection

TabNet uses sequential decision steps to mimic tree-like sparse feature selection while retaining end-to-end differentiability:
- At decision step $i$, an attentive transformer computes a sparse feature selection mask $M[i] \in [0, 1]^{B \times D_{\text{feat}}}$:
  $$M[i] = \operatorname{sparsemax}\left( P[i-1] \cdot h_i(a[i-1]) \right)$$
- **Prior Scale $P[i]$:** Accumulates historical feature usage to ensure diverse feature coverage across steps:
  $$P[i] = \prod_{j=1}^i (\gamma - M[j]) = P[i-1] \cdot (\gamma - M[i])$$
  where $\gamma \ge 1$ is a relaxation factor ($\gamma = 1$ enforces mutual exclusivity of selected features across steps).
- The masked features $M[i] \odot x$ pass through a feature transformer (shared and step-dependent GLU layers) to output split representations for decision aggregation.

## 4. Sparsemax: Sparse Probability Simplex Projections

Standard Softmax $\operatorname{softmax}(z)_i \propto \exp(z_i)$ assigns strictly positive probabilities to every coordinate, preventing exact zero attribution. Sparsemax projects input logit vectors $z \in \mathbb{R}^D$ onto the probability simplex $\Delta^{D-1} = \{p \in \mathbb{R}^D : \sum_i p_i = 1, p_i \ge 0\}$ under Euclidean distance:

$$\operatorname{sparsemax}(z) = \arg\min_{p \in \Delta^{D-1}} \| p - z \|^2$$

The closed-form solution computes a threshold $\tau(z)$:

$$\operatorname{sparsemax}(z)_i = \max(z_i - \tau(z), 0)$$

where sorting $z$ in descending order identifies $k(z) = \max \{k \in [1, D] : 1 + k z_{(k)} > \sum_{j=1}^k z_{(j)}\}$, yielding $\tau(z) = \frac{\sum_{j=1}^{k(z)} z_{(j)} - 1}{k(z)}$. This produces truly sparse feature selection masks.

## 5. Progressive Layered Extraction (PLE): Multi-Task Tabular Architectures

In industrial recommendation and tabular systems, models predict multiple simultaneous targets (e.g., Click-Through Rate and Conversion Rate). Shared Bottom and Multi-gate Mixture-of-Experts (MMoE) often suffer from "negative transfer" (improving one task hurts another).

PLE explicitly divides expert modules into task-specific experts and shared experts:
- Task $k$ routes only over its dedicated experts $\mathcal{E}_k$ and shared experts $\mathcal{S}$:
  $$\hat{y}_k = \sum_{j \in \mathcal{E}_k \cup \mathcal{S}} g_{k, j}(x) \cdot \text{Expert}_j(x)$$
- Higher extraction layers progressively extract task-specific semantics while selectively communicating via shared channels, eliminating the seesaw dilemma.

## 6. TabPFN: In-Context Learning via Synthetic Prior-Fitted Networks

TabPFN (Prior-Fitted Networks) revolutionizes tabular learning by training a massive Transformer purely on synthetic datasets generated from causal structural equation priors:
- **No Training at Inference Time:** The model accepts an entire dataset (both labeled training examples $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$ and unlabeled test queries $x_{\text{test}}$) in its context window.
- In a single forward pass ($\sim 1$ second), self-attention across training rows and columns performs Bayesian posterior inference directly.

## 7. Inductive Biases: Tree Ensembles vs. Deep Tabular Models

| Attribute | GBDTs (XGBoost, CatBoost) | FT-Transformer / TabNet |
| :--- | :--- | :--- |
| **Inductive Bias** | Axis-aligned orthogonal splits | Dense rotationally invariant embeddings |
| **Data Efficiency** | High on small $N$ ($< 10^4$) | Excels on large $N$ ($> 10^5$) & multi-modal |
| **Categorical Handling** | Native target / frequency encoding | Differentiable continuous embedding tables |
| **Continual / Multi-Task** | Difficult to fine-tune incrementally | Native gradient-based multi-task routing (PLE) |

## 8. Cross-Track Verification and Parity

All tabular algorithms and operators are validated across the 7-track curriculum against seed-42 golden safetensors:
- **PyTorch 2.x:** Reference implementations of FT-Transformer tokenization, closed-form Sparsemax projection, TabNet attentive prior updates, and PLE multi-task routing.
- **JAX:** Functional implementation utilizing `jnp.sort`, cumulative sum thresholding, and vectorized gating, verified to $< 10^{-5}$ error.
- **Candle:** Native Rust implementation with row-wise simplex projection and dynamic expert slicing, passing `cargo test -p ai-candle --test 07_domain_tasks test_10`.
- **WGSL Kernel:** Fused FT-Transformer numerical tokenization compute shader eliminating temporary memory allocations during tabular embedding.
