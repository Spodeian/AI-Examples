# NLP Sequence Tagging: BiLSTM-CRF & Span-Based Named Entity Recognition

## 1. Evolution of Sequence Tagging: From Local Classifiers to Structured Spans

Sequence tagging assigns a categorical label to each token in a sequence (e.g., Part-of-Speech tagging, Named Entity Recognition, chunking):

1. **Local Softmax Classifiers**:
   - Classifies each token $x_t$ independently via $\operatorname{softmax}(W h_t)$.
   - **Fatal Flaw (Illegal Transitions)**: Suffers from unconstrained transition violations under BIO/BILOU tagging schemes (e.g., an `I-PER` token erroneously appearing after an `O` tag or `B-ORG` tag).

2. **Linear-Chain Conditional Random Fields (BiLSTM-CRF)** (Lample et al., 2016):
   - Combines contextual recurrent or Transformer token representations with a global transition matrix.
   - Solves the **label bias problem** via global sequence-level normalization $\frac{1}{Z(\mathbf{x})}$.

3. **Span-Based Named Entity Recognition** (Lee et al., 2017; Joshi et al., 2020):
   - Completely abandons token-level BIO sequence tagging in favor of **direct span classification**.
   - Evaluates candidate spans $s = (i, j)$ directly using boundary representations $[h_i; h_j; h_j - h_i]$, naturally accommodating **nested entities** (e.g., `[Bank of [America]]`) and discontinuous entities.

---

## 2. Linear-Chain Conditional Random Fields (CRF)

Given an input sequence $\mathbf{x} = (x_1, \dots, x_T)$ and a sequence of labels $\mathbf{y} = (y_1, \dots, y_T) \in \mathcal{Y}^T$:

The conditional probability of the sequence $\mathbf{y}$ is defined as:
$$P(\mathbf{y} \mid \mathbf{x}) = \frac{\exp\left( \text{Score}(\mathbf{x}, \mathbf{y}) \right)}{Z(\mathbf{x})},$$
where the global sequence score decomposes into emission potentials and transition potentials:
$$\text{Score}(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^T E_{t, y_t} + \sum_{t=1}^{T-1} A_{y_t, y_{t+1}}.$$
Here:
- $E_{t, k} \in \mathbb{R}$ is the emission logit of tag $k$ at time step $t$ produced by the neural encoder (BiLSTM or Transformer).
- $A_{j, k} \in \mathbb{R}$ is the learnable transition score representing the compatibility of transitioning from tag $j$ to tag $k$.

---

## 3. Forward Log-Sum-Exp Algorithm for Partition Function $\log Z$

The partition function $Z(\mathbf{x})$ normalizes over all $|\mathcal{Y}|^T$ exponential possible label paths:
$$Z(\mathbf{x}) = \sum_{\mathbf{y}' \in \mathcal{Y}^T} \exp\left( \text{Score}(\mathbf{x}, \mathbf{y}') \right).$$

Direct summation is intractable ($\mathcal{O}(K^T)$). The **Forward Algorithm** computes $\log Z(\mathbf{x})$ in polynomial time $\mathcal{O}(T K^2)$ via dynamic programming in the log-space.

### 3.1 Forward Recursion
Define the forward variable $\alpha_t(j)$ as the log-sum-exp score of all prefixes ending in state $j$ at step $t$:
$$\alpha_1(j) = E_{1, j}.$$
For each time step $t = 2, \dots, T$:
$$\alpha_t(j) = \operatorname{LogSumExp}_{i=1}^K \left( \alpha_{t-1}(i) + A_{i, j} \right) + E_{t, j}.$$

The total log-partition function is:
$$\log Z(\mathbf{x}) = \operatorname{LogSumExp}_{j=1}^K \alpha_T(j).$$

### 3.2 Negative Log-Likelihood Loss
Model parameters are updated by minimizing the sequence NLL:
$$\mathcal{L}_{\text{CRF}} = -\log P(\mathbf{y} \mid \mathbf{x}) = \log Z(\mathbf{x}) - \text{Score}(\mathbf{x}, \mathbf{y}).$$
Because $\nabla_{\theta} \log Z(\mathbf{x}) = \mathbb{E}_{\mathbf{y}' \sim P(\mathbf{y}' \mid \mathbf{x})} [\nabla_\theta \text{Score}(\mathbf{x}, \mathbf{y}')]$, training pushes the score of the true path $\mathbf{y}$ above the expectation of all competitive paths.

---

## 4. Viterbi Decoding for Global Optimal Paths

At inference time, we seek the global maximum a posteriori (MAP) label path:
$$\mathbf{y}^\star = \arg\max_{\mathbf{y} \in \mathcal{Y}^T} P(\mathbf{y} \mid \mathbf{x}) = \arg\max_{\mathbf{y} \in \mathcal{Y}^T} \text{Score}(\mathbf{x}, \mathbf{y}).$$

Replacing the $\operatorname{LogSumExp}$ operator with $\max$ yields the classical **Viterbi Algorithm**:
$$V_t(j) = \max_{i=1}^K \left( V_{t-1}(i) + A_{i, j} \right) + E_{t, j}, \quad B_t(j) = \arg\max_{i=1}^K \left( V_{t-1}(i) + A_{i, j} \right).$$
Backtracking pointers $B_t(j)$ from step $T$ to $1$ retrieves the globally optimal path $\mathbf{y}^\star$, strictly adhering to learned transition grammar.

---

## 5. The Label Bias Problem & Global Normalization

Standard Maximum Entropy Markov Models (MEMMs) normalize transition probabilities locally at every step:
$$P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^T \frac{\exp(E_{t, y_t} + A_{y_{t-1}, y_t})}{\sum_{k} \exp(E_{t, k} + A_{y_{t-1}, k})}.$$

### 5.1 Pathology of Local Normalization
In locally normalized models, states with low outgoing entropy (few transition choices) transfer almost all probability mass to their successors regardless of the emission evidence. This causes the model to ignore strong observation signals in downstream tokens (**label bias**).

CRF overcomes label bias because the partition function $Z(\mathbf{x})$ normalizes **globally across the entire sequence path**, allowing future high-confidence evidence to retroactively veto weak local decisions.

---

## 6. Span-Based Named Entity Recognition

While BIO sequence tagging assumes every token belongs to at most one entity, real-world text frequently exhibits **nested entity structures**:
$$\text{"[[Bank of America]}_{\text{ORG}} \text{ President}_{\text{TITLE}} \text{ [Brian Moynihan]}_{\text{PER}} \dots\text{"}$$

### 6.1 Span Feature Construction
Given contextual token embeddings $H = (h_1, \dots, h_T) \in \mathbb{R}^{T \times D}$, each candidate span $s = (i, j)$ with $1 \le i \le j \le \min(i + L_{\max}, T)$ is represented by:
$$g(i, j) = \left[ h_i; \ h_j; \ h_j - h_i; \ \phi(j - i + 1) \right] \in \mathbb{R}^{3D + d_w},$$
where:
- $h_i, h_j$ capture the left and right boundary semantics.
- $h_j - h_i$ encodes internal content delta.
- $\phi(w)$ is a learned span-length embedding vector.

Each candidate span is classified independently into entity types or $\varnothing$ (non-entity) via an MLP, naturally solving nested entity extraction without transition state machines.

---

## 7. Comparative Architecture Matrix

| Dimension | Token Softmax | BiLSTM-CRF | Span-Based NER | Token-Level Hypergraph |
| :--- | :--- | :--- | :--- | :--- |
| **Output Space** | Independent tokens | Valid sequence paths | Candidate span tuples | Directed hyperedge set |
| **Transition Constraints** | Unconstrained (errors) | Strictly enforced | Inherent in span limits | Graph consistency |
| **Nested Entities** | Incapable | Incapable | Native Support | Native Support |
| **Decoding Complexity** | $\mathcal{O}(T K)$ | $\mathcal{O}(T K^2)$ (Viterbi) | $\mathcal{O}(T \cdot L_{\max} \cdot K)$ | $\mathcal{O}(T^3)$ |
| **Loss Function** | Cross-Entropy | Sequence NLL $\log Z - S$ | Multi-Class Cross-Entropy | Hypergraph Margin Loss |

---

## 8. Cross-Track Verification & Systems-Level Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every sequence tagging operator adheres to strict mathematical invariants:

| Operator | Verification Metric | Tolerance | Numerical Invariant |
| :--- | :--- | :--- | :--- |
| **CRF Log Partition** | Forward LogSumExp DP over all $K^T$ paths | $< 10^{-5}$ | Upper bound: $\log Z \ge \text{Score}(\mathbf{y})$ for any $\mathbf{y}$ |
| **Gold Sequence Score** | Analytical sum of emissions and transitions | $< 10^{-5}$ | Exact linear accumulation |
| **CRF NLL Loss** | $\log Z - \text{Score}(\mathbf{y}^\star)$ | $< 10^{-5}$ | Strictly non-negative: $\mathcal{L}_{\text{CRF}} \ge 0$ |
| **Span Features** | $[h_{\text{start}}; h_{\text{end}}; h_{\text{end}} - h_{\text{start}}]$ | $< 10^{-5}$ | Boundary difference: $g[2D..3D] = g[D..2D] - g[0..D]$ |
| **Linear Anchor** | Shared cross-track regression anchor | $< 10^{-5}$ | Deterministic match against golden tensor |


