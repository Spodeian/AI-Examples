# Text Generation Decoding and Structured Grammar Sampling

## 1. Architectural Role & Context

In autoregressive sequence generation, modern language models compute unnormalized logit distributions over a discrete vocabulary $\mathcal{V}$ at each step $t$:
$$z_t = \text{Model}(x_{<t}) \in \mathbb{R}^{|\mathcal{V}|}$$
The decoding algorithm dictates how continuous unnormalized logits are transformed into next-token probabilities and selected into discrete outputs. Naive argmax greedy search leads to repetitive loops, while unconstrained stochastic sampling from $p = \text{softmax}(z/T)$ causes hallucination and incoherent tail drift. Furthermore, real-world deployment frequently mandates structured outputs—such as valid JSON schemas, SQL queries, or molecular SELFIES grammar—which cannot tolerate syntactic violations.

This module unifies modern token decoding and constrained sampling strategies:
- **Multiplicative Repetition Penalties** (Keskar et al., 2019) to mitigate degeneration.
- **Pre-Softmax FSM Grammar Bitmasking** (Willard & Louf, 2023 / Outlines) guaranteeing 100% syntactically valid outputs on the first pass.
- **Temperature Scaling** to control entropy.
- **Top-$K$ and Min-$P$ Sampling** (Nguyen et al., 2024) to eliminate low-probability tail garbage dynamically.

---

## 2. Mathematical Formalism

### 2.1 Repetition Penalty
Given prompt and generated token history $\mathcal{H}_t = (x_1, \dots, x_{t-1})$ and penalty hyperparameter $\alpha_{\text{rep}} \ge 1.0$:
$$\tilde{z}_{t, i} = \begin{cases} z_{t, i} / \alpha_{\text{rep}}, & \text{if } z_{t, i} > 0 \text{ and } i \in \mathcal{H}_t \\ z_{t, i} \cdot \alpha_{\text{rep}}, & \text{if } z_{t, i} \le 0 \text{ and } i \in \mathcal{H}_t \\ z_{t, i}, & \text{otherwise} \end{cases}$$

### 2.2 Pre-Softmax Finite State Machine (FSM) Bitmasking
Let $s_t \in \mathcal{S}$ denote the current deterministic finite automaton state. The grammar defines allowed token transitions $\mathcal{A}(s_t) \subseteq \mathcal{V}$. The binary grammar mask $M_t \in \{0, 1\}^{|\mathcal{V}|}$ is applied prior to normalization:
$$z^{\text{masked}}_{t, i} = \begin{cases} \tilde{z}_{t, i}, & \text{if } i \in \mathcal{A}(s_t) \\ -\infty \ (\text{or } -10^4), & \text{if } i \notin \mathcal{A}(s_t) \end{cases}$$

### 2.3 Temperature Scaling and Top-$K$ Filtering
Logits are scaled by temperature $T > 0$:
$$\hat{z}_{t, i} = z^{\text{masked}}_{t, i} / T$$
Top-$K$ selects the set $\mathcal{K}_t$ of $K$ highest logits, assigning $-\infty$ to all $i \notin \mathcal{K}_t$.

### 2.4 Softmax and Min-$P$ Truncation
Probabilities are computed via stable softmax:
$$p_{t, i} = \frac{\exp(\hat{z}_{t, i})}{\sum_{j} \exp(\hat{z}_{t, j})}$$
Min-$P$ filtering eliminates tokens whose probability falls below a fraction $p_{\min}$ of the mode $p_{\max} = \max_j p_{t, j}$:
$$\bar{p}_{t, i} = \begin{cases} p_{t, i}, & \text{if } p_{t, i} \ge p_{\min} \cdot p_{\max} \\ 0, & \text{otherwise} \end{cases}, \quad p^{\text{final}}_{t, i} = \frac{\bar{p}_{t, i}}{\sum_j \bar{p}_{t, j}}$$

---

## 3. Algorithmic Implementation & Pipeline

The end-to-end decoding transformation pipeline executes sequentially:
1. `apply_repetition_penalty(logits, history, alpha)`: Dampens logits of repeated tokens.
2. `apply_grammar_mask(logits, grammar_mask)`: Eliminates syntactically illegal tokens via precomputed bitmasks.
3. `apply_temperature(logits, temperature)`: Adjusts distribution entropy.
4. `apply_top_k(logits, k)`: Truncates distribution support to $K$ candidates.
5. `softmax(logits)`: Computes normalized probability distribution.
6. `apply_min_p(probs, min_p)`: Dynamically cuts tail mass scaled to top-token confidence.
7. Greedy or multinomial token sampling.

---

## 4. Cross-Track Verification & Parity

All 5 implementations (PyTorch 2.x, JAX, Rust Candle, WGPU/CPU Kernel, and Jupyter Notebook) are validated against seed-42 golden tensors stored in `weights/17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling.safetensors`:
- Linear anchor $\|Y_{\text{track}} - Y_{\text{golden}}\|_\infty < 10^{-5}$.
- Exact bit-accurate match on `greedy_tokens`.
- Continuous probability parity $\|P_{\text{track}} - P_{\text{golden}}\|_\infty < 10^{-5}$ across all vocabulary bins.

---

## 5. Failure Modes & Misuse Guardrails

| Misuse Mode | Mechanism | Operational Fix |
| :--- | :--- | :--- |
| **Top-$P$ before Temperature** | Distorts cumulative mass threshold because variance alters the distribution tail shape before filtering. | Strictly apply Temperature scaling prior to Top-$K$, Top-$P$, or Softmax. |
| **Post-Hoc Regex Stripping** | Generating unconstrained text and discarding invalid JSON wastes $50\text{–}90\%$ of inference compute. | Use precomputed FSM bitmasks pre-softmax to guarantee 100% parseable outputs on pass 1. |
| **High Temperature Tail Garbage** | High temperature ($T > 1.2$) flattens tails, causing standard Top-$P$ to sample nonsensical tokens. | Use **Min-$P$** ($p_i \ge p_{\min} \cdot p_{\max}$), which dynamically shrinks allowed vocabulary when top-token confidence drops. |

---

## 6. Roofline Analysis & Hardware Performance

Decoding operations execute on small batch sizes ($B=1\text{–}32$) with vocabulary sizes $|\mathcal{V}| \in [32000, 128000]$. Because arithmetic intensity is low ($\approx 1\text{–}3$ FLOPs/Byte), separating repetition penalty, masking, and temperature into separate GPU kernel launches wastes memory bandwidth. Fusing grammar masking and temperature scaling into a single WGSL / CUDA kernel (`FUSED_GRAMMAR_MASK_TEMP_WGSL`) achieves peak memory bandwidth saturation and sub-microsecond latency.

---

## 7. Cross-Domain Transfer & Isomorphisms

| Domain | Sampling Mechanism | Target Structure |
| :--- | :--- | :--- |
| **Language & Code** | JSON / Python / SQL Grammar FSM Masking | Valid syntax trees and compiler-parseable tokens. |
| **Molecular Biology** | SELFIES Grammar FSM Masking | 100% chemically valid 3D SMILES / SELFIES molecular graphs. |
| **Robotics Control** | Safe Action Space Convex Polytope Masking | Dynamic obstacle avoidance and joint limit compliance. |

---

## 8. References & Further Reading

1. Holtzman, A., et al. (2019). *The Curious Case of Neural Text Degeneration*. ICLR.
2. Keskar, N. S., et al. (2019). *CTRL: A Conditional Transformer Language Model for Controllable Generation*. arXiv:1909.05858.
3. Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. arXiv:2307.09702.
4. Nguyen, H., et al. (2024). *Min-P Sampling: Dynamic Truncation for Autoregressive Language Models*. arXiv:2407.01082.
