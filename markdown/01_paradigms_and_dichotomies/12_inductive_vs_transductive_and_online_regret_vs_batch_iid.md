# Inductive vs. Transductive Learning and Online Regret vs. Batch IID (Topic 12)

## 1. Executive Summary & Intuitive Mental Model

In machine learning theory and systems design, two foundational dichotomies dictate how models generalise and how they process observational data over time:
1. **The Inductive vs. Transductive Dichotomy (Vapnik, 1998; Zhou et al., 2003):**
   - **Inductive Learning:** A general mapping function $f: \mathcal{X} \to \mathcal{Y}$ is inferred from training points $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$. At test time, $f$ evaluates arbitrary, previously unseen out-of-sample points $x_{\text{test}} \notin \mathcal{D}_{\text{train}}$ in isolation without retraining.
   - **Transductive Learning:** The algorithm does not construct a universal mapping function across the entire input space $\mathcal{X}$. Instead, it leverages the geometric, manifold, or graph cluster structure of the specific test set $\mathcal{D}_{\text{test}} = \{x_j\}_{j=1}^M$ directly during training (e.g., Graph Laplacian harmonic energy minimization / Label Spreading).
2. **The Batch IID vs. Online Regret Minimization Dichotomy (Zinkevich, 2003; McMahan et al., 2013):**
   - **Batch IID Learning:** Data points are assumed to be independent and identically distributed according to a stationary Borel probability measure $P(X, Y)$. Training optimizes empirical risk over static multi-epoch batches.
   - **Online Adversarial Regret Minimization:** Observations arrive sequentially $t = 1, 2, \dots, T$ in an adversarial or non-stationary stream. Rather than assuming a stationary distribution, the learner aims to minimize cumulative external regret against the best fixed comparator in hindsight. When the distribution undergoes abrupt or gradual **Concept Drift**, statistical sequential change detectors (Page-Hinkley / CUSUM) signal retraining or adaptive learning rates.

### Decision Boundary: When NOT to Use
> [!WARNING]
> - Do NOT use pure Transductive Label Spreading in low-latency production microservices that require instantaneous ($< 1\text{ ms}$) single-sample scoring, because transductive inference requires re-solving or propagating over an expanded $N + 1$ graph system. Use inductive distillation or inductive graph neural networks instead.
> - Do NOT deploy static Batch IID models in non-stationary streaming environments (e.g., fraud detection, high-frequency trading) without an active **Concept Drift Detector**; static models experience catastrophic silent performance degradation under covariate and concept shift.
> - Do NOT use standard SGD with an $L_1$ penalty in online streaming environments when memory-constrained sparsity is required; SGD oscillates around zero and rarely produces exact zeros. Use **FTRL-Proximal**, which thresholds dual accumulators $z_t$ to exact machine zeros.

---

## 2. Mathematical Foundations & First-Principles Derivations

### Transductive Learning: Graph Laplacian Harmonic Energy & Label Spreading
Let $V = L \cup U$ be a dataset with $l = |L|$ labeled points and $u = |U|$ unlabeled points ($n = l + u$).
Let $W \in \mathbb{R}^{n \times n}$ be a symmetric affinity matrix:
$$W_{ij} = \exp\left(-\frac{\|x_i - x_j\|_2^2}{2\sigma^2}\right) \quad (i \ne j), \quad W_{ii} = 0$$
Let $D = \text{diag}(d_1, \dots, d_n)$ with $d_i = \sum_j W_{ij}$. The normalized graph Laplacian is:
$$S = D^{-1/2} W D^{-1/2}$$
Let $Y \in \mathbb{R}^{n \times C}$ be the clamped label indicator matrix ($Y_{ic} = 1$ if node $i \in L$ has class $c$, and $0$ otherwise).
The **Label Spreading** objective (Zhou et al., 2003) minimizes smoothness on the graph manifold with a fidelity regularizer:
$$\mathcal{E}(F) = \frac{1}{2} \left[ \sum_{i,j=1}^n W_{ij} \left\| \frac{F_i}{\sqrt{d_i}} - \frac{F_j}{\sqrt{d_j}} \right\|_2^2 + \mu \sum_{i=1}^n \|F_i - Y_i\|_2^2 \right]$$
Setting $\alpha = \frac{1}{1 + \mu} \in (0, 1)$, the iterative propagation scheme:
$$F^{(t+1)} = \alpha S F^{(t)} + (1 - \alpha) Y$$
converges geometrically as $t \to \infty$ to the closed-form harmonic solution:
$$F^* = (I_n - \alpha S)^{-1} Y$$
Probabilities are extracted via row-wise softmax:
$$P(y_i = c | x_i) = \frac{\exp(F^*_{ic})}{\sum_{k=1}^C \exp(F^*_{ik})}$$

---

### Inductive Learning: Linear Ridge Regression
Given strictly labeled points $X_L \in \mathbb{R}^{l \times d}$ and continuous labels $y_L \in \mathbb{R}^l$:
$$w_{\text{ind}} = \arg\min_w \|X_L w - y_L\|_2^2 + \lambda \|w\|_2^2 = (X_L^\top X_L + \lambda I_d)^{-1} X_L^\top y_L$$
For any new, out-of-sample point $x_{\text{new}} \in \mathbb{R}^d$:
$$\hat{y}_{\text{new}} = x_{\text{new}}^\top w_{\text{ind}}$$
Evaluation requires $\mathcal{O}(d)$ operations and does not require recomputing graph affinities or inverting $(n + 1) \times (n + 1)$ systems.

---

### Online Regret Minimization: Follow-The-Regularized-Leader (FTRL-Proximal)
In online learning, at each step $t = 1, \dots, T$, the learner selects $w_t \in \mathbb{R}^d$, incurs convex loss $f_t(w_t)$, receives gradient $g_t = \nabla f_t(w_t)$, and updates weights.
Cumulative external regret against any static benchmark $w^*$ is:
$$\text{Regret}_T(w^*) = \sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(w^*)$$
Under Online Mirror Descent / FTRL, $\text{Regret}_T = \mathcal{O}(\sqrt{T})$.

**FTRL-Proximal (McMahan et al., 2013)** solves the proximal optimization at each step:
$$w_{t+1} = \arg\min_w \left[ \sum_{s=1}^t g_s^\top w + \lambda_1 \|w\|_1 + \frac{1}{2}\lambda_2 \|w\|_2^2 + \sum_{s=1}^t \frac{1}{2}\sigma_s \|w - w_s\|_2^2 \right]$$
where $\sigma_{s,i} = \frac{1}{\alpha}(\sqrt{n_{s,i} + g_{s,i}^2} - \sqrt{n_{s,i}})$.
Tracking dual gradient accumulator $z_t = \sum_{s=1}^t g_s - \sum_{s=1}^t \sigma_s w_s$ and squared gradient sum $n_t = \sum_{s=1}^t g_s^2$, the coordinate-wise update has closed-form soft-thresholding:
$$w_{t+1, i} = \begin{cases} 0 & \text{if } |z_{t, i}| \le \lambda_1 \\ -\left(\frac{\beta + \sqrt{n_{t, i}}}{\alpha} + \lambda_2\right)^{-1} \left(z_{t, i} - \text{sign}(z_{t, i})\lambda_1\right) & \text{otherwise} \end{cases}$$
This produces **exact structural zeros** natively on sparse streaming data.

---

### Concept Drift Detection: Page-Hinkley / CUSUM
Given a sequential prediction error stream $e_1, e_2, \dots, e_t$:
Let $\bar{e}$ be the estimated baseline error mean and $\delta$ a tolerance slack parameter.
The cumulative sum is:
$$S_t = \sum_{s=1}^t (e_s - \bar{e} - \delta), \quad m_t = \min_{1 \le s \le t} S_s$$
The **Page-Hinkley statistic** is:
$$PH_t = S_t - m_t$$
When $PH_t > \lambda_{\text{threshold}}$, a concept drift is flagged, triggering model adaptation or buffer reset.

---

## 3. Tensor Mechanics & Memory Layout Architecture

| Tensor Name | Shape | Dtype | Contiguity | Semantic Role |
| :--- | :--- | :--- | :--- | :--- |
| `node_coords` | `(8, 2)` | `f32` | C-Contiguous | 2D spatial coordinates for $N=8$ graph nodes |
| `affinity_w` | `(8, 8)` | `f32` | C-Contiguous | Dense Gaussian affinity matrix $W$ |
| `laplacian_s` | `(8, 8)` | `f32` | C-Contiguous | Normalized graph Laplacian $S = D^{-1/2}WD^{-1/2}$ |
| `y_init` | `(8, 2)` | `f32` | C-Contiguous | Initial clamped label matrix ($C=2$) |
| `f_star` | `(8, 2)` | `f32` | C-Contiguous | Harmonic solution $(I - \alpha S)^{-1}Y$ |
| `f_probs` | `(8, 2)` | `f32` | C-Contiguous | Softmax normalized class probabilities |
| `transductive_preds` | `(8,)` | `f32` | C-Contiguous | Argmax transductive class assignments |
| `w_ind` | `(2,)` | `f32` | C-Contiguous | Inductive ridge regression weight vector |
| `inductive_scores` | `(4,)` | `f32` | C-Contiguous | Out-of-sample continuous decision scores |
| `inductive_preds` | `(4,)` | `f32` | C-Contiguous | Inductive binary class predictions |
| `grad_stream` | `(4, 4)` | `f32` | C-Contiguous | Streaming gradient vectors ($T=4, D=4$) |
| `final_w_ftrl` | `(4,)` | `f32` | C-Contiguous | Sparse weight vector produced by FTRL-Proximal |
| `z_accum` | `(4,)` | `f32` | C-Contiguous | FTRL dual gradient accumulator |
| `n_accum` | `(4,)` | `f32` | C-Contiguous | FTRL squared gradient accumulator |
| `stream_errors` | `(16,)` | `f32` | C-Contiguous | Streaming loss/error sequence ($T=16$) |
| `cusum_s` | `(16,)` | `f32` | C-Contiguous | Cumulative sum sequence $S_t$ |
| `ph_stat` | `(16,)` | `f32` | C-Contiguous | Page-Hinkley test statistic $S_t - m_t$ |
| `drift_flags` | `(16,)` | `f32` | C-Contiguous | Binary drift indicator flags ($PH_t > \lambda$) |

---

## 4. Multi-Ecosystem Implementation Comparison Matrix

| Feature / Property | PyTorch 2.x (`pytorch2/`) | JAX Functional (`jax/`) | Rust Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Matrix Inversion** | `torch.linalg.inv` | `jnp.linalg.inv` | Gauss-Jordan Elimination | Host / GPU Solver |
| **Graph Laplacian** | Broadcast matmul | Pure functional `jnp.dot` | Vectorized tensor matmul | Blocked parallel reduction |
| **FTRL Coordinate Loop** | Vectorized loop + sign | Pure state-passing list scan | Sequential mutable vector scan | Parallel WGSL coordinate shader |
| **Exact L1 Zeroing** | Exact threshold clamping | Exact conditional branch | Hard `0.0f32` assignment | GPU `select(0.0, ...)` logic |
| **CUSUM Running Min** | Cumulative min tracking | Pure list scan comprehension | Vector running accumulator | Fast streaming loop |
| **Parity Tolerance** | $< 10^{-5}$ max-norm | $< 10^{-5}$ max-norm | $< 10^{-5}$ max-norm | $< 10^{-5}$ max-norm |

---

## 5. Pathological Failure Modes & Misuse Patterns

1. **Deploying Transductive Models for Online Scoring:**
   - *Failure:* Adding single new incoming samples to a graph and re-inverting $(N+1) \times (N+1)$ laplacians at inference time, causing latency to spike from $0.1\text{ ms}$ to $> 500\text{ ms}$.
   - *Fix:* Use transductive label spreading strictly for offline semi-supervised pseudo-labeling, then train an inductive model on the pseudo-labels.
2. **Standard SGD for Sparse Streaming Data:**
   - *Failure:* Applying SGD with $L_1$ penalty to streaming high-cardinality features; weights fluctuate around $10^{-6}$ but never reach exact zero, consuming gigabytes of memory.
   - *Fix:* Use **FTRL-Proximal**, which provably sets inactive coordinates to exact `0.0`.
3. **Ignoring Concept Drift in Batch Models:**
   - *Failure:* Deploying a static model on a non-stationary stream where feature distributions or optimal decision boundaries drift.
   - *Fix:* Pair streaming models with Page-Hinkley / CUSUM change detectors to trigger adaptive learning rates or model retraining.
4. **Disconnected Graph Components in Label Spreading:**
   - *Failure:* Using a Gaussian affinity scale $\sigma$ that is too small, resulting in isolated nodes with degree $d_i = 0$ and `NaN` in $D^{-1/2}$.
   - *Fix:* Use adaptive $k$-nearest-neighbor graph construction and verify positive node degrees.
5. **Learning Rate Divergence in FTRL:**
   - *Failure:* Setting $\alpha_{\text{ftrl}}$ too high, causing accumulated dual gradients to explode before the denominator $\sqrt{n_i}$ can stabilize the step size.
   - *Fix:* Ensure $\alpha_{\text{ftrl}} \in [0.01, 0.1]$ and $\beta_{\text{ftrl}} \ge 1.0$.

---

## 6. Cross-Domain Generalisation & Scientific Isomorphisms

- **Vision & Pathology:** Transductive graph propagation spreads rare histological annotations across whole-slide tissue patches.
- **Natural Language Processing & Search:** FTRL-Proximal trains billion-parameter click-through rate (CTR) prediction models with 95% exact parameter sparsity.
- **Quantitative Finance:** Page-Hinkley CUSUM detects macro-economic regime shifts and volatility breaks in millisecond tick data.
- **Cybersecurity & Network Telemetry:** Streaming online learning adapts immediately to novel DDoS attack vectors without requiring full-dataset retraining.
- **Bioinformatics:** Transductive semi-supervised learning infers protein functional categories from protein-protein interaction (PPI) networks.
- **Robotics & Autonomous Driving:** Detects sudden sensor degradation or road surface transition (dry asphalt to ice) via sequential innovation CUSUM.
- **Recommender Systems:** Real-time online user preference updates via FTRL without storing raw interaction histories.
- **Industrial IoT:** Monitors vibration sensor drift in jet turbines, flagging early bearing wear before catastrophic failure.

---

## 7. Verification, Parity & Benchmarking Protocol

- **Golden Reference Checkpoint:** `weights/01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.safetensors`
- **Tolerances:** Absolute difference $\|Y - Y_{\text{ref}}\|_\infty < 10^{-5}$ across all tracks.
- **Performance Characteristics:**
  - Transductive Inversion: $\mathcal{O}(N^3)$ offline; prediction $\mathcal{O}(NC)$.
  - Inductive Prediction: $\mathcal{O}(D)$ per sample ($< 10\ \mu\text{s}$).
  - FTRL Step: $\mathcal{O}(D)$ per streaming record.
  - CUSUM Drift Check: $\mathcal{O}(1)$ streaming update.

---

## 8. Navigation & Synchronized Track Artifacts

- [Interactive Concept Explorer Notebook](../../notebooks/01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.ipynb)
- [Production PyTorch 2.x Module](../../pytorch2/01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.py)
- [Functional JAX Module](../../jax/01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.py)
- [Native Rust Candle Module](../../candle/01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.rs)
- [Hardware Native Kernel](../../kernels/01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.rs)
- [Golden SafeTensors Weights](../../weights/01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.safetensors)
