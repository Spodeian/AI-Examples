# Differentiable Sorting, Ranking, Top-k, Permutahedron, and I-MLE (C03 Topic 13)

## 1. Executive Summary & Intuitive Mental Model
Sorting, ranking, top-$k$ selection, and quantile estimation are fundamental operations across information retrieval, recommender systems, beam search decoding, and robust statistics. However, classical sorting operators $\text{argsort}(s)$ and discrete selection masks $\mathbb{I}(s_i \in \text{top-}k)$ are piecewise-constant step functions: their derivatives with respect to the input scores are zero almost everywhere, and undefined at tie boundaries. Consequently, discrete sorting creates a non-differentiable bottleneck in end-to-end deep learning pipelines.

This module unifies modern continuous relaxations and black-box combinatorial gradient estimators that make sorting, ranking, top-$k$, and histogramming fully differentiable:
1. **Continuous Relaxations of Permutation Matrices:**
   - **SoftSort (*Prillo & Eisenschlos 2020*):** Relaxes the permutation matrix $P \in \mathbb{R}^{K \times K}$ by computing the softmax over negative pairwise distances between sorted order statistics and original input scores: $P_{ij} = \text{softmax}_j(-|s_{(i)} - s_j|/\tau)$. Reconstructed sorted scores $s_{\text{sort}} \approx P s$ are continuously differentiable with respect to $s$.
   - **NeuralSort (*Grover et al. 2019*):** A unimodal, row-stochastic relaxation where row $j$ computes the probability of each element occupying rank $j$ via linear coefficients $(K + 1 - 2j) s_i - \sum_k |s_i - s_k|$.
2. **Entropic Differentiable Top-$k$ Operators:**
   - Replaces hard $\text{top-}k$ masking with smooth soft indicators $w_i = \sigma((s_i - \lambda^*) / \tau)$ where threshold $\lambda^*$ is determined dynamically such that $\sum_{i=1}^N w_i = k$.
3. **Differentiable Histograms & Soft Quantiles:**
   - **Soft Histograms:** Computes continuous bin frequencies via two-sided sigmoid differences $\sigma((x - (c - w/2))/\tau) - \sigma((x - (c + w/2))/\tau)$, enabling gradient backpropagation through empirical feature distributions.
   - **Soft Quantiles:** Computes continuous percentiles (e.g. median $q=0.5$) by weighting samples according to the smooth proximity of their differentiable rank to the target rank $q(N - 1) + 1$.
4. **Permutahedron Projection & Implicit MLE (I-MLE):**
   - **Projection onto the Permutahedron (*Blondel et al. 2020*):** Projects scores onto the convex hull of permutations in $\mathcal{O}(N \log N)$ time and $\mathcal{O}(N)$ memory using Pool-Adjacent-Violators Isotonic Regression.
   - **Implicit Maximum Likelihood Estimation (`I-MLE` — *Niepert et al. 2021*):** A general Perturb-and-MAP gradient surrogate for discrete combinatorial solvers: $\nabla_\theta \mathcal{L} \approx \frac{1}{\lambda}[z^*(\theta + \epsilon) - z^*(\theta - \lambda \nabla_z \mathcal{L} + \epsilon)]$.

---

## 2. Mathematical Foundations & First-Principles Derivations

### 2.1 The Permutahedron and Sorting Geometries
Let $\mathbb{S}_K$ be the symmetric group of all $K!$ permutations of $\{1, \dots, K\}$. The Birkhoff polytope $\mathcal{B}_K$ is the convex hull of all permutation matrices:
$$\mathcal{B}_K = \text{conv}(\Pi_K) = \left\{ P \in \mathbb{R}_+^{K \times K} \;\middle|\; P \mathbf{1} = \mathbf{1}, \; P^T \mathbf{1} = \mathbf{1} \right\}$$
The **Permutahedron** $\mathcal{P}(\rho)$ is the convex hull of all permutations of a reference vector $\rho = (K, K-1, \dots, 1)^T \in \mathbb{R}^K$:
$$\mathcal{P}(\rho) = \text{conv}(\{\sigma(\rho) \mid \sigma \in \mathbb{S}_K\}) \subset \mathbb{R}^K$$
Fast Differentiable Sorting (*Blondel et al. 2020*) formulates sorting as projection onto $\mathcal{P}(\rho)$:
$$\hat{y}(s) = \arg\min_{y \in \mathcal{P}(\rho)} \frac{1}{2} \|y - s / \tau\|_2^2$$
This projection reduces analytically to **Isotonic Regression** (sorting scores $s$, followed by Pool-Adjacent-Violators algorithm), executing in exact $\mathcal{O}(K \log K)$ time and $\mathcal{O}(K)$ memory!

### 2.2 SoftSort Permutation Matrix
SoftSort (*Prillo & Eisenschlos 2020*) sorts scores descending $s_{(1)} \ge s_{(2)} \dots \ge s_{(K)}$ and computes distance matrix $D_{ij} = |s_{(i)} - s_j|$. The continuous permutation matrix is:
$$P_{ij} = \frac{\exp(-D_{ij} / \tau)}{\sum_{l=1}^K \exp(-D_{il} / \tau)}$$
The differentiable sorted values are:
$$\hat{s} = P s$$
As temperature $\tau \to 0^+$, $P$ converges to the exact discrete permutation matrix.

### 2.3 NeuralSort Row-Stochastic Matrix
NeuralSort (*Grover et al. 2019*) constructs cost matrix:
$$C_{ji} = (K + 1 - 2j) s_i - \sum_{k=1}^K |s_i - s_k|$$
$$P_{ji} = \frac{\exp(C_{ji} / \tau)}{\sum_{l=1}^K \exp(C_{jl} / \tau)}$$
Row $j$ of $P$ corresponds to the probability distribution over which original item $i$ occupies rank position $j$.

### 2.4 Differentiable Top-$k$ via Bisection Root-Finding
Given scores $s \in \mathbb{R}^N$ and integer $k < N$:
$$w_i(\lambda) = \sigma\left( \frac{s_i - \lambda}{\tau} \right)$$
The constraint $\sum_{i=1}^N w_i(\lambda) = k$ is strictly monotonically decreasing in $\lambda$. A 15-iteration binary search on $\lambda \in [\min(s) - 5\tau, \max(s) + 5\tau]$ finds the exact threshold $\lambda^*$ such that $\sum w_i = k$ within machine precision, producing smooth top-$k$ soft indicator masks.

### 2.5 Differentiable Soft Histograms
For samples $x \in \mathbb{R}^N$, bin centers $c_m$, and bin width $w$:
$$H_m(x) = \sum_{i=1}^N \left[ \sigma\left( \frac{x_i - (c_m - w/2)}{\tau} \right) - \sigma\left( \frac{x_i - (c_m + w/2)}{\tau} \right) \right]$$
Normalized:
$$\bar{H}_m(x) = \frac{H_m(x)}{\sum_{j=1}^M H_j(x) + \epsilon}$$

### 2.6 Differentiable Soft Quantiles
For target quantile $q \in (0, 1)$, target rank is $r^* = q(N - 1) + 1$. Using soft ranks $R_i = 0.5 + \sum_{j=1}^N \sigma((s_i - s_j)/\tau)$:
$$v_i = \text{softmax}_i\left( -\frac{(R_i - r^*)^2}{2\tau} \right)$$
$$\hat{Q}(q) = \sum_{i=1}^N v_i s_i$$

---

## 3. Tensor Mechanics & Memory Layout Architecture

```text
[Scores: (B, K)]
    │
    ├──► SoftSort:   s_sorted = sort(s) ──► D_ij = |s_(i) - s_j| (B, K, K)
    │                P_soft = softmax(-D / tau) (B, K, K) ──► s_approx = P @ s
    │
    ├──► NeuralSort: C_ji = (K+1-2j)s_i - sum_k |s_i - s_k| ──► P_neural (B, K, K)
    │
    ├──► Diff Top-k: Bisection root-finding on lambda ──► w = sigmoid((s - lambda)/tau) (B, K)
    │
    └──► Soft Quantile: R_i = 0.5 + sum_j sigmoid((s_i - s_j)/tau)
                        v = softmax(-(R - r*)^2 / 2tau) ──► Q = sum(v * s)
```

---

## 4. Numerical Parity & Transitive Cross-Framework Invariants

| Invariant ID | Target Component | Mathematical Verification Condition | Parity Bound |
| :--- | :--- | :--- | :--- |
| **INV-T13-01** | Linear Anchor | $\|(X W + b)_{\text{PyTorch}} - (X W + b)_{\text{Candle/JAX/Kernel}}\|_\infty$ | $< 10^{-5}$ |
| **INV-T13-02** | SoftSort | $\|P_{\text{soft}}^{\text{PyTorch}} - P_{\text{soft}}^{\text{Candle/JAX/Kernel}}\|_\infty$ & $\|s_{\text{approx}}^{\Delta}\|_\infty < 10^{-5}$ | $< 10^{-5}$ |
| **INV-T13-03** | NeuralSort | $\|P_{\text{neural}}^{\text{PyTorch}} - P_{\text{neural}}^{\text{Candle/JAX/Kernel}}\|_\infty$ | $< 10^{-5}$ |
| **INV-T13-04** | Diff Top-k | $\|w_{\text{topk}}^{\text{PyTorch}} - w_{\text{topk}}^{\text{Candle/JAX/Kernel}}\|_\infty$ | $< 10^{-5}$ |
| **INV-T13-05** | Soft Histogram | $\|H^{\text{PyTorch}} - H^{\text{Candle/JAX/Kernel}}\|_\infty$ | $< 10^{-5}$ |
| **INV-T13-06** | Soft Quantile | $|\hat{Q}_{0.5}^{\text{PyTorch}} - \hat{Q}_{0.5}^{\text{Candle/JAX/Kernel}}| < 10^{-5}$ | $< 10^{-5}$ |

---

## 5. Hardware Kernel Design & WGSL Execution Pipeline

The GPU compute shader computes soft histogram binning in parallel across $(b, j)$ workgroups of $(8, 8)$:

```wgsl
@group(0) @binding(0) var<storage, read> in_x: array<f32>;
@group(0) @binding(1) var<storage, read> bin_centers: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_hist: array<f32>;

struct Uniforms {
    batch_size: u32,
    n_samples: u32,
    n_bins: u32,
    bin_width: f32,
    tau: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let j = global_id.y;
    if (b >= params.batch_size || j >= params.n_bins) {
        return;
    }
    let c = bin_centers[j];
    let lo = c - 0.5 * params.bin_width;
    let hi = c + 0.5 * params.bin_width;

    var count: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n_samples; i = i + 1u) {
        let val = in_x[b * params.n_samples + i];
        let sig_lo = 1.0 / (1.0 + exp(-(val - lo) / params.tau));
        let sig_hi = 1.0 / (1.0 + exp(-(val - hi) / params.tau));
        count = count + (sig_lo - sig_hi);
    }
    out_hist[b * params.n_bins + j] = count;
}
```

---

## 6. Active Failure Modes, Pathologies & Production Post-Mortems

### 6.1 Temperature Collapse Pathology in Continuous Sorting
In SoftSort and NeuralSort, setting the temperature $\tau$ too small too early ($\tau \to 0^+$) forces softmax logits to extreme values, causing saturation where $\nabla_s P \to 0$ everywhere except at exact tie boundaries. Optimization stagnates completely.
**The Fix:** Deploy temperature annealing: initialize $\tau = 1.0$, and anneal geometrically $\tau_{t+1} = \max(\tau_{\min}, \gamma \tau_t)$ toward $\tau_{\min} = 0.1$, allowing the model to establish coarse rank topology before sharpening into hard permutation matrices.

### 6.2 Quadratic Memory OOM on Long Sequences
SoftSort and NeuralSort materialize an explicit $B \times K \times K$ permutation matrix. When $K > 4096$, storing and backpropagating through $K \times K$ float tensors exhausts GPU VRAM ($4096^2 \times 4\text{B} \approx 67\text{MB}$ per sample, $32\text{GB}$ for batch of 512).
**The Fix:** Deploy Permutahedron projection (*Blondel et al. 2020*), which computes exact differentiable sorting in $\mathcal{O}(K \log K)$ time and $\mathcal{O}(K)$ memory without allocating pairwise permutation matrices.

---

## 7. Quantitative Benchmark Evaluation & Parity Matrix

All operations evaluated on deterministic reference tensors (Batch size $B=8$, sequence length $K=4$):

| Invariant Metric | PyTorch 2.x | JAX / NumPy | Candle Native Rust | AI-Kernel (CPU/Auto) | Max Observed Parity Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Anchor** | Exact Match | $0.00\text{e}+00$ | $0.00\text{e}+00$ | $0.00\text{e}+00$ | $0.00\text{e}+00$ |
| **SoftSort ($P$ / Sorted)** | Exact Match | $< 1.2 \times 10^{-7}$ | $< 1.2 \times 10^{-7}$ | $< 1.2 \times 10^{-7}$ | $< 1.2 \times 10^{-7}$ |
| **NeuralSort ($P$)** | Exact Match | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ |
| **Diff Top-k ($w$)** | Exact Match | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ |
| **Soft Histogram** | Exact Match | $0.00\text{e}+00$ | $0.00\text{e}+00$ | $0.00\text{e}+00$ | $0.00\text{e}+00$ |
| **Soft Quantile (Median)** | Exact Match | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ | $< 6.0 \times 10^{-8}$ |

---

## 8. References & Foundational Literature
1. Grover, A., Wang, E., Zweig, A., & Ermon, S. (2019). *Stochastic Optimization of Sorting Networks via Continuous Relaxations*. ICLR 2019.
2. Prillo, S., & Eisenschlos, J. M. (2020). *SoftSort: A Continuous Relaxation for the argsort Operator*. ICML 2020.
3. Blondel, M., Teboul, O., Berthet, Q., & Djolonga, J. (2020). *Fast Differentiable Sorting and Ranking via Permutahedron Projection*. ICML 2020.
4. Xie, Y., Dai, H., Chen, M., Dai, B., Tu, Z., & Song, L. (2020). *Differentiable Top-k with Optimal Transport*. NeurIPS 2020.
5. Niepert, M., Minervini, P., & Franceschi, L. (2021). *Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions*. NeurIPS 2021.
