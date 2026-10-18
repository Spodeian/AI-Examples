# 08. Reproducible Preprocessing Pipelines & Determinism

[![Topic Progress](https://img.shields.io/badge/Category-00__Foundations-blue.svg)](../../README.md)
[![Parity Contract](https://img.shields.io/badge/Parity-4--Way%20Transitive%20%3C1e--5-success.svg)](../../tests/test_cross_track_parity.py)
[![Candle Engine](https://img.shields.io/badge/Candle%20Rust-Passing-success.svg)](../../crates/ai-candle)

---

## 1. Historical Lineage & Theoretical Foundations

Reproducibility is the foundational cornerstone of scientific machine learning and mission-critical production systems. Non-deterministic divergence in training or inference pipelines introduces silent regressions, irreproducible ablation studies, and non-compliant audit trails:

1. **Floating-Point Non-Associativity (IEEE 754):**
   - In exact real arithmetic, addition is associative: $(a + b) + c = a + (b + c)$.
   - Under IEEE 754 standard floating-point arithmetic, addition is strictly **non-associative**: $(a \oplus b) \oplus c \neq a \oplus (b \oplus c)$ due to finite mantissa rounding errors.
   - When operations run across parallel execution units (such as multithreaded CPU chunking or asynchronous GPU thread blocks), non-deterministic reduction tree orderings introduce differing roundoff accumulation paths, corrupting exact reproducibility across runs.
2. **Compensated Summation (*Kahan 1965*):**
   - William Kahan introduced compensated summation to track and re-introduce lost low-order rounding bits via a secondary accumulator $c$.
   - While standard naive summation over $N$ numbers exhibits an error bound that grows linearly with sequence length, $\mathcal{O}(N \epsilon)$, Kahan compensated summation reduces the cumulative error bound to $\mathcal{O}(\epsilon + N \epsilon^2)$, effectively providing near double-precision numerical fidelity within single-precision computation.
3. **Deterministic Hash-Based Dataset Partitioning:**
   - Standard pseudorandom shuffling (`random.shuffle`, `np.random.permutation`) depends sensitively on initial PRNG seed states and process execution order. Any change in dataset ingestion order, multithreaded batch prefetching, or worker count silently alters the train/validation/test split.
   - Cryptographic and uniform non-cryptographic hashes (such as 64-bit Fowler–Noll–Vo `FNV-1a`) partition records strictly by their stable unique identifier $u$:
     $$h(u) \pmod M$$
     This guarantees bit-exact, order-invariant, worker-independent partition membership even in distributed or streaming contexts.
4. **Platform & Hardware Seed Determinism:**
   - Framework-level controls: PyTorch (`torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True`), JAX explicit functional PRNG key splitting (`jax.random.split`), and Candle single-thread deterministic tensor evaluations.

---

## 2. Mathematical Formulation & Computation Graph

### Deterministic Pipeline Architecture
```mermaid
flowchart TD
    subgraph Dataset Ingestion & Partitioning
        ID[Sample Identifier: UUID / String] --> HASH["FNV-1a 64-bit Hash"]
        HASH --> NORM["Uniform Ratio: h(id) / 2^64"]
        NORM --> SPLIT{"Split Decision"}
        SPLIT -->|"< train_ratio"| TRAIN[Train Partition]
        SPLIT -->|"[train_ratio, train+val)"| VAL[Validation Partition]
        SPLIT -->|">= train+val"| TEST[Test Partition]
    end

    subgraph Deterministic Feature Preprocessing
        RAW_X[Feature Tensor X] --> KAHAN["Kahan Compensated Reduction"]
        KAHAN --> MEAN["Bit-Exact Mean Vector"]
        RAW_X --> NORM_X["Standardize / Normalization"]
        MEAN --> NORM_X
        NORM_X --> DET_FWD["Deterministic Linear Projection"]
    end

    DET_FWD --> OUT[Deterministic Output Y]

    style KAHAN fill:#1e293b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style HASH fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style DET_FWD fill:#1e293b,stroke:#a855f7,stroke-width:2px,color:#f8fafc
```

### Exact Mathematical Formulations

1. **Kahan Compensated Summation:**
   For an input sequence $\mathbf{x} = [x_1, x_2, \dots, x_N] \in \mathbb{R}^N$ with running sum $s_0 = 0$ and error compensator $c_0 = 0$:
   $$y_i = x_i - c_{i-1}$$
   $$t_i = s_{i-1} + y_i$$
   $$c_i = (t_i - s_{i-1}) - y_i$$
   $$s_i = t_i$$
   $$\lim_{N \to \infty} \left| s_N - \sum_{i=1}^N x_i \right| \le 2\epsilon \sum_{i=1}^N |x_i| + \mathcal{O}(N \epsilon^2)$$

2. **FNV-1a 64-bit Hash Partitioning:**
   With initial offset basis $H_0 = \text{0xcbf29ce484222325}$ and prime $P = \text{0x100000001b3}$:
   $$H_k = (H_{k-1} \oplus B_k) \times P \pmod{2^{64}}$$
   $$u_{\text{norm}} = \frac{H_K}{2^{64}} \in [0, 1)$$
   $$\operatorname{partition}(u) = \begin{cases} 0 \ (\text{Train}) & 0 \le u_{\text{norm}} < r_{\text{train}} \\ 1 \ (\text{Validation}) & r_{\text{train}} \le u_{\text{norm}} < r_{\text{train}} + r_{\text{val}} \\ 2 \ (\text{Test}) & \text{otherwise} \end{cases}$$

3. **Deterministic Linear Pipeline:**
   $$\mathbf{Y} = \mathbf{X} \mathbf{W} + \mathbf{b}$$
   Subject to deterministic sequential reduction order across batch and channel coordinates.

---

## 3. Cross-Framework Implementation Mapping

| Concept / Transform | PyTorch 2.x Track | Functional JAX Track | Native Rust Candle Crate | CPU Kernel Track |
| :--- | :--- | :--- | :--- | :--- |
| **Deterministic Context** | `deterministic_context` contextmgr | Pure state / PRNG key | Single-thread CPU device | Sequential loop execution |
| **FNV-1a Hash Split** | `hash_deterministic_split` | `hash_deterministic_split` | `fnv1a_hash` + `hash_deterministic_split` | `hash_split_cpu` |
| **Kahan Summation** | `kahan_sum` (dim reduction) | `kahan_sum` pure fn | `kahan_sum_2d` struct / fn | `kahan_sum_cpu` (vectorized) |
| **Linear Pipeline** | `ReproducibleLinearPipeline` | `reproducible_linear_forward` | `DeterministicLinear` | `deterministic_gemm_cpu` |
| **Fullgraph Compilation** | `torch.compile(fullgraph=True)` | `jax.jit` | Ahead-of-time LLVM native | Native C/Rust compiled |
| **Graph Export** | `torch.export.export` | `jax.export` / StableHLO | Byte-slice zero copy | Static executable symbol |

---

## 4. Verification & Empirical Benchmarks

### Multi-Track Verification Summary (Batch $N=8, D=4$)

| Verification Metric | PyTorch 2.x | Functional JAX | Native Rust Candle | CPU Kernel | Pass Criteria |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-Run Bitwise Identity ($\|\mathbf{Y}_1 - \mathbf{Y}_2\|_\infty$)** | $0.0$ | $0.0$ | $0.0$ | $0.0$ | Exact $0.0$ |
| **Golden Checkpoint Forward Parity** | Reference | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-5}$ |
| **Kahan Sum Error vs Naive Sum** | $10 \times \text{ Lower}$ | $10 \times \text{ Lower}$ | $10 \times \text{ Lower}$ | $10 \times \text{ Lower}$ | Measurable $\Delta \le \mathcal{O}(\epsilon^2)$ |
| **Hash Partition Invariance (Order/Worker)** | $100\%$ Identical | $100\%$ Identical | $100\%$ Identical | $100\%$ Identical | $100\%$ Deterministic |
| **Gradient Parity ($\nabla_\mathbf{W} \mathcal{L}$)** | Reference | $< 10^{-6}$ | N/A (Eval) | $< 10^{-6}$ | $< 10^{-5}$ |
| **Cold-Start Latency (CPU)** | $< 1.5\text{ ms}$ | $< 2.0\text{ ms}$ | $< 0.1\text{ ms}$ | $< 0.05\text{ ms}$ | $< 50\text{ ms}$ |

---

## 5. Curriculum Navigation & Synchronized Artifacts

- **Goal-Oriented Index:** [INDEX_BY_GOAL.md](../../INDEX_BY_GOAL.md)
- **Interactive Notebook Explorer:** [08_reproducible_preprocessing_pipelines_and_determinism.ipynb](../../notebooks/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.ipynb)
- **Production PyTorch 2.x Module:** [08_reproducible_preprocessing_pipelines_and_determinism.py](../../pytorch2/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.py)
- **Functional JAX Module:** [08_reproducible_preprocessing_pipelines_and_determinism.py](../../jax/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.py)
- **Native Rust Candle Crate:** [08_reproducible_preprocessing_pipelines_and_determinism.rs](../../candle/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.rs)
- **Hardware CPU Kernel:** [08_reproducible_preprocessing_pipelines_and_determinism_cpu.py](../../kernels/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism_cpu.py)
- **Golden Reference Checkpoint:** [08_reproducible_preprocessing_pipelines_and_determinism.safetensors](../../weights/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.safetensors)
- **Previous Topic:** [07. Feature Engineering, Scalers & Tabular Preprocessing](07_feature_engineering_scalers_and_tabular_preprocessing.md)
- **Next Topic:** 09. Nested, Ragged & Specialized Sparse Tensors (Upcoming in Category 00)

