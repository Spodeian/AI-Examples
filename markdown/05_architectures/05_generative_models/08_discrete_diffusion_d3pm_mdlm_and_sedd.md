# Discrete Diffusion: D3PM, MDLM, and SEDD

## Executive Summary & ELI5

Continuous diffusion models excel at synthesizing real-valued signals such as continuous image pixels and audio waveforms by adding Gaussian noise and solving reverse-time stochastic or ordinary differential equations. However, applying Gaussian diffusion directly to discrete domains (e.g. text tokens, code characters, graph nodes, genetic sequences) produces severe quantization degradation when mapping continuous embeddings back to discrete vocabularies.

**Discrete Diffusion** models reformulate generative diffusion natively over discrete categorical state spaces $\{1, 2, \dots, K\}$ without projecting into intermediate unconstrained continuous spaces:
- **D3PM (Discrete Denoising Diffusion Probabilistic Models)**: Replaces Brownian motion with discrete Markov transition matrices $Q_t$. Tokens transition between vocabulary symbols either uniformly or into specialized absorbing `[MASK]` states.
- **MDLM (Masked Diffusion Language Models)**: Bridges masked language models (BERT) and continuous-time diffusion by formulating unmasking as continuous-time jump processes, achieving competitive scaling and faster non-autoregressive decoding for long text sequences.
- **SEDD (Score Entropy Discrete Diffusion)**: Extends score-based generative modeling to discrete state spaces through continuous-time Markov Jump Processes (MJPs) and Concrete score matching, optimizing discrete Fisher divergence (score entropy) without requiring tractably closed-form posterior distributions.

---

## Mathematical Foundations & Governing Equations

### 1. D3PM Markov Transition Dynamics
Let $x_t \in \{1, \dots, K\}$ be a categorical random variable represented as a one-hot vector $x_t \in \{0, 1\}^K$. The forward Markov chain transitions with matrix $Q_t \in [0, 1]^{K \times K}$ where $Q_t(i, j) = q(x_t = j | x_{t-1} = i)$:
$$q(x_t | x_{t-1}) = \text{Cat}(x_t; x_{t-1} Q_t)$$
By Chapman-Kolmogorov composition, the $t$-step forward marginal is given in closed form by:
$$q(x_t | x_0) = \text{Cat}(x_t; x_0 \bar{Q}_t), \quad \text{where } \bar{Q}_t = Q_1 Q_2 \cdots Q_t$$

Key transition matrices include:
1. **Uniform Noise**:
   $$Q_t = (1 - \beta_t) I + \frac{\beta_t}{K} \mathbf{1} \mathbf{1}^T$$
2. **Absorbing State (`[MASK]`)**:
   $$Q_t(i, j) = (1 - \beta_t) \delta_{ij} + \beta_t \delta_{j, \text{MASK}}$$

### 2. MDLM Continuous-Time Masked Diffusion
In continuous time $t \in [0, 1]$, the retention probability $\alpha_t \in [0, 1]$ monotonically decreases from $\alpha_0 = 1$ to $\alpha_1 = 0$. The forward marginal distribution for token $x_t$ given clean token $x_0$ satisfies:
$$q(x_t | x_0) = \alpha_t x_0 + (1 - \alpha_t) e_{\text{MASK}}$$
where $e_{\text{MASK}}$ is the one-hot indicator for the absorbing token. The denoising neural network $p_\theta(x_0 | x_t, t)$ predicts categorical logits over the vocabulary for all masked positions in parallel.

### 3. SEDD Score Entropy and Markov Jump Processes
SEDD defines diffusion on discrete states via continuous-time Markov Jump Processes governed by infinitesimal rate matrices $R_t \in \mathbb{R}^{K \times K}$:
$$\frac{d}{dt} p_t(x) = \sum_{y \neq x} \left[ p_t(y) R_t(y, x) - p_t(x) R_t(x, y) \right]$$
The score is defined as the concrete probability ratio $s(x, t)_y = \frac{p_t(y)}{p_t(x)}$. The score entropy loss regresses the predicted ratio $s_\theta(x, t)$ against the empirical transition gradient:
$$\mathcal{L}_{\text{SEDD}}(\theta) = \mathbb{E}_{t, x} \left[ \sum_{y \neq x} R_t(x, y) \left( s_\theta(x, t)_y - \frac{p_t(y)}{p_t(x)} \right)^2 \right]$$

---

## Tensor Memory Layout & Implementation Details

### Discrete Denoiser Architecture
The neural denoiser processes categorical token distributions $x_t \in \mathbb{R}^{B \times L \times K}$ and continuous time scalars $t \in \mathbb{R}^{B \times 1}$:
1. **Time Conditioning Projection**:
   $$t_{\text{emb}} = \text{SiLU}(t \cdot W_t + b_t), \quad t_{\text{emb}} \in \mathbb{R}^{B \times 1 \times D_h}$$
2. **Token Projection & Modulation**:
   $$h = \text{SiLU}(x_t \cdot W_x + b_x + t_{\text{emb}}), \quad h \in \mathbb{R}^{B \times L \times D_h}$$
3. **Logits Prediction**:
   $$\text{logits} = h \cdot W_{\text{out}} + b_{\text{out}}, \quad \text{logits} \in \mathbb{R}^{B \times L \times K}$$
4. **Concrete Score Ratio**:
   $$s_\theta(x, t) = \text{Softmax}(\text{logits}, \text{dim}=-1)$$

---

## Multi-Ecosystem Implementation Matrix

| Metric / Operation | PyTorch 2.x | JAX (vmap + jit) | Candle (Rust) | WGSL Compute Shader |
| :--- | :--- | :--- | :--- | :--- |
| **Categorical Forward Marginal** | `x_0 @ q_bar` batched bmm | Vectorized `jnp.matmul` | Reshaped `x.matmul(q_bar)` | Workgroup per token |
| **Masked State Insertion** | In-place slicing or addition | Pure `at[:, :, mask_idx].set` | Buffer indexing / broadcast | Select operator on thread index |
| **Score Entropy Loss** | `torch.mean((s - s*)^2)` | `jnp.mean` reduction | `sqr()?.mean_all()?` | Global atomic reduction |
| **Sampling Mechanism** | Discrete categorical sampling | `jax.random.categorical` | Inverse CDF / Gumbel | Local PRNG XORWOW kernel |
| **Cross-Platform Precision** | Exact float32 match | $< 10^{-6}$ vs golden | $< 10^{-6}$ vs golden | $< 10^{-5}$ vs golden |

---

## Pathological Misuse & Failure Modes

1. **Continuous Embedding Space Diffusion with Nearest Neighbor Rounding**:
   - *Anti-Pattern*: Diffusing continuous embeddings $e(x) \sim \mathcal{N}(0, I)$ and taking the nearest vocabulary token via cosine similarity or Euclidean distance during sampling.
   - *Consequence*: "Embedding space collapse" where high-dimensional Gaussian spheres do not match discrete semantic token manifolds, generating severe semantic incoherence and grammar collapse.
   - *Remedy*: Use native discrete Markov transitions (D3PM/MDLM) or discrete score matching (SEDD).

2. **Absorbing State Leakage**:
   - *Anti-Pattern*: Allowing transitions out of the `[MASK]` absorbing state during forward corruption ($Q_{t}(\text{MASK}, j) > 0$ for $j \neq \text{MASK}$).
   - *Consequence*: Violates the absorbing boundary property, preventing closed-form continuous-time marginal evaluation $\alpha_t$.
   - *Remedy*: Guarantee that $Q_t(\text{MASK}, \cdot) = \delta_{\text{MASK}}$ holds strictly across all diffusion timesteps.

3. **Uniform Noise in Massive Vocabularies**:
   - *Anti-Pattern*: Using uniform transition matrices $Q = (1-\beta)I + \frac{\beta}{K}\mathbf{1}\mathbf{1}^T$ with large vocabularies ($K > 50,000$).
   - *Consequence*: With high $K$, uniform corruption places virtually all probability mass onto rare tokens, destroying semantic gradients.
   - *Remedy*: Use absorbing state masking (MDLM) or frequency-adjusted transition rates.

---

## Cross-Domain Generalisation Matrix

| Application Domain | State Space $K$ | Transition Process | Generation Capability |
| :--- | :--- | :--- | :--- |
| **Language Modeling & Code Generation** | Vocabulary tokens ($\approx 32\text{k}-100\text{k}$) | Absorbing Masking (MDLM) | Non-autoregressive parallel text infilling |
| **Protein Design & Molecular Biology** | 20 natural amino acids | BLOSUM-weighted transition rate | Functional protein sequence design |
| **Discrete Graph Structure Generation** | Adjacency matrix $\{0, 1\}$ | Uniform bit-flip / edge addition | Molecular graph synthesis |
| **Symbolic Music Generation** | Pitch, velocity, and duration tokens | Hierarchical absorbing diffusion | Polyphonic MIDI sequence generation |

---

## Verification, Parity & Empirical Benchmarks

All operations across PyTorch 2.x, JAX, Candle, and WGSL are validated against `weights/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.safetensors` initialized with deterministic seed 42.

- **Parity Verification**:
  - `linear_anchor`: Parity error $< 10^{-6}$ across CPU and GPU.
  - `d3pm_forward_marginal`: Exact categorical probability distribution alignment.
  - `mdlm_continuous_marginal`: Analytical identity matching across masked token positions.
  - `discrete_denoiser`: Exact parity for time-modulated categorical logit projections.
  - `sedd_score_entropy`: Loss tolerance $< 10^{-6}$ across all execution backends.

---

## Synchronized Track Artifacts & Navigation

- **PyTorch 2.x Reference**: [`pytorch2/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.py)
- **Functional JAX Track**: [`jax/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.py)
- **Candle Engine Implementation**: [`candle/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.rs)
- **Hardware Kernels & WGSL**: [`kernels/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.rs)
- **Integration Tests**: [`crates/ai-candle/tests/05_architectures/test_31_discrete_diffusion_d3pm_mdlm_and_sedd.rs`](file:///c:/Users/Liam/Code/AI-Examples/crates/ai-candle/tests/05_architectures/test_31_discrete_diffusion_d3pm_mdlm_and_sedd.rs)
- **Interactive Jupyter Notebook**: [`notebooks/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.ipynb)
- **Golden Weights**: [`weights/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.safetensors)
