# Visual Autoregressive Modeling: MaskGIT, VAR, and MAR

## Executive Summary & ELI5

Traditional generative modeling of visual data previously split into two divergent schools: raster-scan autoregressive transformers (like ImageGPT or DALL-E) that generate tokens sequentially from top-left to bottom-right, and continuous diffusion models that iteratively denoise image latents. Raster-scan models suffer from quadratic context scaling and unnatural spatial causality, while standard diffusion requires dozens of iterative integration passes.

**Visual Autoregressive (VAR)**, **MaskGIT**, and **Masked Autoregressive (MAR)** redefine generative vision by breaking away from raster-order tokenization:
- **MaskGIT (Masked Generative Image Transformer)**: Uses a bidirectional transformer that treats image synthesis as parallel iterative unmasking. At each generation step, all masked visual tokens are predicted simultaneously; the model preserves high-confidence tokens and re-masks ambiguous tokens according to a cosine schedule, completing full high-resolution image synthesis in just 8–16 steps.
- **VAR (Visual AutoRegressive Modeling)**: Replaces token-by-token raster scanning with **next-scale prediction**. Images are decomposed into multi-scale token grids (e.g. $1\times 1 \to 2\times 2 \to 4\times 4 \to \dots \to 16\times 16$). The transformer generates an entire spatial resolution grid conditioned on all previous coarser scales, mirroring how human artists block out silhouettes before refining intricate details.
- **MAR (Masked Autoregressive Models)**: Solves the codebook collapse and quantization error of discrete vector quantization (VQ) by operating directly on continuous VAE latent representations. It integrates continuous diffusion loss heads into an autoregressive masked transformer, generating continuous visual tokens without discrete codebooks.

---

## Mathematical Foundations & Governing Equations

### 1. MaskGIT Iterative Cosine Masking
During training, a masking ratio $\gamma \in [0, 1]$ is drawn according to a cosine schedule:
$$\gamma(r) = \cos\left(\frac{\pi}{2} r\right), \quad r \sim \mathcal{U}[0, 1]$$
Given ground-truth discrete visual tokens $Y = (y_1, \dots, y_N)$, a binary mask $M \in \{0, 1\}^N$ is constructed by masking $\lceil \gamma(r) \cdot N \rceil$ positions with an absorbing `[MASK]` token.
The network optimizes cross-entropy on masked positions:
$$\mathcal{L}_{\text{MaskGIT}}(\theta) = -\sum_{i=1}^N M_i \log p_\theta(y_i | Y \odot (1 - M) + \text{[MASK]} \odot M)$$

At test time, across $T$ iterative steps ($t = 1 \dots T$):
1. The model predicts categorical probabilities $p(y_i | Y^{(t)})$.
2. Tokens are sampled: $\hat{y}_i \sim p(y_i | Y^{(t)})$.
3. Confidence scores are computed: $c_i = \max_k p(y_i = k | Y^{(t)})$.
4. The number of tokens to mask for step $t+1$ is $n = \lceil \gamma(t/T) \cdot N \rceil$. The $n$ tokens with the lowest confidence $c_i$ are masked back out.

### 2. VAR Next-Scale Autoregressive Prediction
Instead of a 1D sequence of length $L = H \times W$, VAR structures visual representations into $K$ multi-scale token maps:
$$\mathcal{R} = (R_1, R_2, \dots, R_K)$$
where scale $k$ has resolution $h_k \times w_k$ (with $\sum_{k=1}^K h_k w_k = N$).
The joint probability factorizes as a coarse-to-fine autoregressive product:
$$p(\mathcal{R}) = \prod_{k=1}^K p(R_k | R_1, R_2, \dots, R_{k-1})$$
Within scale $R_k$, tokens are generated conditionally in parallel using bidirectional self-attention, while cross-scale attention across $R_1 \dots R_{k-1}$ is causally masked.

### 3. MAR Continuous Masked Diffusion
MAR bypasses discrete tokenization by using continuous patch embeddings $Z = (z_1, \dots, z_N) \in \mathbb{R}^{N \times D}$. For masked tokens indicated by $M_i = 1$, continuous diffusion target velocity $v_i$ is predicted:
$$\mathcal{L}_{\text{MAR}}(\theta) = \frac{\sum_{i=1}^N M_i \| v_\theta(z_i, t) - v_i^* \|^2}{\sum_{i=1}^N M_i \cdot D + \epsilon}$$

---

## Tensor Memory Layout & Implementation Details

### Multi-Scale Grid Expansion
In VAR, transitions between scale $k-1$ and scale $k$ upsample spatial resolution:
$$X_{\text{upsampled}} = \text{repeat\_interleave}(X_{\text{coarse}}, \text{repeats}=4)$$
Followed by learned scale-residual MLP projections:
$$X_{\text{fine}} = X_{\text{upsampled}} + \text{GELU}(X_{\text{upsampled}} W_{\text{scale}} + b_{\text{scale}})$$

### Mask Injection Buffer Layout
For batch size $B$, sequence length $L$, and feature dimension $D$:
- Input tokens: $X \in \mathbb{R}^{B \times L \times D}$
- Binary mask: $M \in \{0, 1\}^{B \times L \times 1}$
- Learned mask embedding: $E_{\text{mask}} \in \mathbb{R}^{1 \times 1 \times D}$
- Injected output: $X_{\text{masked}} = (1 - M) \odot X + M \odot E_{\text{mask}}$

---

## Multi-Ecosystem Implementation Matrix

| Metric / Feature | PyTorch 2.x | JAX (vmap + jit) | Candle (Rust) | WGSL Compute Shader |
| :--- | :--- | :--- | :--- | :--- |
| **Cosine Schedule** | `torch.cos(0.5 * pi * r)` | `jnp.cos(0.5 * pi * r)` | `affine` + `cos()` | Built-in `cos()` WGSL function |
| **Mask Injection** | `(1-M)*X + M*E` | Vectorized `at` / arithmetic | `broadcast_mul` + `affine` | Thread-level `select` operator |
| **Scale Expansion** | `repeat_interleave` | `jnp.repeat` along axis 1 | `repeat((1, repeats, 1))` | Strided memory index mapping |
| **Continuous Loss** | Masked sum reduction | Pure functional `jnp.sum` | `sum_all` scalar conversion | Atomic parallel float accumulator |
| **Numerical Parity** | Native float32 | $< 10^{-6}$ vs golden | $< 10^{-6}$ vs golden | $< 10^{-5}$ vs golden |

---

## Pathological Misuse & Failure Modes

1. **Linear Masking Schedule in MaskGIT**:
   - *Anti-Pattern*: Using a linear schedule $\gamma(r) = 1 - r$ instead of the concave cosine schedule $\cos(\frac{\pi}{2} r)$.
   - *Consequence*: Linear schedules de-mask too many tokens too early in generation, forcing the model to make unconditioned guesses that lock in permanent visual artifacts.
   - *Remedy*: Always use concave schedules (cosine or square root) that de-mask cautiously during early steps and accelerate as global context stabilizes.

2. **Causal Masking Within VAR Scales**:
   - *Anti-Pattern*: Imposing autoregressive causal triangular masking within the same spatial resolution grid $R_k$.
   - *Consequence*: Destroys spatial bidirectional context within the grid, re-introducing raster-scan directional bias without speed benefit.
   - *Remedy*: Enforce causal masking only between different scales ($R_1 \dots R_{k-1}$); allow full bidirectional self-attention within $R_k$.

3. **Loss Normalization Disregard in MAR**:
   - *Anti-Pattern*: Dividing masked MSE loss by the total number of sequence tokens $N$ rather than the dynamic count of actively masked tokens $\sum M$.
   - *Consequence*: Loss magnitudes fluctuate wildly depending on stochastic mask ratios, destabilizing learning rate dynamics.
   - *Remedy*: Normalize strictly by $\sum M \cdot D + \epsilon$.

---

## Cross-Domain Generalisation Matrix

| Visual Architecture | Spatial Order | Latent Representation | Generation Strategy |
| :--- | :--- | :--- | :--- |
| **Standard ImageGPT / Part-1** | 1D Raster-scan (Top-Left $\to$ Bottom-Right) | Discrete VQ tokens | Sequential next-token autoregression |
| **MaskGIT** | Bidirectional Unordered | Discrete VQ tokens | Iterative parallel decoding with cosine re-masking |
| **VAR** | Hierarchical Coarse-to-Fine | Multi-scale Discrete / Continuous | Next-scale autoregression |
| **MAR** | Bidirectional Masked Patches | Continuous VAE embeddings | Continuous diffusion score prediction on masked latents |

---

## Verification, Parity & Empirical Benchmarks

The multi-ecosystem implementations across PyTorch 2.x, JAX, Candle, and WGSL are validated against `weights/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.safetensors` initialized with deterministic seed 42.

- **Parity Metrics**:
  - `linear_anchor`: Parity error $< 10^{-6}$ across all tracks.
  - `maskgit_cosine_schedule`: Exact transcendental match against reference values.
  - `maskgit_apply_mask`: Elementwise mask vector replacement validated across CPU and WebGPU.
  - `var_coarse_to_fine_upsample`: Exact agreement on spatial repetition and GELU activation.
  - `mar_masked_diffusion_loss`: Numerically identical masked MSE reduction.

---

## Synchronized Track Artifacts & Navigation

- **PyTorch 2.x Reference**: [`pytorch2/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.py)
- **Functional JAX Track**: [`jax/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.py)
- **Candle Engine Implementation**: [`candle/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.rs)
- **Hardware Kernels & WGSL**: [`kernels/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.rs)
- **Integration Tests**: [`crates/ai-candle/tests/05_architectures/test_32_visual_autoregressive_maskgit_var_and_mar.rs`](file:///c:/Users/Liam/Code/AI-Examples/crates/ai-candle/tests/05_architectures/test_32_visual_autoregressive_maskgit_var_and_mar.rs)
- **Interactive Jupyter Notebook**: [`notebooks/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.ipynb)
- **Golden Weights**: [`weights/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.safetensors)
