# Large Batch and Layerwise Optimisers (Topic 03)

## 1. Executive Summary & Intuitive Mental Model
- **Core Problem & Intuition:** Distributed scaling across thousands of accelerator cores (GPUs/TPUs) requires increasing the mini-batch size $B$ from hundreds to tens of thousands ($B \ge 32\text{k}$) to saturate compute pipelines and amortize communication overhead. However, naive batch scaling causes catastrophic optimization breakdown: applying the standard Linear Scaling Rule ($\eta \propto B$) causes gradients in certain layers to explode while others vanish, leading to training divergence or generalization collapse. Layer-wise optimisers solve this dilemma by decoupling the global learning rate schedule from per-layer update dynamics:
  1. **LARS (Layer-wise Adaptive Rate Scaling):** Computes a per-layer trust ratio $\phi_l = \frac{\|w_l\|_2}{\|g_l\|_2 + \lambda \|w_l\|_2 + \epsilon}$, scaling SGD momentum updates so that no layer takes a parameter step disproportionate to its existing weight magnitude.
  2. **LAMB (Layer-wise Adaptive Moments for Batch training):** Extends layerwise trust scaling to Adam-style adaptive second moments, computing the layerwise ratio $\phi_l = \frac{\|w_l\|_2}{\|r_l\|_2}$ on the preconditioned Adam update $r_l$, enabling stable BERT pretraining with batch size $B=65,536$.
  3. **Gradient Noise Scale ($B_{\text{simple}}$):** Formulates an exact statistical diagnostic $\frac{\text{Tr}(\Sigma)}{\|G\|^2}$ that predicts the critical batch size $B_{\text{crit}}$ beyond which increasing data parallelism yields diminishing returns.
- **Historical Lineage & Seminal Citations:**
  - *Goyal et al. (2017)*: **Linear Scaling Rule & Warmup**, demonstrating ImageNet training in 1 hour with $B=8,192$ using linear learning rate scaling and gradual warmup.
  - *You, Gitman, Ginsburg (2017)*: **LARS**, scaling ImageNet ResNet-50 training to batch size $B=32,768$ without loss of validation accuracy.
  - *McCandlish et al. (2018)*: **An Empirical Model of Large-Batch Training**, introducing the Gradient Noise Scale and theoretical limits of data-parallel scaling.
  - *You et al. (2019)*: **LAMB**, scaling BERT pretraining from 3 days to 76 minutes on 1,024 TPU v3 cores with batch size $B=65,536$.
- **Decision Boundary:**
  - *When to use LARS:* Large-batch distributed training ($B \ge 8\text{k}$) of Convolutional Networks, vision backbones, and architectures trained with SGD with momentum.
  - *When to use LAMB:* Large-batch distributed pretraining ($B \ge 16\text{k}$) of Transformer language models, self-attention architectures, and ViTs.
  - *When to monitor Gradient Noise Scale:* Continuously throughout training to dynamically schedule batch size growth and determine the empirical ceiling on cluster data-parallel efficiency.

## 2. Mathematical Foundations & First-Principles Derivations
- **The Linear Scaling Rule and Its Layerwise Failure:**
  When batch size scales by a factor of $k$ ($B' = k B$), an equivalent number of gradient steps can be taken by scaling the learning rate by $k$ ($\eta' = k \eta$).
  However, in deep neural networks, the ratio of parameter norm to gradient norm $\frac{\|w_l\|_2}{\|g_l\|_2}$ varies by $3\text{--}4$ orders of magnitude across layers. Under large $\eta$, layers with small $\|w_l\| / \|g_l\|$ take updates that blow up the weight norm ($\Delta w_l \gg w_l$), while layers with large $\|w_l\| / \|g_l\|$ barely move.
- **LARS (Layer-wise Adaptive Rate Scaling):**
  For each layer $l$ with weight tensor $w_l$ and gradient $g_l$:
  $$\phi_l = \gamma \cdot \frac{\|w_l\|_2}{\|g_l\|_2 + \lambda \|w_l\|_2 + \epsilon}$$
  $$\text{local\_lr}_l = \eta \cdot \phi_l$$
  $$v_{t+1}^{(l)} = \mu v_t^{(l)} + \text{local\_lr}_l \cdot \left(g_t^{(l)} + \lambda w_t^{(l)}\right)$$
  $$w_{t+1}^{(l)} = w_t^{(l)} - v_{t+1}^{(l)}$$
  Where $\gamma \approx 10^{-3}$ is the trust coefficient and $\mu$ is momentum (typically $0.9$).
  *Layer Exclusion:* 1D parameters (biases and normalization gains/biases) are excluded from trust ratio scaling ($\phi_l = 1.0$) and weight decay ($\lambda = 0$).
- **LAMB (Layer-wise Adaptive Moments for Batch training):**
  For each layer $l$:
  1. Standard Adam first and second moments:
     $$m_t^{(l)} = \beta_1 m_{t-1}^{(l)} + (1 - \beta_1) g_t^{(l)}$$
     $$v_t^{(l)} = \beta_2 v_{t-1}^{(l)} + (1 - \beta_2) \left(g_t^{(l)} \odot g_t^{(l)}\right)$$
  2. Bias correction:
     $$\hat{m}_t^{(l)} = \frac{m_t^{(l)}}{1 - \beta_1^t}, \quad \hat{v}_t^{(l)} = \frac{v_t^{(l)}}{1 - \beta_2^t}$$
  3. Preconditioned Adam step with decoupled weight decay:
     $$r_t^{(l)} = \frac{\hat{m}_t^{(l)}}{\sqrt{\hat{v}_t^{(l)}} + \epsilon} + \lambda w_t^{(l)}$$
  4. Layer-wise Trust Ratio & Normalized Parameter Update:
     $$\phi_l = \begin{cases} \frac{\|w_t^{(l)}\|_2}{\|r_t^{(l)}\|_2} & \text{if } \|w_t^{(l)}\|_2 > 0 \text{ and } \|r_t^{(l)}\|_2 > 0 \\ 1.0 & \text{otherwise} \end{cases}$$
     $$w_{t+1}^{(l)} = w_t^{(l)} - \eta \cdot \phi_l \cdot r_t^{(l)}$$
- **Gradient Noise Scale ($B_{\text{simple}}$):**
  Let $g_i$ denote the true gradient of sample $i$.
  The true gradient expectation is $G = \mathbb{E}[g_i]$, and its covariance is $\Sigma = \text{Cov}(g_i)$.
  The sample mean estimator over a mini-batch of size $B$ has variance $\text{Var}(\bar{g}_B) = \frac{1}{B} \Sigma$.
  The Signal-to-Noise Ratio (SNR) of the batch gradient is:
  $$\text{SNR}(B) = \frac{\|G\|_2^2}{\frac{1}{B} \text{Tr}(\Sigma)} = \frac{B}{B_{\text{simple}}}, \quad \text{where } B_{\text{simple}} = \frac{\text{Tr}(\Sigma)}{\|G\|_2^2}$$
  - When $B \ll B_{\text{simple}}$, noise dominates: doubling $B$ doubles SNR (linear speedup regime).
  - When $B \gg B_{\text{simple}}$, signal dominates: further increasing $B$ yields negligible variance reduction (saturation regime).

## 3. Tensor Mechanics & Memory Layout Architecture
- **Layerwise vs Elementwise State Footprint:**
  - Standard AdamW computes elementwise ratios $\frac{m_i}{\sqrt{v_i}}$, requiring $O(N)$ tensor operations.
  - LAMB adds a layerwise reduction: for each parameter tensor $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, it computes two scalar Euclidean norms:
    $$\|W\|_2 = \sqrt{\sum_{i,j} W_{ij}^2}, \quad \|R\|_2 = \sqrt{\sum_{i,j} R_{ij}^2}$$
  - The trust ratio $\phi \in \mathbb{R}^1$ is broadcast across all elements of the layer tensor.

```
Elementwise Adam Step:                  Layerwise Trust Scaling:
┌───────────────────────────┐         ┌───────────────────────────┐
│ r_0,0   r_0,1 ... r_0,C   │         │ ||W||_2 / ||R||_2 = φ     │
│ r_1,0   r_1,1 ... r_1,C   │ ──────> │                           │
│  ...     ...  ...  ...    │         │ ΔW = - η * φ * R          │
│ r_R,0   r_R,1 ... r_R,C   │         └───────────────────────────┘
└───────────────────────────┘         All elements scaled by scalar φ
```

## 4. Multi-Ecosystem Comparative Implementation Matrix
| Feature / Characteristic | PyTorch 2.x | JAX / Equinox | Candle (Rust) | WGSL / Hardware Kernel |
| :--- | :--- | :--- | :--- | :--- |
| LARS Step | Custom fused module | Pure functional tree | `step_lars` Candle | Fused `LARS_SHADER` |
| LAMB Step | Fused CUDA / TorchScript | `optax.lamb` | `step_lamb` Candle | Fused `LAMB_SHADER` |
| Trust Ratio Reduction | `torch.norm` layerwise | `jnp.linalg.norm` | `.sqr()?.sum_all()?.sqrt()` | 2-pass norm + uniform dispatch |
| Gradient Noise Scale | Batched sample trace | Batched `jax.vmap` | `compute_gradient_noise_scale` | `gradient_noise_scale_cpu` |

## 5. Pathological Failure Modes, Anti-Patterns & Diagnostic Guide
- **Unconstrained Linear Scaling on 1D Normalization Layers:**
  - *Symptom:* Training diverges with `NaN` loss during warmup when scaling learning rate linearly ($\eta \propto B$).
  - *Root Cause:* In LayerNorm and RMSNorm, scale parameters $\gamma \in \mathbb{R}^D$ are initialized to $1.0$ and have very small norm ($\|\gamma\|_2 = \sqrt{D}$), while gradients can be substantial. If trust ratios are applied to 1D layers or if unconstrained large learning rates are applied to them, activation scales explode exponentially across layers.
  - *Remedy:* Explicitly exclude 1D parameters, biases, and normalization gains from trust ratio modulation ($\phi = 1.0$) or clamp $\phi \le 10.0$.
- **Batch Scaling Beyond Critical Noise Scale:**
  - *Symptom:* Scaling batch size $B$ from $16\text{k}$ to $64\text{k}$ requires $4\times$ more compute hours without reducing total training steps to reach target loss.
  - *Root Cause:* Mini-batch size exceeds $B_{\text{simple}} = \frac{\text{Tr}(\Sigma)}{\|G\|^2}$. In this regime, gradients are already deterministic ($SNR \gg 1$), and larger batches simply compute redundant gradient averages.
  - *Remedy:* Measure $B_{\text{simple}}$ across training; scale batch size only up to $B_{\text{crit}} \approx B_{\text{simple}}$, or employ dynamic batch size schedules that grow $B$ as training progresses (since $\|G\|^2 \to 0$ near minima, causing $B_{\text{simple}}$ to expand).

## 6. Cross-Domain Generalisation & Mathematical Isomorphisms
- **Control Theory:** The layerwise trust ratio acts as an adaptive automatic gain controller (AGC) that normalizes closed-loop feedback gain across heterogeneous sub-systems with disparate transfer functions.
- **Geodesic Optimization:** LAMB's trust ratio normalizes parameter movement along the Riemannian manifold of deep representations, ensuring that step length is calibrated relative to the intrinsic curvature of each layer.
- **Signal-to-Noise Ratio (SNR) in Communications:** The Gradient Noise Scale is isomorphic to Shannon channel capacity formulations, where increasing transmit power (batch size) beyond channel noise floor yields logarithmic rather than linear capacity gains.

## 7. Verification, Parity & Benchmarking Protocol
- **Parity Tolerance:** Strict $\|Y_A - Y_B\|_\infty < 10^{-5}$ across PyTorch, JAX, Candle, and Hardware Kernels.
- **Reference Vectors:** Verified against `weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.safetensors`.
- **Target Invariants:**
  1. $p_{\text{lars}}$ and $v_{\text{lars}}$ match reference within $10^{-5}$.
  2. $p_{\text{lamb}}$, $m_{\text{lamb}}$, and $v_{\text{lamb}}$ match reference within $10^{-5}$.
  3. Trust ratios $\phi_{\text{lars}}$ and $\phi_{\text{lamb}}$ match reference within $10^{-5}$.
  4. Gradient Noise Scale $B_{\text{simple}}$ matches reference within $10^{-5}$.

## 8. Navigation & Synchronized Track Artifacts
- [Interactive Concept Explorer Notebook](../../notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.ipynb)
- [Production PyTorch 2.x Module](../../pytorch2/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.py)
- [Functional JAX / Equinox Module](../../jax/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.py)
- [Native Rust Candle Module](../../candle/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.rs)
- [Hardware Kernels (WGSL + Native Fallback)](../../kernels/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.rs)
