# Modern Sign and Factored Optimisers (Topic 02)

## 1. Executive Summary & Intuitive Mental Model
- **Core Problem & Intuition:** As deep learning models scale into the tens and hundreds of billions of parameters, optimizer state memory becomes the primary bottleneck limiting batch size, sequence length, and model capacity. Standard AdamW allocates 8 bytes per parameter (two 32-bit floating-point buffers for $m_t$ and $v_t$), requiring 56 GB of GPU VRAM for a 7B model before accounting for weights, gradients, or activations. Modern sign and factored optimisers circumvent this memory wall via two distinct mechanisms:
  1. **Sign Momentum (`Lion` / `Tiger`):** Completely discards second-moment variance tracking ($v_t$), replacing per-parameter division with a uniform signum operator $\text{sign}(\beta_1 m_t + (1 - \beta_1) g_t)$, cutting optimizer memory by 50% ($1\times$ state buffer).
  2. **Factored Second Moments (`Adafactor` / `CAME` / `SM3`):** Replaces dense 2D variance matrices $V \in \mathbb{R}^{R \times C}$ with low-rank non-negative matrix factorizations (row vector $r \in \mathbb{R}^R$ and column vector $c \in \mathbb{R}^C$), compressing second-moment memory from $O(RC)$ to $O(R + C)$ (cutting memory by $>75\%$).
  3. **Multi-Horizon Momentum Mixtures (`AdEMAMix`):** Discovers that standard optimisers suffer from short-horizon memory bias, and unites a fast momentum EMA ($\beta_1 = 0.9$, half-life ~7 steps) with an ultra-slow momentum EMA ($\beta_3 = 0.9999$, half-life ~7,000 steps), accelerating convergence without destabilizing gradient tracking.
- **Historical Lineage & Seminal Citations:**
  - *Shazeer & Stern (2018)*: Introduced **Adafactor**, eliminating second-moment memory via rank-1 row/column factorization and parameter-free scale-dependent learning rates.
  - *Anil et al. (2019)*: **SM3** generalizing factored second moments to arbitrary $N$-dimensional tensor layouts using cover-set projections.
  - *Chen et al. (2023)*: Discovered **Lion (EvoLved Sign Momentum)** via symbolic program search, proving that pure signum updates produce larger gradient steps in low-magnitude directions and smaller steps in noisy directions.
  - *Luo et al. (2023)*: **CAME** augmenting Adafactor with confidence-guided second-moment cross-validation.
  - *Pagliardini et al. (2024)*: **AdEMAMix (Adaptive EMA Mixture)** showing that adding a slow-decaying momentum buffer allows models to retain historical gradient direction over thousands of training steps.
- **Decision Boundary:**
  - *When to use Lion:* When training memory-constrained LLMs or vision transformers where saving 50% optimizer memory enables doubling batch size or sequence length. *Critical:* Lion requires $3\text{--}10\times$ smaller base learning rate (e.g. $1\times 10^{-4}$ vs $1\times 10^{-3}$) and $3\text{--}10\times$ larger weight decay ($\lambda = 0.1\text{--}1.0$).
  - *When to use Adafactor:* Pretraining massive models ($>50\text{B}$) where even $1\times$ momentum memory is unaffordable; Adafactor operates with near-zero state overhead ($O(R+C)$).
  - *When to use AdEMAMix:* Fine-tuning and pretraining workloads where faster convergence and superior loss-per-step are prioritized over strict memory budgets.

## 2. Mathematical Foundations & First-Principles Derivations
- **Lion (EvoLved Sign Momentum):**
  Lion operates with a single first-moment buffer $m_t$:
  $$u_t = \text{sign}\left(\beta_1 m_t + (1 - \beta_1) g_t\right)$$
  $$\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta u_t$$
  $$m_{t+1} = \beta_2 m_t + (1 - \beta_2) g_t$$
  *Notice the crucial asymmetry:* The update vector $u_t$ is computed using $\beta_1$ (typically $0.9$), but the momentum state $m_t$ is updated using $\beta_2$ (typically $0.99$). Every coordinate of $u_t \in \{-1, +1\}^d$, ensuring that the parameter step has uniform Euclidean magnitude $\|u_t\|_2 = \sqrt{d}$ regardless of gradient scale!
- **Adafactor (Factored Second Moments for 2D Tensors):**
  Let $W_t, G_t \in \mathbb{R}^{R \times C}$.
  1. Exponential squared gradient:
     $$S_t = G_t \odot G_t + \epsilon_1$$
  2. Factored row and column accumulators:
     $$r_{t+1} = \hat{\beta}_{2,t} r_t + (1 - \hat{\beta}_{2,t}) \left(\frac{1}{C} \sum_{c=1}^C S_t[:, c]\right)$$
     $$c_{t+1} = \hat{\beta}_{2,t} c_t + (1 - \hat{\beta}_{2,t}) \left(\frac{1}{R} \sum_{r=1}^R S_t[r, :]\right)$$
  3. Outer-product reconstruction of approximate second moment:
     $$\bar{r} = \frac{1}{R} \sum_{r=1}^R r_{t+1}[r]$$
     $$V_t \approx \frac{r_{t+1} c_{t+1}^T}{\bar{r} + \epsilon_1} \in \mathbb{R}^{R \times C}$$
  4. Standardized preconditioning with RMS clipping:
     $$U_t = \frac{G_t}{\sqrt{V_t} + \epsilon_2}$$
     $$\text{RMS}(U_t) = \sqrt{\frac{1}{RC} \sum_{i,j} U_{t,ij}^2}, \quad \rho_t = \max\left(1.0, \frac{\text{RMS}(U_t)}{d_{\text{clip}}}\right)$$
     $$\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta \frac{U_t}{\rho_t}$$
- **AdEMAMix (Fast + Ultra-Slow Momentum Mixture):**
  Standard optimisers lose track of persistent gradient directions due to rapid forgetting ($\beta_1 = 0.9 \implies 99\%$ forgetting in 45 steps). AdEMAMix tracks two momentum EMAs:
  $$m_{1,t+1} = \beta_1 m_{1,t} + (1 - \beta_1) g_t, \quad (\beta_1 = 0.9)$$
  $$m_{2,t+1} = \beta_3 m_{2,t} + (1 - \beta_3) g_t, \quad (\beta_3 = 0.9999)$$
  $$v_{t+1} = \beta_2 v_t + (1 - \beta_2) (g_t \odot g_t), \quad (\beta_2 = 0.999)$$
  With bias-corrected terms $\hat{m}_{1,t}, \hat{m}_{2,t}, \hat{v}_t$:
  $$u_t = \frac{\hat{m}_{1,t+1} + \alpha \hat{m}_{2,t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}$$
  $$\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta u_t$$
  Where $\alpha \approx 4\text{--}8$ amplifies the persistent, low-noise signal accumulated by the ultra-slow buffer.

## 3. Tensor Mechanics & Memory Layout Architecture
- **State Footprint Comparison for 2D Weight Matrix $[R, C]$:**
  - *AdamW:* $2RC$ elements ($m$ and $v$ full matrices).
  - *Lion:* $1RC$ elements ($m$ only, no variance).
  - *Adafactor:* $R + C$ elements ($r$ row vector and $c$ column vector). For a $4096 \times 4096$ hidden matrix:
    - AdamW: $2 \times 4096^2 = 33,554,432$ floats ($134.2\text{ MB}$).
    - Lion: $1 \times 4096^2 = 16,777,216$ floats ($67.1\text{ MB}$).
    - Adafactor: $4096 + 4096 = 8,192$ floats ($0.032\text{ MB}$ — a **$4,096\times$ compression!**).

```
Dense Matrix V [R, C]:               Factored Rank-1 Representation:
┌───────────────────────────┐         c_vector: [ 1 x C ]
│ v_0,0   v_0,1 ... v_0,C   │         ┌───┬───┬───┬───┐
│ v_1,0   v_1,1 ... v_1,C   │         └───┴───┴───┴───┘
│  ...     ...  ...  ...    │       r_vector:
│ v_R,0   v_R,1 ... v_R,C   │         ┌───┐   Outer product:
└───────────────────────────┘         │   │   V ≈ (r * c^T) / mean(r)
33.5M floats (AdamW)                  └───┘   8.2k floats (Adafactor)
```

## 4. Multi-Ecosystem Comparative Implementation Matrix
| Feature / Characteristic | PyTorch 2.x | JAX / Equinox | Candle (Rust) | WGSL / Hardware Kernel |
| :--- | :--- | :--- | :--- | :--- |
| Lion Implementation | Native sign step | `optax.lion` | `step_lion` Candle | Fused `LION_SHADER` |
| Adafactor Factorization | `torch.optim.Adafactor` | `optax.adafactor` | `step_adafactor_2d` | Factored reduction shader |
| AdEMAMix Dual Buffers | Fused custom module | Pure functional state | `step_ademamix` Candle | Fused 3-state kernel |
| State Memory Savings | 50% (Lion) / >75% (Adafactor)| 50% / >75% | 50% / >75% | Zero extra global memory writes |

## 5. Pathological Failure Modes, Anti-Patterns & Diagnostic Guide
- **Lion Hyperparameter Re-use Failure:**
  - *Symptom:* Replacing AdamW with Lion using AdamW's learning rate ($\eta = 10^{-3}$) causes catastrophic divergence in early training steps.
  - *Root Cause:* In AdamW, updates are normalized by $\sqrt{v_t}$, so effective updates on noisy coordinates are heavily attenuated ($|u_i| \ll 1$). In Lion, every update coordinate has magnitude exactly $\pm 1$. With $\eta = 10^{-3}$, every single parameter moves by $\pm 10^{-3}$ at every step, destroying delicate initialization structures.
  - *Remedy:* Scale down $\eta$ by $3\text{--}10\times$ (to $\eta = 10^{-4}$), and scale up weight decay by $3\text{--}10\times$ ($\lambda = 0.1\text{--}1.0$).
- **Adafactor Rank-1 Variance Breakdown:**
  - *Symptom:* Adafactor updates explode or oscillate wildly on non-matrix 1D tensors (e.g. LayerNorm gains, biases).
  - *Root Cause:* Rank-1 factorization assumes row-column independence of gradient variances ($V_{ij} \propto r_i c_j$). For 1D parameters, there is no second dimension to factor; Adafactor must fall back to un-factored variance tracking or pure momentum.

## 6. Cross-Domain Generalisation & Mathematical Isomorphisms
- **Robust Statistics:** The signum operator in Lion is isomorphic to Huber robust estimation, replacing $L_2$ quadratic loss on outliers with an $L_1$ absolute loss that bounds gradient influence.
- **Low-Rank Tensor Decompositions:** Adafactor's rank-1 outer-product approximation mirrors CANDECOMP/PARAFAC (CP) tensor factorizations in multi-way chemometrics and quantum tensor networks.
- **Multi-Rate Signal Processing:** AdEMAMix's dual momentum EMAs implement a dual-band IIR low-pass filter separating short-term gradient fluctuations from secular trends.

## 7. Verification, Parity & Benchmarking Protocol
- **Parity Tolerance:** Strict $\|Y_A - Y_B\|_\infty < 10^{-5}$ across PyTorch, JAX, Candle, and Hardware Kernels.
- **Reference Vectors:** Verified against `weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.safetensors`.

## 8. Navigation & Synchronized Track Artifacts
- [Interactive Concept Explorer Notebook](../../../notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.ipynb)
- [Production PyTorch 2.x Module](../../../pytorch2/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.py)
- [Functional JAX / Equinox Module](../../../jax/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.py)
- [Native Rust Candle Module](../../../candle/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.rs)
- [Hardware Kernel](../../../kernels/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.rs)
- [Golden SafeTensors Weights](../../../weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.safetensors)
