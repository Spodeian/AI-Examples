# First-Order and Adaptive Optimisers (Topic 01)

## 1. Executive Summary & Intuitive Mental Model
- **Core Problem & Intuition:** Deep learning training dynamics inhabit high-dimensional non-convex loss surfaces characterized by pathological curvature, ill-conditioned valleys, saddle points, and stochastic gradient noise. Standard Stochastic Gradient Descent (SGD) oscillates violently across narrow canyon walls while making glacial progress along low-curvature directions. Momentum accelerates traversal through valleys by building inertia along consistent directions. Adaptive learning rate optimisers scale per-parameter updates using historical gradient magnitudes: Adagrad accumulates all past squared gradients, RMSprop discounts past history with an exponential moving average, Adam combines momentum with adaptive variance scaling, and AdamW untangles decoupled weight decay from adaptive step sizes.
- **Historical Lineage & Seminal Citations:**
  - *Robbins & Monro (1951)*: Introduced stochastic approximation and established conditions for convergence ($\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$).
  - *Polyak (1964)*: Heavy-ball momentum method adding an inertial velocity vector to damp high-frequency oscillations.
  - *Nesterov (1983)*: Accelerated gradient method computing gradients ahead of the current position along the momentum vector.
  - *Duchi, Hazan & Singer (2011)*: Adagrad introducing per-coordinate learning rates inversely proportional to the square root of historical squared gradients.
  - *Tieleman & Hinton (2012)*: RMSprop resolving Adagrad's premature learning rate decay via exponential moving averages.
  - *Kingma & Ba (2014)*: Adam uniting first-moment momentum with second-moment uncentered variance preconditioning and bias correction.
  - *Loshchilov & Hutter (2017)*: AdamW uncovering the hidden $L_2$ weight decay coupling pathology in Adam and introducing true decoupled weight shrinkage.
  - *Zhuang et al. (2020)*: AdaBelief adapting step sizes according to the "belief" in current gradient direction by tracking $|g_t - m_t|^2$.
- **Decision Boundary:**
  - *When to use AdamW:* The default workhorse for Transformers, LLMs, vision backbones, and multi-modal models where non-convex landscapes and disparate gradient scales across layers require adaptive preconditioning and exact weight shrinkage.
  - *When to use SGD with Momentum:* When training deep convolutional networks (e.g. ResNets, ConvNeXts) from scratch on vision benchmarks where SGD's implicit regularization finds broader, more generalizable minima than adaptive optimisers.
  - *When NOT to use standard Adam:* Never use coupled $L_2$ regularization in Adam when weight decay is intended, as Adam penalizes frequently updated weights exponentially less than infrequently updated ones.

## 2. Mathematical Foundations & First-Principles Derivations
- **SGD with Momentum:**
  Given stochastic gradient $g_t = \nabla_\theta \mathcal{L}_t(\theta_t)$ and weight decay $\lambda$:
  $$\tilde{g}_t = g_t + \lambda \theta_t$$
  $$v_{t+1} = \mu v_t + \tilde{g}_t$$
  $$\theta_{t+1} = \theta_t - \eta v_{t+1}$$
- **Adagrad:**
  $$G_{t+1} = G_t + \tilde{g}_t \odot \tilde{g}_t$$
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} \odot \tilde{g}_t$$
  *Pathology:* As $t \to \infty$, $G_t \to \infty$ monotonically, driving the effective learning rate $\frac{\eta}{\sqrt{G_t}} \to 0$ and stopping learning prematurely.
- **RMSprop:**
  $$v_{t+1} = \alpha v_t + (1 - \alpha) \tilde{g}_t \odot \tilde{g}_t$$
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1}} + \epsilon} \odot \tilde{g}_t$$
- **Adam (Coupled $L_2$ Regularization):**
  $$m_{t+1} = \beta_1 m_t + (1 - \beta_1) \tilde{g}_t, \quad v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\tilde{g}_t \odot \tilde{g}_t)$$
  $$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \quad \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \odot \hat{m}_{t+1}$$
  *The $L_2$ Coupling Pathology:* Notice $\tilde{g}_t = g_t + \lambda \theta_t$ is injected directly into $m_t$ and $v_t$. When gradient magnitudes $g_t$ are large, $\sqrt{v_t}$ is large, so the effective decay penalty on parameter $\theta$ is $\frac{\lambda \theta}{\sqrt{v_t}} \ll \lambda \theta$. Parameters with large gradients escape weight decay, while parameters with tiny gradients receive disproportionately massive shrinkage!
- **AdamW (Decoupled Weight Decay):**
  Loshchilov & Hutter (2017) decouple the weight decay step completely from gradient normalization:
  $$m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t, \quad v_{t+1} = \beta_2 v_t + (1 - \beta_2) (g_t \odot g_t)$$
  $$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \quad \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$
  $$\theta_{t+1} = \theta_t (1 - \eta \lambda) - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \odot \hat{m}_{t+1}$$
  Here, every parameter experiences exact proportional shrinkage $(1 - \eta \lambda)$ independent of gradient scale.
- **AdaBelief:**
  Adapts the step size based on the difference between the observed gradient and its exponential moving average:
  $$s_{t+1} = \beta_2 s_t + (1 - \beta_2) (g_t - m_{t+1})^2 + \epsilon$$
  When gradients are consistent ($g_t \approx m_{t+1}$), $s_{t+1}$ is small, yielding large steps. When gradients fluctuate wildly, $s_{t+1}$ is large, damping the step.

## 3. Tensor Mechanics & Memory Layout Architecture
- **State Buffer Footprint:**
  - SGD: $0\times$ extra memory ($v_t$ optional for momentum $\implies 1\times$).
  - Adagrad / RMSprop: $1\times$ extra state memory ($G_t$ or $v_t$).
  - Adam / AdamW / AdaBelief: $2\times$ extra state memory ($m_t$ and $v_t$). For a 7B parameter model in FP32, $2 \times 7\times 10^9 \times 4 \text{ bytes} \approx 56\text{ GB}$ of optimizer states alone!
- **Kernel Layout:**
  Fusing the elementwise update into a single GPU compute pass reading parameter, gradient, $m$, and $v$ buffers in coalesced 64-byte chunks eliminates multiple global memory roundtrips.

```
       ┌────────────────────────────────────────────────────────┐
       │                   Memory Stride [N]                    │
       ├──────────────┬──────────────┬──────────────┬───────────┤
Param  │   theta_0    │   theta_1    │   theta_2    │    ...    │
Grad   │     g_0      │     g_1      │     g_2      │    ...    │
M_buf  │     m_0      │     m_1      │     m_2      │    ...    │
V_buf  │     v_0      │     v_1      │     v_2      │    ...    │
       └──────────────┴──────────────┴──────────────┴───────────┘
```

## 4. Multi-Ecosystem Comparative Implementation Matrix
| Feature / Characteristic | PyTorch 2.x | JAX / Equinox | Candle (Rust) | WGSL / Hardware Kernel |
| :--- | :--- | :--- | :--- | :--- |
| Primary Abstraction | `torch.optim.AdamW` / `step_*` | Functional state transforms | `step_adamw` Candle | Fused `ADAMW_SHADER` |
| In-Place Mutation | Native `.copy_()` / `add_()` | Pure functional returns | Immutable tensor transforms | In-place storage buffers |
| Decoupled Decay Support | Yes (`AdamW`) | Yes (`optax.adamw`) | Yes (`step_adamw`) | Direct $(1 - \eta \lambda)$ term |
| GPU Kernel Fusion | `torch.compile` / Inductor | XLA operator fusion | Native backend dispatch | Fused WGSL threadblock |

## 5. Pathological Failure Modes, Anti-Patterns & Diagnostic Guide
- **Adam $L_2$ Decay Misunderstanding:**
  - *Symptom:* Setting `weight_decay=0.01` in standard `torch.optim.Adam` fails to regularize large active weights, leading to parameter explosion and overfitting in LLMs.
  - *Root Cause:* $L_2$ penalty is added into the gradient vector before variance preconditioning: $\theta \leftarrow \theta - \frac{\eta}{\sqrt{v_t}}(g_t + \lambda \theta)$. Active weights with large $v_t$ have their decay scaled down by $1/\sqrt{v_t}$.
  - *Remedy:* Replace `Adam` with `AdamW`.
- **Vanishing Bias Correction:**
  - *Symptom:* Step 1 diverges with NaN or gigantic updates if $\beta_1^t$ or $\beta_2^t$ is omitted.
  - *Root Cause:* At $t=1$, uncorrected $m_1 = (1-\beta_1)g_1 = 0.1 g_1$ and $v_1 = (1-\beta_2)g_1^2 = 0.001 g_1^2$. The denominator $\sqrt{0.001} \approx 0.0316$ while numerator is $0.1$, scaling the update by $>3.16\times$ without bias correction.

## 6. Cross-Domain Generalisation & Mathematical Isomorphisms
- **Control Theory:** Heavy-ball momentum is mathematically identical to a second-order damped harmonic oscillator ($m \ddot{x} + \gamma \dot{x} + \nabla V(x) = 0$).
- **Fluid Dynamics:** Advection-diffusion transport equations mirror momentum-accelerated parameter flows in potential force fields.
- **Signal Processing:** Second-moment variance tracking in RMSprop/Adam is isomorphic to an Automatic Gain Control (AGC) circuit normalizing non-stationary signal powers.

## 7. Verification, Parity & Benchmarking Protocol
- **Parity Tolerance:** Strict $\|Y_A - Y_B\|_\infty < 10^{-5}$ across PyTorch, JAX, Candle, and Hardware Kernels.
- **Reference Vectors:** Verified against `weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.safetensors`.

## 8. Navigation & Synchronized Track Artifacts
- [Interactive Concept Explorer Notebook](../../../notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.ipynb)
- [Production PyTorch 2.x Module](../../../pytorch2/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.py)
- [Functional JAX / Equinox Module](../../../jax/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.py)
- [Native Rust Candle Module](../../../candle/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.rs)
- [Hardware Kernel](../../../kernels/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.rs)
- [Golden SafeTensors Weights](../../../weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.safetensors)
