# Schedule-Free and Parameter-Free Optimisers: Schedule-Free AdamW, Polyak Step-Size, Barzilai-Borwein, and D-Adaptation

## 1. Executive Summary & Intuitive Mental Model

In standard deep learning workflows, learning rate schedules (cosine annealing, linear decay, warmup-stable-decay) require fixing the total training horizon $T_{\max}$ upfront. If training is stopped early, the model remains under-converged because the learning rate was still high; if training is extended past $T_{\max}$, the learning rate hits zero or wanders out-of-schedule. Furthermore, tuning base learning rates $\eta$ consumes enormous compute across grid searches.

**Schedule-Free and Parameter-Free Optimisers** eliminate both requirements:
1. **Schedule-Free Optimisation (Defazio et al. 2024)**: Replaces decaying learning rate schedules with continuous, parameter-free iterate averaging. Instead of decaying $\eta_t \to 0$, the optimizer maintains a base sequence $z_t$, an averaged sequence $x_t$, and evaluates gradients at an interpolated sequence $y_t = (1 - c_t) z_t + c_t x_t$. Training can continue indefinitely or be stopped at any arbitrary step $T$ while matching or exceeding the final performance of cosine decay schedules!
2. **Polyak Step-Size (Polyak 1969)**: For problems with known or estimated optimal loss $f^*$, scales the step size dynamically via $\gamma_t = \frac{f(x_t) - f^*}{\|\nabla f(x_t)\|^2}$, guaranteeing linear convergence on quadratics with zero step-size hyperparameters.
3. **Barzilai-Borwein Spectral Step-Size (Barzilai & Borwein 1988)**: Approximates inverse curvature using two consecutive points: $\gamma_t^{BB1} = \frac{s_{t-1}^T s_{t-1}}{s_{t-1}^T y_{t-1}}$. Non-monotonic step selection breaks alignment with ill-conditioned eigenvectors, accelerating convergence by orders of magnitude without line searches.
4. **D-Adaptation & Prodigy (Defazio & Mishchenko 2023; Mishchenko & Defazio 2023)**: Estimates the unknown distance to the optimal solution $D = \|x_0 - x^*\|$ on the fly, setting the learning rate dynamically to $\eta_t = \eta_0 \cdot \hat{D}_t$ without requiring any manual LR tuning.

```
+----------------------------------------------------------------------------------------------------+
|                         SCHEDULE-FREE & PARAMETER-FREE OPTIMISER TAXONOMY                          |
+------------------------------------+-----------------------------------+---------------------------+
| Paradigm                           | Core Mechanism                    | Primary Benefit           |
+------------------------------------+-----------------------------------+---------------------------+
| Schedule-Free AdamW / SGD          | Iterate Interpolation y = (1-c)z+cx| No T_max horizon tuning   |
| Polyak Step-Size                   | gamma = (f(x) - f*) / ||g||^2     | Exact step for known f*   |
| Barzilai-Borwein (BB1 / BB2)       | gamma = (s^T s) / (s^T y)         | Spectral curvature scaling|
| D-Adaptation & Prodigy             | Online D = ||x_0 - x*|| estimate  | Parameter-free base LR    |
| Coin-Betting (COCOB)               | Algorithmic betting bounds        | Zero learning rate needed |
+------------------------------------+-----------------------------------+---------------------------+
```

---

## 2. Mathematical Foundations & First-Principles Derivations

### 2.1 The Schedule-Free Paradigm
Classical optimization theory proves that decaying step sizes (e.g. $\eta_t = \mathcal{O}(1/\sqrt{t})$ or cosine schedules) are necessary for stochastic gradient descent to converge to the exact minimum rather than oscillating in a noise ball of radius $\mathcal{O}(\eta \sigma^2)$. However, *Polyak-Ruppert averaging* proves that maintaining a running average of un-decayed iterates $\bar{x}_t = \frac{1}{t}\sum_{\tau=1}^t x_\tau$ achieves the same asymptotically optimal convergence rate.

Schedule-Free optimization (*Defazio et al. 2024*) unifies momentum and averaging into a three-sequence dynamic system:
- **Base Buffer $z_t$**: Updates aggressively via gradient steps without decay.
- **Averaged Iterate $x_t$**: Tracks the smooth, low-variance center of mass.
- **Evaluation Iterate $y_t$**: Interpolates between $z_t$ and $x_t$ to evaluate the gradient.

Given gradient $g_t = \nabla f(y_t)$ at step $t$:
$$m_{t} = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_{t} = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$u_t = \frac{m_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda y_t$$
$$z_{t+1} = z_t - \gamma u_t$$
$$c_t' = \frac{1}{t + 1}, \quad x_{t+1} = (1 - c_t') x_t + c_t' z_{t+1}$$
$$c_{t+1} = \frac{1}{t + 2}, \quad y_{t+1} = (1 - c_{t+1}) z_{t+1} + c_{t+1} x_{t+1}$$

**Crucial Invariant**: In `training` mode, the model parameters must be set to $y_t$. In `eval` mode or when checkpointing for deployment, parameters must be swapped to $x_t$!

### 2.2 Polyak Step-Size
For convex functions $f$ with Lipschitz continuous gradient and known global minimum $f^* = \min_x f(x)$, consider the distance to the optimum $V(x) = \frac{1}{2} \|x - x^*\|^2$. Taking a gradient step $x_{t+1} = x_t - \gamma_t g_t$:
$$\|x_{t+1} - x^*\|^2 = \|x_t - x^*\|^2 - 2\gamma_t g_t^T (x_t - x^*) + \gamma_t^2 \|g_t\|^2$$
By convexity, $f(x^*) \ge f(x_t) + g_t^T (x^* - x_t) \implies g_t^T (x_t - x^*) \ge f(x_t) - f^*$. Substituting:
$$\|x_{t+1} - x^*\|^2 \le \|x_t - x^*\|^2 - 2\gamma_t (f(x_t) - f^*) + \gamma_t^2 \|g_t\|^2$$
Minimizing this upper bound with respect to $\gamma_t$ yields the exact **Polyak Step-Size**:
$$\frac{d}{d\gamma_t} \left[ -2\gamma_t (f(x_t) - f^*) + \gamma_t^2 \|g_t\|^2 \right] = -2(f(x_t) - f^*) + 2\gamma_t \|g_t\|^2 = 0$$
$$\gamma_t = \frac{f(x_t) - f^*}{\|g_t\|^2}$$

### 2.3 Barzilai-Borwein Spectral Step-Sizes
The classical secant equation for quasi-Newton methods is $B_k s_{k-1} = y_{k-1}$ where $s_{k-1} = x_k - x_{k-1}$ and $y_{k-1} = g_k - g_{k-1}$. Instead of a full matrix $B_k$, Barzilai & Borwein (1988) constrained the approximation to a scalar multiple of identity $B_k = \frac{1}{\gamma} I$:
$$\min_\gamma \left\| \frac{1}{\gamma} s_{k-1} - y_{k-1} \right\|_2^2 \implies \gamma_k^{BB1} = \frac{s_{k-1}^T s_{k-1}}{s_{k-1}^T y_{k-1}}$$
$$\min_\gamma \left\| s_{k-1} - \gamma y_{k-1} \right\|_2^2 \implies \gamma_k^{BB2} = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}$$
The update $x_{k+1} = x_k - \gamma_k^{BB1} g_k$ alternates between overshooting and undershooting along the spectrum of $H$, breaking monotonic traps and converging rapidly on ill-conditioned problems without line searches.

### 2.4 D-Adaptation & Prodigy
Standard adaptive methods require guessing $\eta$. If $\eta$ is off by $10\times$, training fails. D-Adaptation (*Defazio & Mishchenko 2023*) proves that the optimal learning rate is proportional to the distance to the optimum $D = \|x_0 - x^*\|$:
$$\eta_t = \eta_0 \cdot \hat{D}_t$$
where $\hat{D}_t$ is an online lower-bound tracking $\max_{\tau \le t} \frac{\|s_\tau\|_1}{C}$. Prodigy (*Mishchenko & Defazio 2023*) extends this with adaptive second-moment growth rates, achieving optimal test error with zero base LR tuning.

---

## 3. Tensor Mechanics & Memory Layout Architecture

```
TENSOR FLOW: SCHEDULE-FREE DUAL BUFFER HARNESS
=============================================

1. Schedule-Free Memory Buffers:
   x: [d] (running average iterate - EVAL / DEPLOY WEIGHTS)
   z: [d] (base un-decayed buffer - INTERNAL OPTIMIZER STATE)
   y: [d] (interpolated iterate - TRAINING / FORWARD WEIGHTS)
   m: [d] (first moment EMA of g_t)
   v: [d] (second moment EMA of g_t^2)
   Total Footprint: 5 * d * sizeof(float32) bytes (zero schedule state required!)

2. Polyak & BB State:
   polyak_param: [d], polyak_grad: [d], scalars f_val, f_opt
   bb_param:     [d], bb_grad:     [d], bb_s: [d], bb_y: [d]
```

---

## 4. Multi-Ecosystem Comparative Implementation Matrix

| Feature / Metric | PyTorch 2.x (`pytorch2/`) | JAX / Functional (`jax/`) | Rust Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Schedule-Free State** | State dict with `x, z, y` | Pure tuple `(x, z, y, m, v)` | In-place Candle tensors | Fused storage buffer shader |
| **Polyak Step** | Vectorized PyTorch math | Pure `np.sum(grad**2)` | Pure Rust f64 accumulators | CPU / WGSL vector math |
| **Barzilai-Borwein** | Dot product `torch.dot` | Pure `np.dot` | SIMD dot products in f64 | Native CPU / GPU dot shader |
| **D-Adaptation** | Auto L1/L2 norm scaling | Vectorized NumPy norms | Pure Rust vector iterators | Fused CPU / GPU dispatch |
| **Parity Invariant** | Golden reference (seed 42) | $< 10^{-6}$ max abs diff | $< 10^{-6}$ max abs diff | $< 10^{-6}$ max abs diff |

---

## 5. Pathological Failure Modes, Anti-Patterns & Diagnostic Guide

### Failure Mode 1: Evaluating Schedule-Free Models in Training Mode ($y_t$ instead of $x_t$)
- **Mechanism**: In Schedule-Free optimization, $y_t$ is the training iterate where gradients are queried. Because $z_t$ takes aggressive un-decayed steps, $y_t$ carries high stochastic noise. The actual smooth, low-variance solution is $x_t$ (the Polyak average). Evaluating or deploying $y_t$ directly causes severe test loss spikes and accuracy degradation.
- **Diagnostic Signal**: Validation loss during training appears erratic, or test evaluation performance drops sharply compared to traditional cosine-annealed runs.
- **Cure**: Always call `optimizer.eval()` before running validation or saving checkpoints, which swaps model weights to $x_t$. Call `optimizer.train()` to restore $y_t$ before continuing training.

### Failure Mode 2: Polyak Step Size Division by Zero Near Minima
- **Mechanism**: Polyak's step size $\gamma_t = \frac{f(x_t) - f^*}{\|g_t\|^2}$ divides by $\|g_t\|^2$. Near critical points where $\|g_t\| \to 0$, if $f(x_t) - f^*$ is contaminated by numerical noise, $\gamma_t \to \infty$, causing catastrophic parameter explosion.
- **Diagnostic Signal**: Sudden `NaN` or `Inf` weights when the model approaches convergence.
- **Cure**: Add a stabilizing denominator epsilon $\gamma_t = \frac{f(x_t) - f^*}{\|g_t\|^2 + \epsilon}$ and clip $\gamma_t \le \gamma_{\max}$.

---

## 6. Cross-Domain Generalisation & Mathematical Isomorphisms

| Discipline / Domain | Isomorphic Problem | Parameter-Free / Schedule-Free Role |
| :--- | :--- | :--- |
| **Continual Learning & Streaming** | Non-stationary Data Streams | Schedule-Free enables perpetual training without deciding token horizons upfront |
| **Scientific Computing & PDE Inversion** | Least-Squares Optimization | Polyak step-size provides optimal damping with zero tuning when minimum residual is known |
| **Signal Processing & Geophysics** | Seismic Inversion & Deconvolution | Barzilai-Borwein spectral steps accelerate gradient descent $10\times$ on ill-conditioned wave equations |
| **Hyperparameter Optimization (HPO)** | AutoML & Black-Box Search | D-Adaptation removes learning rate from the hyperparameter search space |
| **Robotics & Online Control** | Real-Time Trajectory Optimization | Parameter-free step sizes eliminate offline schedule calibration during runtime changes |

---

## 7. Verification, Parity & Benchmarking Protocol

All implementations verify 5 quantitative invariants against seed-42 golden reference tensors:
1. `inv1_schedule_free`: Multi-buffer state update (`sf_x`, `sf_z`, `sf_y`) under Schedule-Free AdamW ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).
2. `inv2_polyak`: Parameter update under Polyak optimal step size ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).
3. `inv3_bb`: Parameter update under Barzilai-Borwein BB1 spectral step ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).
4. `inv4_d_adapt`: Parameter update and distance estimate under D-Adaptation ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).
5. `inv5_anchor`: Deterministic linear projection baseline ($\|Y - Y_{\text{golden}}\|_\infty < 10^{-5}$).

---

## 8. Navigation & Synchronized Track Artifacts

- **Jupyter Notebook Track**: [`notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.ipynb)
- **PyTorch 2.x Track**: [`pytorch2/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.py)
- **JAX Functional Track**: [`jax/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.py)
- **Rust Candle Track**: [`candle/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.rs)
- **Hardware Kernel Track**: [`kernels/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.rs)
- **Reference Checkpoints**: [`weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.safetensors)
