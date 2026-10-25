# Implicit Neural Representations: SIREN, Instant-NGP, NeRF, and 3DGS

## Executive Summary & ELI5

Traditional computer graphics and vision represent 3D scenes explicitly using discrete polygons (triangle meshes), voxel grids, or depth maps. However, explicit representations scale poorly with resolution: voxel memory scales cubically ($O(N^3)$), while dense triangle meshes are topologically non-differentiable.

**Implicit Neural Representations (INRs)** and **Neural Radiance Fields (NeRF)** re-envision physical signals as continuous mathematical functions parameterized by deep neural networks $f_\theta: \mathbb{R}^D \to \mathbb{R}^M$. Instead of storing discrete color values in a grid, the network takes a continuous $(x, y, z)$ spatial coordinate (plus viewing direction) and outputs the local density and radiance:
- **SIREN (Sinusoidal Representation Networks)**: Discovers that standard activations (ReLU, GELU, Tanh) suffer from spectral bias, failing to learn high-frequency details and non-zero higher-order spatial derivatives. By using periodic sinusoidal activations $\phi(x) = \sin(\omega_0 (W x + b))$ with principled initialization, SIREN models complex signed distance fields (SDFs), images, audio signals, and Poisson boundary value problems.
- **NeRF (Neural Radiance Fields)**: Synthesizes photorealistic novel views from 2D photos by casting rays through a continuous 5D neural scene and numerically integrating classical volume rendering equations.
- **Instant-NGP (Instant Neural Graphics Primitives)**: Replaces massive MLPs with multi-resolution spatial hash tables, enabling real-time neural graphics training in seconds.
- **3DGS (3D Gaussian Splatting)**: Bridges the gap between continuous radiance fields and raster graphics pipelines by modeling scenes as millions of anisotropic 3D Gaussians splatted directly onto screen tiles in real time (>100 FPS).

---

## Mathematical Foundations & Governing Equations

### 1. SIREN Periodic Activation and Derivative Properties
Standard MLPs with ReLU activations have second derivatives that are identically zero everywhere except at boundaries. SIREN adopts periodic activations:
$$\Phi(x) = \sin(\omega_0 (W x + b))$$
Because $\frac{d}{dx} \sin(x) = \cos(x) = \sin(x + \frac{\pi}{2})$, any derivative of a SIREN is itself a phase-shifted SIREN with identical spectral properties.
To prevent vanishing or exploding gradients across deep layers, weights $W$ are initialized uniformly from:
$$W_0 \sim \mathcal{U}\left(-\frac{1}{n}, \frac{1}{n}\right) \quad (\text{first layer}), \quad W_l \sim \mathcal{U}\left(-\frac{\sqrt{6/n}}{\omega_0}, \frac{\sqrt{6/n}}{\omega_0}\right) \quad (l > 0)$$
where $\omega_0 \approx 30.0$ scales high-frequency input spatial features.

### 2. NeRF Continuous Volume Rendering Equation
Given a camera ray $r(t) = o + t d$ parameterized by camera origin $o$, viewing direction $d$, and bounds $t \in [t_n, t_f]$, the expected rendered color $C(r)$ is:
$$C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt, \quad T(t) = \exp\left(-\int_{t_n}^t \sigma(r(s)) ds\right)$$
where $\sigma(x)$ is volume density (extinction coefficient), $c(x, d)$ is directional emitted color, and $T(t)$ is accumulated transmittance (probability of light traveling from $t_n$ to $t$ without hitting particles).

Discretized along $S$ sampled points $\{t_1 < t_2 < \dots < t_S\}$ with step size $\delta_i = t_{i+1} - t_i$:
$$\hat{C}(r) = \sum_{i=1}^S T_i \alpha_i c_i, \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i), \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right) = \prod_{j=1}^{i-1} (1 - \alpha_j)$$

### 3. 3D Gaussian Splatting Alpha Evaluation
A 3D Gaussian is parameterized by center position $\mu \in \mathbb{R}^3$, covariance matrix $\Sigma \in \mathbb{R}^{3 \times 3}$, opacity $o \in [0, 1]$, and spherical harmonic color coefficients:
$$G(x) = \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$
Projected into 2D camera pixel space via viewing transform $W$ and Jacobian $J$, the 2D screen covariance is $\Sigma' = J W \Sigma W^T J^T$.
The 2D Gaussian splat evaluates alpha density at screen coordinate $p$:
$$\alpha(p) = o \cdot \exp\left(-\frac{1}{2} (p - \mu_{2D})^T (\Sigma')^{-1} (p - \mu_{2D})\right)$$

---

## Tensor Memory Layout & Implementation Details

### NeRF Volume Rendering Quadrature Buffer
For batch of rays $B$ and $S$ samples along ray:
- Sample densities: $\sigma \in \mathbb{R}^{B \times S}$
- Sample colors: $c \in \mathbb{R}^{B \times S \times 3}$
- Interval distances: $\delta \in \mathbb{R}^{B \times S}$
- Optical depth: $\tau = \text{ReLU}(\sigma) \odot \delta \in \mathbb{R}^{B \times S}$
- Segment opacity: $\alpha = 1 - \exp(-\tau) \in \mathbb{R}^{B \times S}$
- Cumulative optical depth: $\tau_{\text{cum}} = \text{cumsum}(\tau) \in \mathbb{R}^{B \times S}$
- Transmittance: $T = \exp(-\text{shift\_right}(\tau_{\text{cum}})) \in \mathbb{R}^{B \times S}$
- Quadrature weights: $w = T \odot \alpha \in \mathbb{R}^{B \times S}$
- Composite color: $C_{\text{ray}} = \sum_{i=1}^S w_i \cdot c_i \in \mathbb{R}^{B \times 3}$

---

## Multi-Ecosystem Implementation Matrix

| Operation / Feature | PyTorch 2.x | JAX (vmap + jit) | Candle (Rust) | WGSL Compute Shader |
| :--- | :--- | :--- | :--- | :--- |
| **SIREN Activation** | `torch.sin(omega * (x @ W + b))` | `jnp.sin(omega * ...)` | `affine(omega, 0)?.sin()` | Native `sin()` WGSL instruction |
| **Optical Depth & Alpha** | `1.0 - torch.exp(-tau)` | `1.0 - jnp.exp(-tau)` | Chained `exp()` + `affine` | Fused ray march instruction |
| **Transmittance Cumsum** | `cumsum` + `cat` zero-pad | `jnp.cumsum` + concat | Host-side prefix loop or tensor | Parallel exclusive scan |
| **Gaussian Splat Quad Form** | `sum((delta @ cov_inv) * delta)` | Vectorized quadratic product | `matmul` + `sum_keepdim` | Per-pixel 2x2 inverse dot product |
| **Parity Tolerance** | Native float32 | $< 10^{-6}$ vs golden | $< 10^{-6}$ vs golden | $< 10^{-5}$ vs golden |

---

## Pathological Misuse & Failure Modes

1. **Standard ReLU Initialization in SIREN**:
   - *Anti-Pattern*: Initializing SIREN layers with standard He or Xavier Gaussian distributions without $\omega_0$ scaling.
   - *Consequence*: The sinusoidal activations quickly collapse into zero gradients or high-frequency chaotic noise, completely preventing convergence on coordinate regressions.
   - *Remedy*: Strictly use Sitzmann uniform initialization: $W_l \sim \mathcal{U}(-\sqrt{6/n}/\omega_0, \sqrt{6/n}/\omega_0)$.

2. **Negative Densities in Volume Rendering**:
   - *Anti-Pattern*: Leaving predicted density $\sigma$ unconstrained without clipping or ReLU activation.
   - *Consequence*: Negative densities yield optical depths $\tau < 0$, producing transmittances $T > 1$ (spontaneous light generation / amplification), causing visual blowouts and unbounded loss values.
   - *Remedy*: Pass predicted densities through $\text{ReLU}(\sigma)$ or $\text{softplus}(\sigma)$.

3. **Singular Covariances in 3DGS**:
   - *Anti-Pattern*: Parameterizing covariance $\Sigma$ without guaranteeing positive definiteness.
   - *Consequence*: Non-positive definite matrices cannot be inverted ($(\Sigma')^{-1}$ produces `NaN` or negative Mahalanobis distances), crashing real-time rasterizers.
   - *Remedy*: Decompose covariance into scale vector $s$ and rotation quaternion $q$: $\Sigma = R S S^T R^T$.

---

## Cross-Domain Generalisation Matrix

| INR Paradigm | Coordinate Space | Target Field | Primary Application |
| :--- | :--- | :--- | :--- |
| **SIREN** | 2D / 3D $(x, y, z)$ | Signed Distance $d$, Intensity, Audio $A$ | PDE solving, high-precision geometry fitting |
| **NeRF** | 5D $(x, y, z, \theta, \phi)$ | Volume Density $\sigma$, Emitted Radiance $c$ | Photorealistic novel view synthesis |
| **Instant-NGP** | Hash-interpolated 3D coordinates | Trilinear features $\to$ Density/Color | Real-time gigapixel neural graphics |
| **3D Gaussian Splatting** | 3D Anisotropic Gaussians | Screen-projected alpha splats | 100+ FPS cinematic real-time rendering |

---

## Verification, Parity & Empirical Benchmarks

The multi-ecosystem implementations across PyTorch 2.x, JAX, Candle, and WGSL are validated against `weights/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.safetensors` initialized with deterministic seed 42.

- **Parity Verification**:
  - `linear_anchor`: Parity error $< 10^{-6}$ across CPU and GPU.
  - `siren_layer`: Exact transcendental alignment for $\omega_0 = 30.0$ periodic transformations.
  - `volume_rendering_quadrature`: Transmittance prefix integration and composite color match to $< 10^{-6}$.
  - `gaussian_splat_2d_eval`: Quadratic form and exponential alpha falloff verified across all platforms.

---

## Synchronized Track Artifacts & Navigation

- **PyTorch 2.x Reference**: [`pytorch2/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.py`](file:///c:/Users/Liam/Code/AI-Examples/pytorch2/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.py)
- **Functional JAX Track**: [`jax/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.py`](file:///c:/Users/Liam/Code/AI-Examples/jax/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.py)
- **Candle Engine Implementation**: [`candle/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.rs`](file:///c:/Users/Liam/Code/AI-Examples/candle/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.rs)
- **Hardware Kernels & WGSL**: [`kernels/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.rs`](file:///c:/Users/Liam/Code/AI-Examples/kernels/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.rs)
- **Integration Tests**: [`crates/ai-candle/tests/05_architectures/test_37_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.rs`](file:///c:/Users/Liam/Code/AI-Examples/crates/ai-candle/tests/05_architectures/test_37_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.rs)
- **Interactive Jupyter Notebook**: [`notebooks/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.ipynb`](file:///c:/Users/Liam/Code/AI-Examples/notebooks/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.ipynb)
- **Golden Weights**: [`weights/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.safetensors`](file:///c:/Users/Liam/Code/AI-Examples/weights/05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.safetensors)
