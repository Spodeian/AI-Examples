# Neural Operators: Fourier Neural Operator (FNO), DeepONet, and MeshGraphNets

## 1. Executive Summary & Architectural Overview

Standard deep learning architectures map between finite-dimensional Euclidean spaces ($\mathbb{R}^N \to \mathbb{R}^M$). In scientific computing and continuum physics, however, governing partial differential equations describe mappings between **infinite-dimensional function spaces** (e.g. mapping an initial coefficient field $a(x) \in \mathcal{A}$ to a solution field $u(x) \in \mathcal{U}$). Traditional discretization methods (Finite Elements, Finite Differences) require resolving dense linear systems on fixed meshes, incurring cubic computational complexity $\mathcal{O}(N^3)$ and failing to transfer across varying grid resolutions.

**Neural Operators** learn continuous operator mappings $G_\theta: \mathcal{A} \to \mathcal{U}$ that are inherently **mesh-independent** and **resolution-invariant**:
1. **Fourier Neural Operator (FNO)**: Parameterizes integral kernel operators directly in Fourier frequency space via discrete spectral convolutions $\mathcal{K}(v) = \mathcal{F}^{-1}(R \cdot \mathcal{F}(v))$. By truncating to lower Fourier modes, FNO achieves global receptive fields in $\mathcal{O}(N \log N)$ operations with zero-shot super-resolution.
2. **Deep Operator Networks (DeepONet)**: Formulates universal operator approximation via uncoupled **Branch** and **Trunk** networks: $G(u)(y) \approx \sum_{k=1}^p b_k(u) \cdot t_k(y) + b_0$. The branch encodes discrete sensor observations $u(x_i)$, while the trunk encodes arbitrary continuous query locations $y \in \mathbb{R}^d$.
3. **MeshGraphNets**: Extends operator learning to complex, irregular, moving 3D meshes (e.g. airfoils, structural mechanics, Eulerian fluid grids) via an Encode-Process-Decode message-passing graph architecture.

---

## 2. Mathematical Foundations & Governing Equations

### 2.1 Fourier Neural Operator (FNO) Architecture
Let $\Omega \subset \mathbb{R}^d$ be a bounded domain and $a \in \mathcal{A}(\Omega; \mathbb{R}^{d_{\text{in}}})$ an input function.
1. **Lifting Layer $P$**: Projects input function values locally to higher-dimensional hidden channel space:
   $$v_0(x) = P(a(x)) = \sigma(W_p a(x) + b_p), \quad P: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_v}$$
2. **Fourier Operator Layers**: Updates state $v_l$ via non-local spectral convolution and local linear bypass:
   $$v_{l+1}(x) = \sigma\left( W v_l(x) + (\mathcal{K} v_l)(x) \right)$$
   where the spectral convolution kernel is defined by:
   $$(\mathcal{K} v)(x) = \mathcal{F}^{-1}\left( R(k) \cdot (\mathcal{F} v)(k) \right)(x)$$
   Truncating to $K_{\text{max}}$ lower Fourier modes, $R \in \mathbb{C}^{K_{\text{max}} \times d_v \times d_v}$ acts as a learnable frequency filter.
3. **Projection Layer $Q$**: Projects final representation back to target physical field:
   $$u(x) = Q(v_L(x)) = W_q v_L(x) + b_q, \quad Q: \mathbb{R}^{d_v} \to \mathbb{R}^{d_{\text{out}}}$$

### 2.2 DeepONet (Universal Operator Approximation)
Based on the Chen & Chen (1995) universal approximation theorem for non-linear operators, DeepONet decouples input function encoding from output query spatial coordinates:
$$G(u)(y) \approx \sum_{k=1}^p b_k(u(x_1), u(x_2), \dots, u(x_m)) \cdot t_k(y) + b_0$$
- **Branch Network $\mathbf{b}(u) \in \mathbb{R}^p$**: Encodes function $u$ evaluated at $m$ fixed sensor points $\{x_i\}_{i=1}^m$.
- **Trunk Network $\mathbf{t}(y) \in \mathbb{R}^p$**: Continuous coordinate representation evaluating basis functions at arbitrary query location $y \in \Omega$.
- **Zero-Shot Evaluation**: Querying $N_q$ points simultaneously evaluates via a single matrix product $\mathbf{b} \mathbf{t}^T + b_0 \in \mathbb{R}^{B \times N_q}$.

### 2.3 MeshGraphNets (Unstructured Mesh Operator Learning)
For physical domains discretized into irregular triangulated or tetrahedral graphs $G = (V, E)$:
1. **Edge Feature Construction**: $e_{ij} = [x_j - x_i, \|x_j - x_i\|_2, e_{ij}^{\text{type}}]$.
2. **Message Passing**:
   $$e_{ij}' = \text{MLP}_e\left([v_i, v_j, e_{ij}]\right)$$
   $$\bar{e}_i = \sum_{j \in \mathcal{N}(i)} e_{ij}'$$
   $$v_i' = \text{MLP}_v\left([v_i, \bar{e}_i]\right)$$
3. **Physical State Update**: Predicts time derivatives or coordinate displacements $\Delta v_i = \text{MLP}_d(v_i')$.

---

## 3. Tensor Memory Layout & Implementation Details

| Component | Tensor Name | Shape | Layout / Memory Semantics |
| :--- | :--- | :--- | :--- |
| **FNO Input Field** | `fno_input` | `[2, 1, 8]` | Input function $a(x)$ evaluated on $S=8$ spatial grid |
| **Lifting Weights** | `w_p` | `[4, 1]` | Channel lifting projection weights ($d_v=4$) |
| **Lifting Biases** | `b_p` | `[4, 1]` | Channel lifting projection biases |
| **Lifted Hidden Field**| `v0` | `[2, 4, 8]` | Lifted hidden field representations |
| **Spectral Kernel (Real)**| `r_real` | `[4, 4, 4]` | Real Fourier mode weights ($K=4$ modes) |
| **Spectral Kernel (Imag)**| `r_imag` | `[4, 4, 4]` | Imaginary Fourier mode weights ($K=4$ modes) |
| **Spectral Conv Field**| `k_v0` | `[2, 4, 8]` | Inverse Fourier reconstructed convolution |
| **Bypass Weights** | `w_bypass` | `[4, 4]` | Local linear residual bypass matrix |
| **Fourier Layer State**| `v1` | `[2, 4, 8]` | Post-activation Fourier layer state |
| **Projection Weights** | `w_q` | `[1, 4]` | Target output field projection weights |
| **Projection Bias** | `b_q` | `[1, 1]` | Target output field projection bias |
| **FNO Output Field** | `fno_output` | `[2, 1, 8]` | Predicted solution field $u(x)$ |
| **DeepONet Sensors** | `u_sensor` | `[2, 8]` | Input function evaluated at $m=8$ sensor locations |
| **DeepONet Query** | `y_query` | `[4, 1]` | Continuous query coordinates $y \in [0, 1]$ |
| **Branch Feature** | `branch_feat` | `[2, 8]` | Encoded branch representation $\mathbf{b}(u) \in \mathbb{R}^p$ |
| **Trunk Feature** | `trunk_feat` | `[4, 8]` | Encoded trunk basis functions $\mathbf{t}(y) \in \mathbb{R}^p$ |
| **DeepONet Output** | `deeponet_output` | `[2, 4]` | Predicted operator evaluation $G(u)(y)$ |
| **Mesh Nodes** | `v_nodes` | `[4, 4]` | Graph vertex physical features |
| **Mesh Edge Features** | `e_feat` | `[4, 2]` | Directed edge relative displacements |
| **MeshGraphNet Output**| `mgn_output` | `[4, 4]` | Updated physical state after message passing |

---

## 4. Multi-Ecosystem Implementation Matrix

| Framework | Implementation Module | Execution Model | Spectral Processing Strategy |
| :--- | :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/.../06_neural_operators_fno_deeponet_and_meshgraphnets.py` | Eager / `torch.compile` | `torch.fft.rfft` and discrete Fourier synthesis matrix parity |
| **JAX** | `jax/.../06_neural_operators_fno_deeponet_and_meshgraphnets.py` | Functional `jax.jit` | Vectorized `jnp.einsum`, `jax.nn.gelu(approximate=False)` |
| **Candle** | `candle/.../06_neural_operators_fno_deeponet_and_meshgraphnets.rs` | Zero-GC Rust | Discrete cosine/sine transform matrices, Abramowitz-Stegun GELU |
| **WGSL / CPU** | `kernels/.../06_neural_operators_fno_deeponet_and_meshgraphnets.rs` | WebGPU compute / SIMD | Tiled parallel DeepONet branch-trunk inner product compute shader |

---

## 5. Pathological Misuse & Failure Modes

1. **Non-Periodic Boundary Gibbs Ringing in FNO**: Applying standard Fourier transforms to non-periodic physical domains causes high-frequency Gibbs oscillations near domain boundaries. Mitigate by padding input domains with reflection/zero buffers (e.g. padding by 25–50% before FFT and cropping after IFFT).
2. **Trunk Feature Saturation in DeepONet**: Using bounded activations like Sigmoid in the Trunk network restricts continuous query representation capacity. For multi-scale or oscillatory solutions, use sinusoidal (`SIREN`-style) or Fourier feature trunk embeddings.
3. **High-Frequency Wave Number Truncation**: Truncating Fourier modes too aggressively ($K \ll K_{\text{Nyquist}}$) filters out turbulence, shock waves, and localized shear layers. When high-frequency features dominate, combine FNO with Wavelet Neural Operators (WNO) or multi-scale U-Net bypasses.

---

## 6. Cross-Domain Generalisation Matrix

| Scientific Domain | Operator Learning Architecture | Concrete Application |
| :--- | :--- | :--- |
| **Fluid Dynamics (Navier-Stokes)** | 2D/3D FNO | Zero-shot super-resolution velocity/vorticity simulation $1000\times$ faster than DNS |
| **Aerodynamics & Solid Mechanics** | MeshGraphNets | Stress/strain and pressure drag prediction on complex unstructured aircraft meshes |
| **Material Science & Fracture** | DeepONet | Variable load history to stress trajectory mapping across continuous query points |
| **Weather & Climate Forecasting** | Spherical FNO (SFNO) | Global atmospheric circulation modeling on the sphere ($S^2$) using spherical harmonics |

---

## 7. Verification, Parity & Empirical Benchmarks

All operators are verified against the deterministic golden reference checkpoint:
- Checkpoint: `weights/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.safetensors` (Seed 42)
- Tensors Verified: 35 tensors (anchor, FNO fields, DeepONet features, MeshGraphNet states).
- Numerical Parity:
  - FNO spectral convolution vs FFT reference: max difference $< 7.2 \times 10^{-7}$.
  - PyTorch vs JAX: max absolute difference $< 10^{-6}$.
  - Candle Test `test_06_neural_operators_fno_deeponet_and_meshgraphnets_parity`: 0 failures, max difference $< 10^{-5}$.

---

## 8. Synchronized Track Artifacts & Navigation

- PyTorch: `pytorch2/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.py`
- JAX: `jax/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.py`
- Candle: `candle/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.rs`
- Kernels: `kernels/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.rs`
- Weights: `weights/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.safetensors`
- Notebook: `notebooks/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.ipynb`
