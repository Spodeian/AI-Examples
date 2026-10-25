# Neuromorphic Computing: Spiking Neural Networks (SNNs), LIF Dynamics, Surrogate Gradients, and Capsules

## 1. Executive Summary & Architectural Overview

Traditional artificial neural networks transmit continuous, rate-coded activation values synchronously at every layer. In contrast, biological nervous systems operate asynchronously through sparse binary action potentials (spikes) and structured spatial groupings (cortical columns).

This topic implements and bridges two bio-inspired architectural paradigms:
1. **Spiking Neural Networks & Leaky Integrate-and-Fire (LIF)**: Neuromorphic dynamics where each neuron integrates synaptic input current into a decaying membrane potential $V(t)$, emitting a discrete binary spike $S(t) \in \{0, 1\}$ upon crossing a threshold $V_{\text{th}}$, followed by an instantaneous reset.
2. **Surrogate Gradient Optimization**: Overcoming the zero-gradient pathology of the non-differentiable Heaviside step function by substituting continuous surrogate derivatives (such as ATan or fast-sigmoid) during backpropagation through time (BPTT).
3. **Capsule Networks with Dynamic Routing**: Replacing scalar neurons with vector-valued capsule activations that encode both entity presence (activation vector norm $\|v\| \in [0, 1)$) and equivariant instantiation parameters (vector orientation), routed via iterative agreement.

---

## 2. Mathematical Foundations & Governing Equations

### 2.1 Discrete-Time LIF Dynamics with Hard Reset
For discrete time step $t$ with membrane decay factor $\beta = \exp(-\Delta t / \tau_{\text{mem}}) \in (0, 1)$:

$$V[t] = \beta \cdot V[t-1] \odot (1 - S[t-1]) + I[t]$$

$$S[t] = \Theta(V[t] - V_{\text{th}}) = \begin{cases} 1 & \text{if } V[t] \ge V_{\text{th}} \\ 0 & \text{otherwise} \end{cases}$$

Here, $(1 - S[t-1])$ enforces hard reset to zero voltage immediately after a spike discharge.

### 2.2 Surrogate Gradient Relaxation
While the Heaviside step $\Theta(x)$ has Dirac delta derivative $\Theta'(x) = \delta(x)$ (zero almost everywhere), surrogate gradient methods replace $\frac{\partial S}{\partial V}$ during the backward pass with a continuous relaxation:

$$\frac{\partial S}{\partial V} \approx \sigma'(V - V_{\text{th}}) = \frac{1}{2 \left(1 + \left(\frac{\pi}{2} \alpha (V - V_{\text{th}})\right)^2\right)}$$

### 2.3 Capsule Dynamic Routing & Squashing
Given primary capsule outputs transformed by weight tensors into voting predictions $\hat{u}_{j|i} \in \mathbb{R}^{D_{\text{out}}}$:

$$c_{ij} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}, \quad s_j = \sum_i c_{ij} \hat{u}_{j|i}$$

Vector squashing maps total input $s_j$ to bounded orientation-preserving vector $v_j$:

$$v_j = \text{squash}(s_j) = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \frac{s_j}{\|s_j\|_2}$$

Agreement routing updates scalar coupling logits: $b_{ij} \leftarrow b_{ij} + \hat{u}_{j|i} \cdot v_j$.

---

## 3. Tensor Memory Layout & Implementation Details

| Component | Tensor Name | Shape | Layout / Memory Semantics |
| :--- | :--- | :--- | :--- |
| **Synaptic Currents** | `currents` | `[T, B, D]` | Time-major sequence of input currents |
| **Membrane Potential**| `lif_voltages` | `[T, B, D]` | Subthreshold continuous voltage trajectory |
| **Discrete Spikes** | `lif_spikes` | `[T, B, D]` | Binary float32 activation events $\{0.0, 1.0\}$ |
| **Capsule Votes** | `u_hat` | `[B, N_in, N_out, D_out]` | High-dimensional spatial capsule predictions |
| **Routing Logits** | `b_logits` | `[B, N_in, N_out]` | Unnormalized dynamic coupling coefficients |
| **Output Capsules** | `v_capsules` | `[B, N_out, D_out]` | Squashed pose/presence representation |

---

## 4. Multi-Ecosystem Implementation Matrix

| Framework | Implementation Module | Execution Model | Memory Strategy |
| :--- | :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/.../03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.py` | Eager / `torch.compile` | Sequential step loop with in-place reset |
| **JAX** | `jax/.../03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.py` | Functional `jax.jit` / `lax.scan` | Pure state transformations |
| **Candle** | `candle/.../03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.rs` | Rust zero-cost safety | Stacked temporal tensors, broadcasted norm |
| **WGSL / CPU** | `kernels/.../03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.rs` | WebGPU compute / SIMD | Fused temporal step workgroups |

---

## 5. Pathological Misuse & Failure Modes

1. **Quiescence vs. Epileptic Spiking**: If threshold $V_{\text{th}}$ is set too high relative to input current variance, the network emits zero spikes, choking downstream computation. If $V_{\text{th}}$ is too low, all neurons spike at every step, degrading into an inefficient binary channel.
2. **Dynamic Routing Iteration Saturation**: Increasing routing iterations beyond $R=3$ often causes over-clustering and vanishing gradients through the iterative loop without improving classification accuracy.
3. **Surrogate Scale Hyperparameter Sensitivity**: A surrogate width $\alpha$ that is too narrow leads to vanishing gradients; one that is too broad allows spurious gradient leakage across subthreshold voltages.

---

## 6. Cross-Domain Generalisation Matrix

| Domain | Spiking / Capsule Primitive | Operational Advantage |
| :--- | :--- | :--- |
| **Event-Based Neuromorphic Vision** | LIF with temporal BPTT | Microsecond temporal resolution with DVS silicon retina |
| **Ultra-Low-Power Edge Hardware** | Binary spike events | Sparse accumulate-only (AC) operations instead of MAC |
| **Part-Whole Hierarchical Vision** | Capsule dynamic routing | Viewpoint equivariance without pooling data destruction |
| **Brain-Computer Interfaces (BCI)** | Recurrent LIF dynamics | Decoding asynchronous neural motor cortex spike trains |

---

## 7. Verification, Parity & Empirical Benchmarks

All operators are verified against the deterministic golden reference checkpoint:
- `weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.safetensors` (Seed 42)
- Max numerical discrepancy across PyTorch, JAX, and Candle: $\Delta < 1.0 \times 10^{-6}$.
- Candle test `test_41_spiking_neural_networks_lif_surrogate_gradients_and_capsules_parity` passed with 0 failures.

---

## 8. Synchronized Track Artifacts & Navigation

- PyTorch: `pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.py`
- JAX: `jax/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.py`
- Candle: `candle/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.rs`
- Kernels: `kernels/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.rs`
- Weights: `weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.safetensors`
- Notebook: `notebooks/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.ipynb`
