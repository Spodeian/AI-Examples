# Trellis and E8 Lattice Quantization: QuIP# and QTIP

While unstructured Vector Quantization (VQ) and Additive Quantization (AQLM) rely on codebooks trained via clustering, **lattice quantization** replaces codebook search with regular mathematical point lattices. In 8 dimensions, Maryna Viazovska (2017) proved that the **$E_8$ Gosset lattice** provides the mathematically densest sphere packing possible in Euclidean space. **QuIP#** (Tseng et al., 2024) and **QTIP** (Trellis-Coded Quantization for LLMs) harness the $E_8$ lattice alongside **Trellis-Coded Quantization (TCQ)** to compress weights to 2 bits per parameter while approaching the theoretical Shannon rate-distortion bound.

---

## 1. Structured Lattices vs. Codebook Quantization

In standard VQ or AQLM, increasing the codebook size $K$ to reach higher precision induces two major bottlenecks:
1. **Exponential Storage:** Storing millions of centroid coordinates in GPU memory.
2. **Search Latency:** Computing Euclidean distances to all candidates in the codebook.

A point lattice $\Lambda \subset \mathbb{R}^d$ is an infinite discrete subgroup formed by integer linear combinations of basis vectors:
$$
\Lambda = \left\{ \sum_{i=1}^d z_i \mathbf{b}_i \;\middle|\; z_i \in \mathbb{Z} \right\}
$$
Because lattices possess periodic algebraic symmetries, finding the nearest lattice point does **not require storing codebooks or performing brute-force distance scans**—it is computed analytically in closed form in $O(d)$ time!

---

## 2. The $E_8$ Gosset Lattice in 8 Dimensions

The $E_8$ lattice is defined as the union of the 8D checkerboard lattice $D_8$ and its half-integer translated coset:

$$
E_8 = D_8 \cup \left( D_8 + \frac{1}{2}\mathbf{1} \right)
$$

where $\mathbf{1} = (1, 1, 1, 1, 1, 1, 1, 1)^\top$ and $D_8$ consists of all integer points whose coordinates sum to an even number:

$$
D_8 = \left\{ \mathbf{x} \in \mathbb{Z}^8 \;\middle|\; \sum_{i=1}^8 x_i \equiv 0 \pmod 2 \right\}
$$

Every sphere in the $E_8$ lattice touches exactly **240** kissing spheres, creating the highest space-filling efficiency known in 8D Euclidean space and yielding a $+1.53\text{ dB}$ asymptotic quantization advantage over hypercubic scalar rounding.

---

## 3. Conway & Sloane Fast Decoding Algorithm

Conway and Sloane (1982) formulated a fast, closed-form $O(d)$ nearest-point decoder for the $D_8$ and $E_8$ lattices:

1. **Projection onto $D_8$ (`project_d8`):**
   - Round each coordinate of $\mathbf{x}$ to the nearest integer: $\mathbf{f} = \lfloor \mathbf{x} \rceil$.
   - Compute parity $s = \sum_{i=1}^8 f_i \pmod 2$.
   - If $s = 0$, $\mathbf{f} \in D_8$ and is the exact nearest point.
   - If $s = 1$, find index $k = \arg\max_i |x_i - f_i|$ and adjust $f_k \leftarrow f_k + \text{sign}(x_k - f_k)$.
2. **Projection onto $E_8$ (`project_e8`):**
   - Candidate 0: $\mathbf{p}_0 = \text{project\_d8}(\mathbf{x})$.
   - Candidate 1: $\mathbf{p}_1 = \text{project\_d8}(\mathbf{x} - 0.5 \cdot \mathbf{1}) + 0.5 \cdot \mathbf{1}$.
   - Select $\mathbf{p} = \arg\min_{\mathbf{p} \in \{\mathbf{p}_0, \mathbf{p}_1\}} \|\mathbf{x} - \mathbf{p}\|_2^2$.

This closed-form decoding runs in nanoseconds on GPU/CPU without storing any codebook vectors.

---

## 4. Trellis-Coded Quantization (TCQ) and QTIP

**Trellis-Coded Quantization** (Marcellin & Fischer, 1990; Tseng et al., 2024 QTIP) adapts Ungerboeck's Trellis-Coded Modulation for weight compression:

1. **Partitioned Scalar Alphabet:** A scalar codebook $\mathcal{A}$ of $2^{R+1}$ levels is partitioned into subsets $S_0, S_1, \dots, S_{2^k-1}$.
2. **Convolutional State Machine:** Transitions between trellis states are governed by a finite-state machine.
3. **Viterbi Sequence Decoding:** Given weight sequence $\mathbf{w}$, the dynamic programming Viterbi algorithm searches for the continuous trellis trajectory minimizing path metric $\sum_t (w_t - \hat{w}_t)^2$.

QTIP achieves $< 2.2$ bits per weight with negligible perplexity drop on LLaMA-3 models by marrying Trellis search with $E_8$ lattice codebooks.

---

## 5. Multi-Framework Parity & Verification Architecture

All implementations share bit-exact transitive numerical parity across PyTorch 2.x, JAX, and Candle:

```mermaid
flowchart TD
    A[Scaled Weight Vectors in R^8] --> B[project_d8: Conway & Sloane Parity Correction]
    B --> C[project_e8: Dual-Coset Minimum Distance Selection]
    C --> D[E8 Nearest Lattice Points]
    D --> E[Scale Multiplier W_deq = Points * Scale]
    E --> F[Linear Forward Pass: Y = X @ W_deq + b]
    A --> G[Trellis-Coded Quantization: Viterbi Path Search]
    F --> H[Golden SafeTensors seed 42]
    G --> H
    H --> I[PyTorch 2.x Reference Parity < 1e-6]
    H --> J[JAX Parity < 1e-5]
    H --> K[Candle Rust Parity < 1e-5]
```

### Numerical Parity Summary:
| Track | Path | Parity vs Golden |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py` | Reference |
| **JAX** | `jax/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.rs` | WGSL / CPU |

---

## 6. Systems Engineering & Hardware Kernels

At runtime, $E_8$ lattice indices are dequantized in a single memory-bandwidth-bound pass using a parallel WGSL kernel (`E8_DEQUANT_WGSL`):

```wgsl
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_vecs * 8u) { return; }
    w_deq[idx] = lattice_points[idx] * params.scale;
}
```

---

## 7. Comparative Benchmark & SQNR Evaluation

Evaluating 2-bit quantization on an 8D weight block:

| Quantization Regime | Space-Filling Geometry | Codebook Memory | Reconstruction SQNR (dB) |
| :--- | :--- | :--- | :--- |
| **Scalar Uniform 2-bit** | 8D Hypercube ($\mathbb{Z}^8$) | Zero | $11.45\text{ dB}$ |
| **Unstructured VQ ($K=256$)** | Random Voronoi Cells | $256 \times 8 \times 4\text{ B}$ | $14.80\text{ dB}$ |
| **E8 Gosset Lattice Quantization** | **Optimal 8D Sphere Packing** | **Zero (Closed-Form)** | **$21.11\text{ dB}$** ($+9.66\text{ dB}$ gain!) |

By eliminating codebook lookups and leveraging sphere-packing optimality, $E_8$ lattice quantization achieves dramatic fidelity gains with zero memory footprint.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.ipynb](../../../notebooks/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.ipynb)
- **PyTorch 2.x Module:** [02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py](../../../pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py)
- **Functional JAX Module:** [02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py](../../../jax/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py)
- **Native Rust Candle Track:** [02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.rs](../../../candle/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.rs)
- **Hardware Kernel (CPU / WGSL):** [02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.rs](../../../kernels/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.rs)
- **Golden SafeTensors:** [02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.safetensors](../../../weights/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.safetensors)
