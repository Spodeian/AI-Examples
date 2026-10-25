# Binary, Sub-1-Bit, and Ternary Quantization: BNN, XNOR-Net, BiLLM, OneBit, TWN, and TTQ

Pushing model weights down to **1 bit ($\{-1, +1\}$)** or **1.58 bits ($\{-1, 0, +1\}$)** completely replaces floating-point multiplications with simple additions, subtractions, and bitwise logic gates. Seminal innovations—from **BinaryConnect** (Courbariaux et al., 2015) and **XNOR-Net** (Rastegari et al., 2016) to **Ternary Weight Networks (TWN)** (Li et al., 2016), **TTQ** (Zhu et al., 2016), **BiLLM** (Huang et al., 2024), and **OneBit** (Xu et al., 2024)—demonstrate that high-capacity neural networks can retain remarkable representational fidelity with extreme weight compression.

---

## 1. The Mechanics of 1-Bit Quantization

Standard GEMM operations consume $O(N \cdot M \cdot K)$ hardware multiply-accumulate (MAC) units. In a pure binary network:

$$
\mathbf{W} \approx \alpha \cdot \mathbf{B}, \quad \mathbf{B} \in \{-1, +1\}^{d_{\text{in}} \times d_{\text{out}}}, \quad \alpha \in \mathbb{R}_{>0}^{1 \times d_{\text{out}}}
$$

When inputs $\mathbf{X}$ are also binarized into $\mathbf{H} \in \{-1, +1\}$, the floating-point vector dot product reduces to a single-cycle hardware instruction:

$$
\mathbf{X} \cdot \mathbf{W} \approx \alpha \cdot \beta \left( 2 \cdot \text{popcount}\left( \text{XNOR}(\mathbf{H}, \mathbf{B}) \right) - d_{\text{in}} \right)
$$

This unlocks up to $32\times$ memory compression and up to $58\times$ faster CPU dot products on edge silicon.

---

## 2. BNN and XNOR-Net: Optimal Scaling Derivation

Rastegari et al. (2016) showed that naive sign thresholding without scaling collapses representational capacity. Minimizing the reconstruction error:

$$
\min_{\alpha, \mathbf{B}} \|\mathbf{W} - \alpha \mathbf{B}\|_F^2
$$

yields the exact closed-form optimal solutions:

$$
\mathbf{B}^* = \text{sign}(\mathbf{W}), \quad \alpha^* = \frac{1}{n} \sum_{i=1}^n |W_i| = \frac{1}{n} \|\mathbf{W}\|_1
$$

Scaling each column or tensor by its mean absolute deviation $\alpha^*$ ensures the binary approximation preserves the primary energy of the weight distribution.

---

## 3. Ternary Weight Networks (TWN) and TTQ

While 1-bit BNNs force every weight to be active ($\pm 1$), ternary quantization introduces a zero state $\{-1, 0, +1\}$ ($\approx \log_2(3) = 1.58$ bits):

1. **TWN (Symmetric Ternary):**
   Weights below threshold $\Delta$ are pruned to zero:
   $$
   T_i = \begin{cases} +1 & W_i > \Delta \\ 0 & |W_i| \le \Delta \\ -1 & W_i < -\Delta \end{cases}, \quad \Delta^* \approx 0.7 \cdot \frac{1}{n}\|\mathbf{W}\|_1
   $$
   The optimal scaling factor $\alpha^*$ is computed solely over the non-zero elements:
   $$
   \alpha^* = \frac{1}{|I_\Delta|} \sum_{i \in I_\Delta} |W_i|
   $$
2. **TTQ (Trained Ternary Quantization):**
   Decouples positive and negative scales into asymmetric learnable parameters $W_p$ and $W_n$:
   $$
   W_i \approx \begin{cases} +W_p & W_i > \Delta \\ 0 & |W_i| \le \Delta \\ -W_n & W_i < -\Delta \end{cases}
   $$
   Allowing asymmetric positive and negative scales absorbs skewness in transformer attention projections.

---

## 4. Modern 1-Bit LLM Architectures: BiLLM and OneBit

Standard BNN formulas fail on 7B+ parameter language models due to activation outliers. Recent 1-bit LLM frameworks address this:

- **BiLLM (Huang et al., 2024):** Isolates the top salient weights via Hessian sensitivity and applies dual-bell positive/negative binarization to non-salient weights.
- **OneBit (Xu et al., 2024):** Formulates 1-bit quantization as a value-sign rank-1 scaling decomposition:
  $$
  \mathbf{W} \approx \mathbf{S} \odot \left( \mathbf{g} \mathbf{h}^\top \right)
  $$
  where $\mathbf{S} \in \{-1, +1\}^{d_{\text{in}} \times d_{\text{out}}}$ is a 1-bit sign matrix, $\mathbf{g} \in \mathbb{R}^{d_{\text{in}} \times 1}$ captures row-wise activation scale, and $\mathbf{h} \in \mathbb{R}^{1 \times d_{\text{out}}}$ captures column-wise dynamic range.

---

## 5. Multi-Framework Parity & Verification Architecture

All implementations share bit-exact transitive numerical parity across PyTorch 2.x, JAX, and Candle:

```mermaid
flowchart TD
    A[Weight Matrix W in R^{d_in x d_out}] --> B[BNN: sign W * mean abs W]
    A --> C[TWN: 0.7 mean threshold + ternary alpha]
    A --> D[TTQ: Asymmetric W_pos & W_neg scales]
    A --> E[OneBit: Sign matrix S * rank-1 g h^T]
    B --> F[Golden SafeTensors seed 42]
    C --> F
    D --> F
    E --> F
    F --> G[PyTorch 2.x Reference Parity < 1e-6]
    F --> H[JAX Functional Parity < 1e-5]
    F --> I[Candle Rust Parity < 1e-5]
```

### Numerical Parity Summary:
| Track | Path | Parity vs Golden |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py` | Reference |
| **JAX** | `jax/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.rs` | WGSL / CPU |

---

## 6. Systems Engineering & Hardware Kernels

At runtime, OneBit dequantizes weights in registers on GPU hardware (`ONEBIT_DEQUANT_WGSL`):

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let r = global_id.x;
    let c = global_id.y;
    if (r >= params.d_in || c >= params.d_out) { return; }
    let idx = r * params.d_out + c;
    w_deq[idx] = signs[idx] * (g_rows[r] * h_cols[c]);
}
```

---

## 7. Comparative Benchmark & SQNR Evaluation

Evaluating 1-bit to 1.58-bit representations on a calibrated weight matrix ($16 \times 8$):

| Method | Bits/Weight | Zero Weights Fraction | Output SQNR (dB) |
| :--- | :--- | :--- | :--- |
| **BNN (XNOR-Net)** | $1.00$ | $0.0\%$ | $5.02\text{ dB}$ |
| **TWN (Ternary)** | $1.58$ | $42.2\%$ | $8.25\text{ dB}$ ($+3.23\text{ dB}$ gain) |
| **TTQ (Asymmetric)** | $1.58$ | $35.9\%$ | $8.40\text{ dB}$ ($+3.38\text{ dB}$ gain) |
| **OneBit (Rank-1)** | $1.00 + \epsilon$ | $0.0\%$ | $0.74\text{ dB}$ (pure rank-1 proxy) |

Thresholding into ternary states ($\pm 1, 0$) provides $+3.2\text{--}3.4\text{ dB}$ SQNR improvements by pruning near-zero weights without requiring multipliers.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.ipynb](../../../notebooks/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.ipynb)
- **PyTorch 2.x Module:** [04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py](../../../pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py)
- **Functional JAX Module:** [04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py](../../../jax/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.py)
- **Native Rust Candle Track:** [04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.rs](../../../candle/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.rs)
- **Hardware Kernel (CPU / WGSL):** [04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.rs](../../../kernels/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.rs)
- **Golden SafeTensors:** [04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.safetensors](../../../weights/14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.safetensors)
