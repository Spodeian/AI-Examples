# Sub-Byte Bitpacking, T-MAC Table Lookup GEMM, and GGUF K-Quants

While theoretical quantization research operates on uncompressed floating-point tensors simulated with discrete integer ranges, real-world deployment on CPU, edge, and mobile hardware demands physical sub-byte memory packing. Modern CPUs and GPUs lack native hardware addressing for individual 1-bit, 2-bit, 3-bit, or 4-bit memory words; bytes (`uint8`, 8 bits) are the universal atomic memory unit. Sub-byte bitpacking packs multiple low-precision weight elements into dense byte arrays, achieving genuine $2\times$ to $16\times$ RAM reductions. Furthermore, frontier systems paradigms such as **T-MAC** (Table-Lookup MAC, Wei et al. 2024) and **GGUF k-quants** (Gerganov & Kawrakow 2023) eliminate runtime floating-point dequantization entirely, executing matrix multiplication via bit-serial table lookups and hierarchical super-blocks.

---

## 1. Sub-Byte Bitpacking Mechanics & Encodings

Because standard PyTorch tensors store `int8` as a full 8-bit byte, storing a 4-bit weight in `torch.int8` saves **0% memory**. True memory compression requires bitwise packing:

### A. $2 \times \text{INT4}$ Packing (Two 4-bit Nibbles per Byte)
Given two 4-bit unsigned integers $v_0, v_1 \in [0, 15]$:

$$
\text{packed} = (v_0 \ \& \ \text{0x0F}) \mid ((v_1 \ \& \ \text{0x0F}) \ll 4)
$$

Unpacking extracts $v_0 = \text{packed} \ \& \ \text{0x0F}$ and $v_1 = (\text{packed} \gg 4) \ \& \ \text{0x0F}$.

### B. $5 \times \text{Ternary}$ (`trits`) Packing into 1 Byte
For ternary networks with weights in $\{-1, 0, +1\}$ (mapped to $\{0, 1, 2\}$):

$$
3^5 = 243 \le 255 \implies 5 \text{ ternary values fit inside a single } \text{uint8}!
$$

$$
\text{code} = t_0 + 3 \cdot t_1 + 9 \cdot t_2 + 27 \cdot t_3 + 81 \cdot t_4
$$

Decoding uses successive integer division and modulo 3:
$t_k = (c / 3^k) \pmod 3$, saving $80\%$ of memory compared to raw byte arrays.

---

## 2. T-MAC: Bit-Serial Table Lookup GEMM

In traditional quantized inference, packed low-bit weights are loaded from memory and dynamically dequantized into FP16/BF16 registers before executing standard floating-point multiply-accumulate (MAC) instructions. This wastes significant ALU compute on dequantization instructions.

**T-MAC** (Wei et al., 2024) turns this paradigm on its head: **it precomputes all possible dot-product sums of activations, transforming matrix multiplication into pure table lookups and integer additions with zero floating-point multiplications.**

### Formulation:
For a group of $g = 4$ activations $\mathbf{a} = [a_0, a_1, a_2, a_3]^T$ and $b$-bit integer weights $\mathbf{w} \in \{0, \dots, 2^b - 1\}^4$:
1. **Precompute 16-entry Activation Lookup Table (LUT):**
   $$\text{LUT}[i] = \sum_{k=0}^3 \left( (i \gg k) \ \& \ 1 \right) \cdot a_k \quad \text{for } i \in \{0, \dots, 15\}$$
   - $\text{LUT}[0] = 0$
   - $\text{LUT}[1] = a_0$
   - $\text{LUT}[3] = a_0 + a_1$
   - $\text{LUT}[15] = a_0 + a_1 + a_2 + a_3$
2. **Decompose Weights into Bit-Planes:**
   $$\mathbf{w} = \sum_{p=0}^{b-1} 2^p \cdot \mathbf{w}^{(p)} \quad \text{where } \mathbf{w}^{(p)} \in \{0, 1\}^4$$
3. **Execute GEMM via LUT Gather & Shift-Add:**
   $$\mathbf{w} \cdot \mathbf{a} = \sum_{p=0}^{b-1} 2^p \cdot \text{LUT}\left[ \text{index}\left( \mathbf{w}^{(p)} \right) \right]$$

Because each group of 4 weights forms an exact 4-bit nibble index into the 16-entry LUT, dot products execute using single-cycle register table lookups (`tbl` on ARM NEON, `_mm_shuffle_epi8` on x86 AVX2).

---

## 3. GGUF `k-quants` (`Q4_K_M`): Hierarchical Super-Blocks

In `llama.cpp` and local edge LLM runtimes, GGUF `k-quants` provide state-of-the-art compression using two-level hierarchical scaling:

### `Q4_K_M` Super-Block Layout:
- **Super-block size:** 256 weights.
- **Sub-blocks:** 8 sub-blocks of 32 weights each.
- **Super-scale $d$:** FP16 scalar representing global dynamic range: $d = \frac{\max |W|}{127}$.
- **Sub-block scale $d_k$:** 6-bit integer scale factor per sub-block: $d_k = \frac{\max |W_k|}{7 \cdot d}$.
- **Weights:** 256 4-bit quantized nibbles in $[-8, 7]$ packed into 128 bytes.

### Storage Density:
$$\text{Total Bytes} = 128 \text{ (weights)} + 8 \text{ (sub-scales)} + 2 \text{ (super-scale } d\text{)} = 138 \text{ bytes for 256 weights}$$
$$\text{Effective Bitrate} = \frac{138 \times 8}{256} = 4.3125 \text{ bits/parameter}$$

---

## 4. Arithmetic Intensity & Roofline Comparison

| Method | Arithmetic Type | Memory per Param | Decode Throughput ($B=1$) |
| :--- | :--- | :--- | :--- |
| **FP16 Baseline** | Floating-Point MAC | 16 bits (2.0 B) | $1.0\times$ (Baseline) |
| **Dequantize-to-FP16** | Dequant + FP MAC | 4 bits (0.5 B) | $2.5\times$ (ALU dequant overhead) |
| **T-MAC Table Lookup** | LUT Gather + Add | 4 bits (0.5 B) | **$4.8\times$ (Zero dequant, zero float mults)** |
| **GGUF Q4_K_M** | Fused SIMD Block MAC | 4.31 bits (0.54 B) | **$4.2\times$ (Cache-resident sub-blocks)** |

---

## 5. Failure Modes & Misuse Guardrails

1. **Unpacked Storage Footprint:** Saving low-bit models as PyTorch `int8` tensors without bitpacking saves **0% storage** on disk and in memory. Always pack into `uint8` arrays.
2. **Alignment & Padding:** Bitpacking non-multiple dimensions (e.g. 5 trits on length 47) leads to out-of-bounds corruption unless trailing elements are zero-padded to the block boundary.
3. **Table Cache Misses:** In T-MAC, keeping LUT size at $2^4 = 16$ entries ensures tables fit directly in SIMD vector registers (`xmm`/`ymm`/`v0-v31`). Expanding group size to $g=8$ ($2^8 = 256$ entries) causes L1 cache evictions and degrades speedup.

---

## 6. Cross-Framework Numerical Parity

| Framework | Implementation File | Precision Parity |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/03_...py` | Reference |
| **JAX** | `jax/14_quantisation/02_pt2e_torchao_and_bitpacking/03_...py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/02_pt2e_torchao_and_bitpacking/03_...rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/02_pt2e_torchao_and_bitpacking/03_...rs` | WGSL / CPU |

---

## 7. Production Edge Deployment Best Practices

- **CPU LLM Serving:** Use T-MAC or GGUF `Q4_K_M` to deploy 7B/13B parameter language models on edge Raspberry Pi 5 or mobile laptops at $>30\text{ tokens/sec}$.
- **Ternary Bitpacking:** For `BitNet b1.58` and TWN models, 5-trit base-3 packing provides maximum packing density ($1.58\text{ bits/weight}$) on disk and over networks.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.ipynb](../../../notebooks/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.ipynb)
- **PyTorch 2.x Module:** [03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.py](../../../pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.py)
- **Functional JAX Module:** [03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.py](../../../jax/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.py)
- **Native Rust Candle Track:** [03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.rs](../../../candle/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.rs)
- **Hardware Kernel (CPU / WGSL):** [03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.rs](../../../kernels/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.rs)
- **Golden SafeTensors:** [03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.safetensors](../../../weights/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.safetensors)

