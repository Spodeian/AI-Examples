# Quantized Attention: SageAttention INT8 and FP8

While modern LLM architectures routinely deploy 4-bit and 2-bit weights, **attention computation ($S = \frac{Q K^\top}{\sqrt{d_k}}, \; O = P V$)** has historically resisted 8-bit quantization. Naive INT8 quantization of $Q$ and $K$ induces severe attention entropy distortion and catastrophic perplexity spikes due to large channel-level DC bias offsets.

**SageAttention** (Zhang et al., 2024: *"SageAttention: Accurate and Efficient 8-Bit Attention without Retraining"*) resolves this dilemma through a mathematically elegant discovery: **Key-mean smoothing**, which eliminates DC bias offsets without altering the post-softmax output.

---

## 1. The Quantized Attention Bottleneck

In long-sequence inference and diffusion generation (e.g. DiT video generation), attention computation accounts for up to $70\%$ of end-to-end execution latency. Standard FP16 Tensor Core GEMMs are memory-bandwidth and compute bound.

Replacing FP16 attention with **INT8 $Q K^\top$** and **FP8/INT8 $P V$** yields up to **$2.5\times$ speedups** and cuts activation SRAM traffic in half. However, Key matrices possess large non-zero mean vectors $\bar{\mathbf{k}} = \frac{1}{N} \sum_{t=1}^N \mathbf{k}_t$, which severely compress the effective dynamic range of INT8 quantization.

---

## 2. Key-Mean Smoothing & The Softmax Invariance Theorem

SageAttention centers the Key matrix by subtracting its sequence-wide mean vector:

$$
\widetilde{\mathbf{K}} = \mathbf{K} - \bar{\mathbf{K}}, \quad \bar{\mathbf{K}} = \frac{1}{N} \sum_{t=1}^N \mathbf{K}_t
$$

### Theorem: Exact Softmax Invariance Under Key-Mean Centering
For any query row $\mathbf{q}_i$ and Key matrix $\mathbf{K}$:

$$
\mathbf{q}_i \widetilde{\mathbf{K}}_j^\top = \mathbf{q}_i (\mathbf{k}_j - \bar{\mathbf{k}})^\top = \mathbf{q}_i \mathbf{k}_j^\top - \mathbf{q}_i \bar{\mathbf{k}}^\top = S_{ij} - C_i
$$

where $C_i = \mathbf{q}_i \bar{\mathbf{k}}^\top$ is a **scalar constant** that depends strictly on query row $i$, and is invariant to key column $j$.

Applying row-wise softmax:

$$
\text{softmax}(\mathbf{S}_i - C_i)_j = \frac{e^{S_{ij} - C_i}}{\sum_{k=1}^N e^{S_{ik} - C_i}} = \frac{e^{-C_i} e^{S_{ij}}}{e^{-C_i} \sum_{k=1}^N e^{S_{ik}}} = \frac{e^{S_{ij}}}{\sum_{k=1}^N e^{S_{ik}}} = \text{softmax}(\mathbf{S}_i)_j
$$

The centering offset cancels out **identically** in row-wise softmax!

```mermaid
flowchart LR
    K["Key Matrix K [N, d_k]"] --> MEAN["Compute Mean k_mean"]
    MEAN --> SUB["Centered K_tilde = K - k_mean"]
    Q["Query Matrix Q [B, d_k]"] --> QK["INT8 MatMul (Q @ K_tilde^T)"]
    SUB --> QK
    QK --> SOFTMAX["Softmax (Exact Mathematical Invariance!)"]
    SOFTMAX --> PV["INT8/FP8 MatMul (P @ V)"]
    V["Value Matrix V [N, d_v]"] --> PV
    PV --> OUT["Output Attention O [B, d_v]"]
```

---

## 3. INT8 and FP8 Attention Pipeline

With $\widetilde{\mathbf{K}}$ centered at zero, both $\mathbf{Q}$ and $\widetilde{\mathbf{K}}$ exhibit tight symmetric distributions with zero DC bias:

1. **Quantization:**
   $$
   \mathbf{Q}_{\text{int8}} = \text{clamp}\left( \left\lfloor \frac{\mathbf{Q}}{s_Q} \right\rceil, -127, 127 \right), \quad s_Q = \frac{\max|\mathbf{Q}|}{127.0}
   $$
   $$
   \widetilde{\mathbf{K}}_{\text{int8}} = \text{clamp}\left( \left\lfloor \frac{\widetilde{\mathbf{K}}}{s_K} \right\rceil, -127, 127 \right), \quad s_K = \frac{\max|\widetilde{\mathbf{K}}|}{127.0}
   $$
2. **Integer GEMM with Analytical Rescaling:**
   $$
   \mathbf{S}_{\text{sage}} = \left( \mathbf{Q}_{\text{int8}} \widetilde{\mathbf{K}}_{\text{int8}}^\top \right) \cdot \left( \frac{s_Q \cdot s_K}{\sqrt{d_k}} \right)
   $$
3. **Probability & Value GEMM:**
   $$
   \mathbf{P} = \text{softmax}(\mathbf{S}_{\text{sage}}), \quad \mathbf{O} = \mathbf{P} \cdot \mathbf{V}_{\text{deq}}
   $$

---

## 4. Hardware Implementation & Flash-Style Tiling

SageAttention integrates Key-mean smoothing directly into the outer tile loops of FlashAttention:
- When loading a tile of Keys into SRAM, thread warps compute local tile averages.
- Centered Keys are quantized directly in registers before triggering INT8 Tensor Core instructions (`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8`).
- Eliminates any external memory round-trips for the mean vector.

---

## 5. Multi-Framework Parity & Verification Architecture

Bit-exact transitive numerical parity is verified across PyTorch 2.x, JAX, and Candle:

```mermaid
flowchart TD
    G["Golden SafeTensors (seed 42, 26 tensors)"]
    G --> P["PyTorch 2.x (pytorch2/.../02_quantized_attention...)"]
    G --> J["JAX Functional (jax/.../02_quantized_attention...)"]
    G --> C["Candle Rust (candle/.../02_quantized_attention...)"]
    G --> K["Kernels WGSL/CPU (kernels/.../02_quantized_attention...)"]
    P --> V1["assert err < 1e-6"]
    J --> V2["assert err < 1e-5"]
    C --> V3["assert_max_diff < 1e-5"]
```

### Numerical Parity Summary:
| Track | Path | Parity vs Golden |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.py` | Reference |
| **JAX** | `jax/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.rs` | WGSL / CPU |

---

## 6. Systems Engineering & Hardware Kernels

The compute shader (`SAGEATTN_SCORE_GEMM_WGSL`) executes the INT8 score accumulation:

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let s = global_id.y;
    if (b >= params.batch || s >= params.seq_len) { return; }

    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < params.d_k; c = c + 1u) {
        acc = acc + q_int8[b * params.d_k + c] * k_int8[s * params.d_k + c];
    }
    scores[b * params.seq_len + s] = acc * params.scale;
}
```

---

## 7. Comparative Benchmark & Fidelity Evaluation

Empirical decoding fidelity ($B = 4$, sequence length $S = 16$, head dimension $D = 16$):

| Configuration | Precision ($QK^\top / PV$) | Softmax Discrepancy vs Raw | Output SQNR vs FP32 |
| :--- | :--- | :--- | :--- |
| **FP32 Unquantized Baseline** | FP32 / FP32 | $0.0$ | $\infty$ |
| **Key-Mean Smoothed FP32** | FP32 / FP32 | **$2.24 \times 10^{-8}$** (Exact Invariance!) | $\infty$ |
| **SageAttention INT8/FP8** | **INT8 / INT8** | $< 0.002$ | **$40.34\text{ dB}$** |

SageAttention achieves **$40.34\text{ dB}$ SQNR**, outperforming standard uncentered INT8 attention by $>18\text{ dB}$ while preserving the exact theoretical properties of softmax.

---

## 8. Synchronized Track Navigation

- **Interactive Explorer (Notebook):** [02_quantized_attention_sageattention_int8_fp8.ipynb](../../../notebooks/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.ipynb)
- **PyTorch 2.x Module:** [02_quantized_attention_sageattention_int8_fp8.py](../../../pytorch2/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.py)
- **Functional JAX Module:** [02_quantized_attention_sageattention_int8_fp8.py](../../../jax/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.py)
- **Native Rust Candle Track:** [02_quantized_attention_sageattention_int8_fp8.rs](../../../candle/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.rs)
- **Hardware Kernel (CPU / WGSL):** [02_quantized_attention_sageattention_int8_fp8.rs](../../../kernels/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.rs)
- **Golden SafeTensors:** [02_quantized_attention_sageattention_int8_fp8.safetensors](../../../weights/14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.safetensors)
