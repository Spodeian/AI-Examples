# KV Cache Engineering: PagedAttention, RadixAttention, H2O, SnapKV, and YOCO

In autoregressive large language model serving, autoregressive decoding is strictly memory-bandwidth bound rather than compute-bound. Generating each token requires reloading all previous key-value states across all layers and attention heads from high-bandwidth memory (HBM). Without specialized management, contiguous memory allocation induces catastrophic fragmentation ($60\%-80\%$ wasted VRAM), prevents long-context processing, and bounds concurrent batch size. This topic covers the complete modern stack of KV cache optimizations: virtual memory pagination (**PagedAttention**), prefix trie caching (**RadixAttention**), dynamic eviction (**H2O**), observation-window prompt compression (**SnapKV**), and structural cross-decoder caching (**YOCO**).

---

## 1. KV Cache Bottlenecks & Modern Serving Systems

For a model with $L$ layers, hidden dimension $d$, $H_{\text{kv}}$ key-value heads, head dimension $d_k$, sequence length $N$, and batch size $B$, the KV cache memory footprint in 16-bit precision is:

$$\text{Memory}_{\text{KV}} = 2 \times 2 \times B \times L \times H_{\text{kv}} \times d_k \times N \text{ bytes}$$

For Llama-3-70B ($L=80, H_{\text{kv}}=8, d_k=128$) at sequence length $N = 8192$ and batch size $B = 16$:
$$\text{Memory}_{\text{KV}} = 4 \times 16 \times 80 \times 8 \times 128 \times 8192 \approx 42.95 \text{ GB}$$

| Optimization Technique | Primary Bottleneck Solved | Mechanism | VRAM Savings / Throughput |
| :--- | :--- | :--- | :--- |
| **PagedAttention (vLLM)** | Internal/external memory fragmentation | Virtual memory blocks & page tables | Cuts fragmentation from $70\%$ to $<4\%$ |
| **RadixAttention (SGLang)** | Redundant prompt prefill computation | Radix tree prefix matching & LRU reuse | Eliminates $80\%-95\%$ of multi-turn prefill |
| **H2O (Heavy-Hitter Oracle)** | Linear $O(N)$ growth during generation | Recency window + cumulative attention top-$k$ | Bounds KV cache to fixed size $W + H$ |
| **SnapKV** | Long-context prompt storage overhead | Observation window pooling for key features | $5\times-10\times$ prompt KV compression |
| **YOCO (You Only Cache Once)** | Multi-layer $L \times \text{KV}$ redundant storage | Self-decoder + Cross-decoder global KV | $2\times$ memory reduction, $L \to 1$ cache |

---

## 2. PagedAttention & Virtual Memory Block Management (vLLM)

In standard PyTorch serving, memory for maximum sequence length is pre-allocated contiguously, causing two catastrophic failure modes:
1. **Internal fragmentation**: Reserved slots for tokens yet to be generated sit idle.
2. **External fragmentation**: Variable request arrival/departure times fragment physical HBM into unallocatable chunks.

**PagedAttention (Kwon et al., SOSP 2023)** adapts OS virtual memory to attention. Logical KV sequences are partitioned into fixed-size physical blocks (e.g. `block_size = 16` tokens):

```
Logical KV Space (Sequence b):
[ Token 0 ... 3 ] ──► Logical Block 0 ──► Page Table ──► Physical Block 4
[ Token 4 ... 7 ] ──► Logical Block 1 ──► Page Table ──► Physical Block 12
[ Token 8 ... 9 ] ──► Logical Block 2 ──► Page Table ──► Physical Block 7 (partially filled)
```

### Mathematical Mapping
For token index $t \in [0, N-1]$:
$$\text{logical\_block} = \lfloor t / B_s \rfloor, \quad \text{offset} = t \pmod{B_s}$$
$$\text{phys\_block} = \text{block\_table}[b, \text{logical\_block}]$$
$$K[b, t] = \text{physical\_pool}[\text{phys\_block}, \text{offset}]$$

Paged attention computes non-contiguous dot-products directly within GPU shared memory, eliminating contiguous re-allocation.

---

## 3. RadixAttention & Prefix KV Caching (SGLang)

Many real-world LLM workloads exhibit massive prompt overlap: multi-turn chat conversations, few-shot exemplars, agentic loops, and document QA. Traditional systems recompute KV states for identical prefixes on every invocation.

**RadixAttention (Zheng et al., 2023)** maintains a dynamic radix tree (compact prefix tree) of token sequences mapped to resident physical KV blocks in HBM:
- When a new request arrives, SGLang traverses the radix tree to find the longest common prefix.
- Cached KV blocks are directly retained and pinned into the request's page table.
- Only newly appended suffix tokens undergo prefill computation.
- Unreferenced blocks are evicted using an LRU cache strategy during memory pressure.

---

## 4. Heavy-Hitter Oracle (H2O) Dynamic Eviction

During extended generation, maintaining thousands of historical KV tokens degrades memory capacity. Zhang et al. (NeurIPS 2023) discovered the **Heavy-Hitter phenomenon**: attention distributions in autoregressive generation are intensely sparse; a tiny subset ($<10\%$) of "heavy-hitter" (H2) tokens accumulate $>90\%$ of total attention mass across steps.

### H2O Retention Rule
Given cumulative attention score vector $S_t = \sum_{\tau=1}^t A_{:, \tau} \in \mathbb{R}^N$:
$$\text{Keep}(i) = \begin{cases} 1 & \text{if } i \ge N - W \text{ (local recency window)} \\ 1 & \text{if } i \in \text{top-}H(S_t[0 : N - W]) \text{ (heavy hitters)} \\ 0 & \text{otherwise (evict from cache)} \end{cases}$$

Total KV cache size is strictly bounded at $K = W + H$, enabling infinite-length generation within constant VRAM.

---

## 5. SnapKV & Observation Window Prefix Compression

While H2O operates dynamically during generation, **SnapKV (Li et al., 2024)** compresses massive prompt KV caches (e.g. 32k–128k context) instantly at prefill completion.

### Observation Window Pooling
SnapKV recognizes that the final $L_{\text{obs}}$ tokens of the prompt (the observation window) query precisely the critical context required for downstream generation:
$$I_j = \frac{1}{L_{\text{obs}}} \sum_{i = N_{\text{prompt}} - L_{\text{obs}}}^{N_{\text{prompt}}} A_{i, j}, \quad \forall j < N_{\text{prompt}} - L_{\text{obs}}$$

Selecting the top-$k$ tokens according to pooled importance $I_j$ compresses the prompt KV cache by $5\times - 10\times$ with zero retraining and $<0.1$ perplexity degradation.

---

## 6. You Only Cache Once (YOCO): Cross-Decoder Global KV

Standard transformers repeat $K$ and $V$ projections and cache maintenance across all $L$ layers ($L \times \text{KV}$). **YOCO (Sun et al., 2024)** bifurcates the model architecture:
1. **Self-Decoder (Layers $1 \dots L/2$)**: Evaluates local sliding-window self-attention without persistent global caching.
2. **Intermediate Global KV Generation**: Computes a single global key-value representation $(K_{\text{global}}, V_{\text{global}})$ once at midpoint layer $L/2$.
3. **Cross-Decoder (Layers $L/2 + 1 \dots L$)**: Subsequent layers execute cross-attention against the single global KV cache:
   $$h^{(l)} = h^{(l-1)} + \text{softmax}\left(\frac{h^{(l-1)} W_Q^{(l)} K_{\text{global}}^\top}{\sqrt{d_k}}\right) V_{\text{global}} W_O^{(l)}$$

YOCO achieves exact $2\times$ KV memory reduction, slashes prefill-to-decode memory transfers, and matches standard transformer scaling laws.

---

## 7. Multi-Framework Parity & Verification (PyTorch, JAX, Candle)

All five KV cache architectures are validated against deterministic seed-42 golden tensors:

```rust
// Candle PagedAttention verification
let (p_out, p_weights) = topic::paged_kv_gather_and_attend(
    &w["query"], &w["key_blocks"], &w["value_blocks"], &w["block_table"], 10, 4
)?;
common::assert_max_diff(&p_out, &w["paged_out"], 1e-5)?;

// Candle YOCO forward verification
let y_out = topic::yoco_cross_layer_forward(&w["h_mid"], &w["k_shared"], &w["v_shared"], &w["w_q"], &w["w_out"])?;
common::assert_max_diff(&y_out, &w["yoco_out"], 1e-5)?;
```

Maximum absolute error across all tracks is bounded below $10^{-5}$.

---

## 8. Production Serving Engineering: Throughput, Fragmentation & Latency

1. **Optimal Block Size Selection**: In PagedAttention, small block sizes ($B_s = 8$ or $16$) minimize internal fragmentation at sequence tail, while larger blocks ($B_s = 32$) maximize GPU memory bus saturation.
2. **Combine Radix + H2O**: In multi-turn systems, prefix blocks are locked in the Radix tree, while dynamic turn generation is bounded via H2O eviction.
3. **Observation Window Tuning**: For SnapKV, setting $L_{\text{obs}} = 32$ tokens with budget $B = 0.2 \times N_{\text{prompt}}$ captures all positional sinks and semantic pivots.
4. **Hardware Affinity**: PagedAttention memory layouts should align block memory allocations to 128-byte cache lines to prevent memory bank conflicts on NVIDIA Hopper/Blackwell and AMD CDNA architectures.
