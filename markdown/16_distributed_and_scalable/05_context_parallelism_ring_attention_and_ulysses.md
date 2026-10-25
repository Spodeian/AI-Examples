# Context Parallelism: Ring Attention and DeepSpeed Ulysses

Context Parallelism (CP) partitions the input sequence dimension across multiple accelerators during self-attention computation. As transformer context windows scale from 4,096 tokens to 32k, 128k, and 1M+ tokens, intermediate attention activations scale quadratically ($O(S^2)$), causing immediate out-of-memory (OOM) failures even when model parameters fit on a single device. **Ring Attention** and **DeepSpeed Ulysses** are the two foundational paradigms enabling million-token context training and inference.

---

## 1. The Long-Context Scaling Crisis

In standard Multi-Head Attention (MHA), attention scores for a sequence of length $S$, batch size $B$, and $H$ heads require:
$$\text{Attention Matrix Memory} = B \times H \times S \times S \times 2 \text{ bytes (FP16/BF16)}$$

For $B=1$, $H=32$, and $S=131,072$ (128k context):
$$\text{Memory} = 1 \times 32 \times (131,072)^2 \times 2 \approx 1,099.5 \text{ GB}$$

Over $1\text{ TB}$ of memory is required solely to materialize the pre-softmax attention logits of a single layer! While FlashAttention-2/3 avoids materializing the $S \times S$ matrix in HBM by tiling into fast SRAM, the input query, key, and value vectors ($3 \times B \times S \times D$) still exceed device memory when $S$ reaches millions of tokens.

---

## 2. Context Parallelism (CP) Paradigm & Taxonomy

Context Parallelism solves this by partitioning the sequence length $S$ across $C$ accelerators (the Context Parallel group), such that each accelerator hosts only $S_{\text{chunk}} = \frac{S}{C}$ tokens:

```
                          Context Parallel Paradigms
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
    Ring Attention                                       DeepSpeed Ulysses
- Decentralized Ring P2P                                 - Dense All-to-All Collective
- Constant Memory ($O(S/C)$)                             - Low Latency on High-Bandwidth Fabrics
- Sharded Q, circulates K, V                             - Transposes Sequence <-> Attention Heads
- Excellent for Causal Masking                           - Requires $H \ge C$ (head divisibility)
```

---

## 3. Ring Attention: Circular Blockwise P2P Attention

Introduced by Liu et al. (2023), **Ring Attention** organizes $C$ accelerators into a logical ring:
1. **Initial Placement:** Each rank $r \in [0, C-1]$ holds its local query block $Q_r$, key block $K_r$, and value block $V_r$, each of length $\frac{S}{C}$.
2. **Circular Shift:** Over $C$ steps:
   - Rank $r$ computes block attention between $Q_r$ and the currently held $K_k, V_k$ blocks.
   - Concurrently, rank $r$ sends its current $K, V$ blocks to rank $(r + 1) \pmod C$ and receives blocks from rank $(r - 1 + C) \pmod C$ via asynchronous peer-to-peer (P2P) transfers.
3. **Online Softmax Update:** Because attention is computed incrementally across blocks, each rank maintains running row maximums $m^{(s)}$ and normalizers $l^{(s)}$ (FlashAttention online softmax):
   $$m^{(s)} = \max\left( m^{(s-1)}, \text{rowmax}(S_s) \right)$$
   $$l^{(s)} = l^{(s-1)} e^{m^{(s-1)} - m^{(s)}} + \sum e^{S_s - m^{(s)}}$$
   $$O^{(s)} = O^{(s-1)} \frac{l^{(s-1)} e^{m^{(s-1)} - m^{(s)}}}{l^{(s)}} + \frac{e^{S_s - m^{(s)}}}{l^{(s)}} V_s$$

Computation and P2P communication are perfectly overlapped: the time to compute block attention ($O((S/C)^2 \cdot D)$) conceals the P2P transfer time of the key/value blocks ($O((S/C) \cdot D)$).

---

## 4. Causal Masking in Ring Topology

For autoregressive (causal) language models, token $i$ can only attend to token $j \le i$. When sequence chunks are distributed across ranks:
- **Upper Off-Diagonal Blocks ($k > r$):** Completely masked out (zero computation). In a ring of size $C$, roughly half of the block computations are skipped:
  $$\text{Active Blocks} = \frac{C (C + 1)}{2}$$
- **Diagonal Blocks ($k = r$):** Standard lower-triangular causal mask applied locally.
- **Lower Off-Diagonal Blocks ($k < r$):** Unmasked full rectangular attention.

By scheduling the ring circulation to prioritize causal non-empty blocks, Ring Attention cuts total compute and communication in half for autoregressive workloads.

---

## 5. DeepSpeed Ulysses: All-to-All Sequence-Head Transposition

**DeepSpeed Ulysses** (Jacobs et al., 2023) leverages high-performance `All-to-All` collective communication:
1. **Initial Sharding:** Inputs $Q, K, V$ are partitioned along the sequence dimension ($S/C$ tokens per rank), but contain all $H$ attention heads: `[B, H, S/C, D]`.
2. **All-to-All Transposition (Seq $\to$ Head):** An `All-to-All` collective transposes the tensor so that each rank receives the **full sequence length $S$**, but only a subset of heads $\frac{H}{C}$:
   $$\left[ B, H, \frac{S}{C}, D \right] \xrightarrow{\text{All-to-All}} \left[ B, \frac{H}{C}, S, D \right]$$
3. **Local Full Attention:** Each rank executes standard optimized attention (FlashAttention) over all $S$ tokens for its assigned $\frac{H}{C}$ heads.
4. **All-to-All Transposition (Head $\to$ Seq):** A reverse `All-to-All` transposes the attention outputs back to the sequence-sharded layout:
   $$\left[ B, \frac{H}{C}, S, D \right] \xrightarrow{\text{All-to-All}} \left[ B, H, \frac{S}{C}, D \right]$$

---

## 6. Ring Attention vs. DeepSpeed Ulysses: Tradeoff Analysis

| Metric / Dimension | Ring Attention | DeepSpeed Ulysses |
| :--- | :--- | :--- |
| **Communication Primitive** | Non-blocking Ring P2P (`send` / `recv`) | Dense `All-to-All` collective |
| **Communication Volume** | $2 \cdot \frac{C-1}{C} \cdot S \cdot D$ | $2 \cdot \frac{C-1}{C} \cdot S \cdot D$ |
| **Network Sensitivity** | Tolerant to lower bisection bandwidth; needs ring topology | Requires high all-to-all crossbar bandwidth (NVSwitch / Fat-Tree) |
| **Head Count Constraint** | None ($C$ can exceed $H$) | $H$ must be divisible by $C$ ($H \ge C$) |
| **Implementation Complexity**| Custom online softmax & async ring scheduling | Standard FlashAttention wrapped by two All-to-Alls |
| **Hybrid Synergy** | Outer inter-node Ring Attention | Inner intra-node Ulysses |

In practice, state-of-the-art long-context frameworks combine both: **DeepSpeed Ulysses** is deployed across GPUs within a single node (over NVLink), while **Ring Attention** links multiple nodes across InfiniBand clusters.

---

## 7. Multi-Framework Parity & Verification

Mathematical equivalence between monolithic attention, Ring Attention, and DeepSpeed Ulysses is maintained across all four framework backends:
- **PyTorch 2.x:** Simulates Ring Attention blockwise gathering and Ulysses All-to-All transpositions.
- **JAX:** Expresses sequence chunking and head gathering via functional array manipulations (`jnp.swapaxes`, `jnp.concatenate`).
- **Candle:** Employs `narrow` and `cat` operators alongside `candle_nn::ops::softmax_last_dim` for zero-copy blockwise attention.
- **WGSL Kernel:** Accelerates blockwise logit dot-products ($S = (Q K^\top) \cdot \text{scale}$) via cooperative workgroups.

All implementations match the golden reference outputs within an absolute error tolerance $< 10^{-5}$.

---

## 8. Failure Modes, Pathology, and Debugging

| Failure Mode | Root Cause | Observable Symptom | Mitigation / Remedy |
| :--- | :--- | :--- | :--- |
| **Ulysses Head Indivisibility** | $H \pmod C \ne 0$ | Crash on All-to-All tensor dimension mismatch | Ensure CP degree $C$ divides head count $H$, or pad dummy heads. |
| **Online Softmax Numerical Underflow** | Omitting running maximum subtraction $(m^{(s)} - m^{(s-1)})$ | `NaN` or `Inf` in attention weights after 4+ ring steps | Apply FlashAttention online rescaling formula strictly. |
| **Ring Deadlock** | Synchronous P2P send/recv ordering without pairing | Cluster hangs indefinitely at step 1 | Use non-blocking `isend`/`irecv` or alternating even/odd rank ordering. |
| **GQA Head Mismatch** | Applying Ulysses directly to Grouped-Query Attention (GQA) where $H_K < C$ | Key/Value heads cannot be partitioned across $C$ ranks | Replicate KV heads across query head groups before Ulysses All-to-All. |
