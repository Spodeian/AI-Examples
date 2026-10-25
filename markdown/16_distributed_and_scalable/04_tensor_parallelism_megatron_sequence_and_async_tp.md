# Tensor Parallelism: Megatron-LM, Sequence Parallelism, and Async TP

Tensor Parallelism (TP) shards individual weight tensors within a transformer layer across multiple accelerators in a tightly coupled interconnect domain (e.g., NVLink or NVSwitch). When individual layers or activation footprints exceed the physical memory capacity of a single GPU, or when execution latency must be minimized during inference or training, Tensor Parallelism decomposes matrix multiplications along column or row dimensions. **Megatron-LM 1D Tensor Parallelism**, **Sequence Parallelism (SP)**, and **Asynchronous Tensor Parallelism (Async TP)** form the cornerstone of modern massive-scale transformer execution.

---

## 1. Architectural Motivation: Intra-Layer Parallelism

While Data Parallelism (DDP) and ZeRO partition batches or static model states across independent nodes, each individual operation must still fit within the memory and compute bounds of a single device. When:
1. A single multi-head attention layer or MLP block exceeds the High-Bandwidth Memory (HBM) capacity of a single GPU,
2. Per-step latency must be sharply curtailed for real-time autoregressive generation, or
3. Intermediate activations during forward/backward passes dominate memory despite gradient checkpointing,

we must partition the matrix operations inside each layer across a cluster of $W$ accelerators (the Tensor Parallel group). Because intra-layer partitioning requires collective communication (AllReduce, AllGather, ReduceScatter) for every single layer in the forward and backward passes, TP is restricted to ultra-high-bandwidth interconnect fabrics ($\ge 900\text{ GB/s}$ NVLink).

---

## 2. Megatron-LM 1D Tensor Parallelism: Column and Row Partitioning

Megatron-LM (Shoeybi et al., 2019) introduced an elegant conjugate sharding formulation that requires only **two AllReduce operations per Transformer layer** (one in Self-Attention and one in the MLP block):

```
                       Megatron-LM MLP Block Formulation
                       
                    x (Replicated across TP ranks: [B, S, D])
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
           Rank 0: W1_0 [D, H/2]                 Rank 1: W1_1 [D, H/2]     (Column Parallel)
                    │                                     │
           h_0 = GELU(x @ W1_0)                  h_1 = GELU(x @ W1_1)      (No Communication!)
                    │                                     │
           Rank 0: W2_0 [H/2, D]                 Rank 1: W2_1 [H/2, D]     (Row Parallel)
                    │                                     │
              y_0 = h_0 @ W2_0                      y_1 = h_1 @ W2_1       (Partial sums)
                    └──────────────────┬──────────────────┘
                                       ▼
                             AllReduce(SUM, [B, S, D])
                                       │
                           y = y_0 + y_1 (Replicated)
```

### Column-Parallel Linear Layer
The weight matrix $W_1 \in \mathbb{R}^{D \times H}$ is split along its columns into $W = [W_1^{(0)}, W_1^{(1)}, \dots, W_1^{(r)}, \dots]$ where $W_1^{(r)} \in \mathbb{R}^{D \times \frac{H}{W}}$:
$$h^{(r)} = \sigma\left( X W_1^{(r)} \right)$$
Because activation functions like GELU or SwiGLU operate elementwise ($\sigma(a \oplus b) = \sigma(a) \oplus \sigma(b)$), the activation can be applied directly to the local column shard without communicating across ranks!

### Row-Parallel Linear Layer
The second weight matrix $W_2 \in \mathbb{R}^{H \times D}$ is partitioned along its rows into $W_2 = [W_2^{(0)}; W_2^{(1)}; \dots]$ where $W_2^{(r)} \in \mathbb{R}^{\frac{H}{W} \times D}$:
$$Y^{(r)} = h^{(r)} W_2^{(r)} = \sigma\left( X W_1^{(r)} \right) W_2^{(r)}$$
Summing the partial outputs across all ranks reproduces the exact unpartitioned output:
$$Y = \sum_{r=0}^{W-1} Y^{(r)} = \text{AllReduce-Sum}\left( Y^{(r)} \right)$$

---

## 3. Communication Minimization: Self-Attention Concurrency

In Multi-Head Attention (MHA), the query, key, and value projection matrices $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ are column-parallel sharded by heads:
$$Q^{(r)} = X W_Q^{(r)}, \quad K^{(r)} = X W_K^{(r)}, \quad V^{(r)} = X W_V^{(r)}$$
Each rank independently computes local scaled dot-product attention for its subset of heads $\frac{N_{\text{heads}}}{W}$:
$$\text{Head}^{(r)} = \text{Softmax}\left( \frac{Q^{(r)} (K^{(r)})^\top}{\sqrt{d_k}} \right) V^{(r)}$$
The attention output projection $W_O \in \mathbb{R}^{D \times D}$ is row-parallel sharded. The local partial outputs are accumulated via a single `AllReduce(SUM)`:
$$\text{Attention}(X) = \text{AllReduce}\left( \text{Head}^{(r)} W_O^{(r)} \right)$$

**Total Communication per Transformer Layer:**
- Forward Pass: 2 AllReduce collectives ($2 \times 2 \frac{W-1}{W} b \cdot s \cdot h$ bytes)
- Backward Pass: 2 AllReduce collectives on input gradients ($2 \times 2 \frac{W-1}{W} b \cdot s \cdot h$ bytes)

---

## 4. Sequence Parallelism (Korthikanti et al., 2022)

Standard Megatron-LM shards matrix multiplications, but non-tensor parallel operations—such as LayerNorm, Dropout, and residual connections—are redundantly replicated across all TP ranks. For long sequence lengths ($s \ge 8192$), activation memory in LayerNorm and Dropout exceeds weight memory.

**Sequence Parallelism (SP)** eliminates this redundancy by recognizing that LayerNorm and Dropout operate independently along the sequence dimension ($s$):

```
                        Sequence Parallelism Flow
                        
       [Sequence Sharded: s/W]           LayerNorm (Local, No Redundancy)
                  │
                  ▼
              AllGather                  Collects full sequence s along dim 1
                  │
                  ▼
       [Full Sequence: s]                Column-Parallel GEMM + Activation
                  │
                  ▼
       [Full Sequence: s]                Row-Parallel GEMM (Partial Sums)
                  │
                  ▼
            ReduceScatter                Reduces and shards sequence into s/W
                  │
                  ▼
       [Sequence Sharded: s/W]           Dropout + Residual Addition (Local)
```

Instead of an `AllReduce` after the row-parallel projection, Sequence Parallelism executes a **ReduceScatter**, producing activations sharded along the sequence length $\frac{s}{W}$. LayerNorm and Dropout execute on $\frac{s}{W}$ tokens, reducing non-TP activation memory by $W\times$. Prior to the next column-parallel GEMM, an **AllGather** reconstructs the sequence.

---

## 5. Communication Volume Equivalence

A common misconception is that Sequence Parallelism increases communication overhead. In fact, Sequence Parallelism has **identical communication volume** to Megatron TP:

Recall the ring collective decomposition:
$$\text{AllReduce}(X) = \text{ReduceScatter}(X) + \text{AllGather}(X)$$

| Operation | Megatron-LM 1D TP | Sequence Parallelism (SP) |
| :--- | :--- | :--- |
| **LayerNorm** | Replicated on full $s$ (0 comm) | Sharded on $\frac{s}{W}$ (0 comm) |
| **Input to Column GEMM** | Replicated on full $s$ (0 comm) | `AllGather` ($\frac{W-1}{W} \cdot S$ bytes) |
| **Column GEMM** | Local GEMM | Local GEMM |
| **Row GEMM** | Local GEMM | Local GEMM |
| **Output of Row GEMM** | `AllReduce` ($2 \frac{W-1}{W} \cdot S$ bytes) | `ReduceScatter` ($\frac{W-1}{W} \cdot S$ bytes) |
| **Total Forward Comm** | **$2 \frac{W-1}{W} \cdot S$ bytes** | **$2 \frac{W-1}{W} \cdot S$ bytes** |

Sequence Parallelism splits the monolithic `AllReduce` into its constituent `ReduceScatter` and `AllGather` primitives placed across layer boundaries, achieving an identical communication footprint while slashing activation memory by up to $5\times$.

---

## 6. Asynchronous Tensor Parallelism (Async TP)

Under standard TP/SP, computation and communication are serialized:
$$\text{Step Latency} = T_{\text{GEMM}} + T_{\text{Comm}}$$

**Async TP** overlaps GEMM computation with collective communication by splitting the hidden dimension or sequence length into $M$ chunks (dual-buffering):
1. **Chunk $0$:** Begin GEMM on device while prior collective completes.
2. **Chunk $1$:** Launch non-blocking AllGather / ReduceScatter on high-priority CUDA stream while GEMM executes on compute stream.

By structuring compute and communication as concurrent CUDA streams synchronized via cross-stream events (`cudaStreamWaitEvent`), communication overhead is hidden behind compute for compute-bound GEMMs ($T_{\text{exposed}} \approx \max(T_{\text{GEMM}}, T_{\text{Comm}})$).

---

## 7. Multi-Framework Implementation & Parity

To ensure mathematical precision across distributed stacks:
- **PyTorch 2.x:** Simulates TP rank execution by slicing parameter matrices into column shards $[W_1^{(0)}, \dots]$ and row shards $[W_2^{(0)}, \dots]$, computing local activations, and accumulating partials.
- **JAX:** Expresses shard-parallel operations through `jax.lax.psum` and device meshes with `P('tp')` partition specs.
- **Candle:** Implements rank iteration and fused reduction using high-performance 3D-by-2D matrix multiplication and erf-based GELU.
- **WGSL Kernel:** Fuses cross-rank reduction into a single GPU compute pass (`FUSED_TP_ROW_ALLREDUCE_WGSL`), avoiding intermediate memory allocations.

Parity verification confirms absolute maximum difference $< 10^{-6}$ across all framework outputs against golden checkpoints.

---

## 8. Failure Modes, Pathology, and Debugging

| Failure Mode | Root Cause | Observable Symptom | Mitigation / Remedy |
| :--- | :--- | :--- | :--- |
| **Activation Divergence** | Missing or asymmetric bias additions before AllReduce | Loss diverges or gradient norms oscillate | Add bias only once after row-parallel AllReduce, not inside local shards. |
| **Non-Linear Sharding Error** | Applying non-linearities across sharded dimensions | Output deviates completely from reference ($> 10^{-1}$) | Restrict column sharding to dimensions preserved by elementwise activations. |
| **Sequence Parallel Shape Mismatch** | Sequence length $s$ not evenly divisible by TP world size $W$ | `RuntimeError: cannot reshape tensor` | Pad sequence to multiple of $W \cdot 128$ before sharding. |
| **Async TP Race Condition** | Missing CUDA event synchronization between compute and comm streams | Silent numerical corruption under high GPU utilization | Ensure `stream.wait_event()` precedes buffer reuse. |
