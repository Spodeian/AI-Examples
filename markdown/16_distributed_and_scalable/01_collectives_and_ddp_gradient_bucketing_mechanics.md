# Collectives and Distributed Data Parallel (DDP) Gradient Bucketing Mechanics

Distributed Data Parallel (DDP) is the foundational scaling paradigm for deep learning, replicating model weights across $W$ accelerator ranks while distributing training minibatches. Synchronizing parameters across nodes requires evaluating global element-wise gradient reductions ($\nabla W = \frac{1}{W} \sum_{r=0}^{W-1} \nabla W_r$). Rather than synchronizing parameter gradients individually (which incurs massive kernel launch and network latency overhead), modern systems employ **Ring AllReduce** and **Contiguous Gradient Bucketing** to achieve near-optimal overlap between backward backpropagation computation and inter-accelerator network communication.

---

## 1. Distributed Training Fundamentals: Parallelism Taxonomy

In distributed deep learning, scaling across clusters leverages three primary orthogonal axes:

```
                                  Parallelism Taxonomy
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   Data Parallelism             Tensor / Model Parallel            Pipeline Parallelism
 (DDP / FSDP / ZeRO)              (Megatron-LM / TP)                  (GPipe / 1F1B)
Replicates model weights,        Partitions weight tensors          Partitions network layers
shards input minibatch.          across GPU tensor cores.           sequentially across nodes.
```

Distributed Data Parallelism (DDP) is the most pervasive, offering linear computational scaling whenever per-rank batch sizes saturate accelerator compute pipelines.

---

## 2. Collective Primitives Taxonomy

Distributed communications rely on standardized collective operations defined by the MPI and NCCL standards:

| Collective Primitive | Input State (per Rank $r$) | Output State (per Rank $r$) | Communication Volume / Rank | Primary Deep Learning Use |
| :--- | :--- | :--- | :--- | :--- |
| **Broadcast** | Rank 0 has $X$, others empty | Every rank has $X$ | $S$ | Initial weight distribution |
| **AllGather** | Rank $r$ has chunk $x_r \in \mathbb{R}^{S/W}$ | Every rank has $[x_0, x_1, \dots, x_{W-1}]$ | $\frac{W-1}{W} S$ | FSDP forward weight gather |
| **ReduceScatter** | Rank $r$ has full tensor $X_r \in \mathbb{R}^S$ | Rank $r$ has $\sum_{i=0}^{W-1} X_{i}[r]$ | $\frac{W-1}{W} S$ | FSDP backward gradient reduce |
| **AllReduce** | Rank $r$ has full tensor $X_r \in \mathbb{R}^S$ | Every rank has $\sum_{i=0}^{W-1} X_i$ | **$2 \frac{W-1}{W} S$** | DDP gradient synchronization |

---

## 3. Ring AllReduce: Algorithmic Mechanics and Complexity

A naive centralized AllReduce (sending all gradients to rank 0 and broadcasting back) requires $\mathcal{O}(W \cdot S)$ traffic through node 0, causing network interface saturation.

The **Ring AllReduce algorithm** arranges $W$ ranks in a virtual ring ($0 \to 1 \to \dots \to W-1 \to 0$). The input tensor of size $S$ is partitioned into $W$ equal chunks ($S / W$ bytes each).

```
Rank 0:  [ Chunk 0 | Chunk 1 | Chunk 2 | Chunk 3 ] ──► Sends Chunk 0 to Rank 1
           ▲                                              │
           │                                              ▼
Rank 3:  [ Chunk 0 | Chunk 1 | Chunk 2 | Chunk 3 ]     Rank 1:  [ Chunk 0 | Chunk 1 | Chunk 2 | Chunk 3 ]
           ▲                                              │
           │                                              ▼
         Sends Chunk 2 to Rank 3 ◄── Rank 2:  [ Chunk 0 | Chunk 1 | Chunk 2 | Chunk 3 ]
```

### Phase 1: Scatter-Reduce ($W - 1$ steps)
In each step $s \in [0, W - 1)$:
- Rank $r$ transmits chunk $(r - s) \pmod W$ to rank $(r + 1) \pmod W$.
- Concurrently, rank $r$ receives chunk $(r - s - 1) \pmod W$ from rank $(r - 1) \pmod W$ and accumulates it into its local buffer.
- After $W - 1$ steps, each rank holds the fully reduced sum of exactly one chunk of size $S / W$.

### Phase 2: AllGather ($W - 1$ steps)
In each step $s \in [0, W - 1)$:
- Rank $r$ transmits its fully reduced chunk $(r - s + 1) \pmod W$ around the ring.
- Concurrently, rank $r$ receives a reduced chunk and replaces its corresponding local buffer.
- After $W - 1$ steps, every rank holds the complete, fully reduced tensor of size $S$.

### Analytical Communication Cost Model
Using the standard $\alpha$-$\beta$ network model ($\alpha$ = latency per transfer, $\beta$ = transfer time per byte):
$$T_{\text{ring}} = \underbrace{(W - 1)\left(\alpha + \frac{S}{W}\beta\right)}_{\text{Scatter-Reduce}} + \underbrace{(W - 1)\left(\alpha + \frac{S}{W}\beta\right)}_{\text{AllGather}} = 2(W - 1)\alpha + 2 \frac{W - 1}{W} S \beta$$

**Key Asymptotic Property:**
As $W \to \infty$, the bandwidth term approaches:
$$\lim_{W \to \infty} 2 \frac{W - 1}{W} S \beta = 2 S \beta$$
The total data transferred per rank is strictly **independent of the number of ranks** $W$!

---

## 4. PyTorch DDP Architecture: Autograd Hooks & Streams

To maximize throughput, PyTorch DDP overlaps gradient communication with backward computation:

```
Main Compute Stream:   [ Backward Layer L ] ──► [ Backward Layer L-1 ] ──► [ Backward Layer 1 ]
                                 │                         │
                                 ▼ (Hook Fires)            ▼ (Hook Fires)
                        Copy grad to Bucket 1     Copy grad to Bucket 1 (Full!)
                                                           │
                                                           ▼ (Event Record)
DDP Comm Stream:                                  [ Async Ring AllReduce Bucket 1 ] ──► ...
```

1. **Parameter Bucket Allocation:** Parameters are packed into flat contiguous 1D buffers (buckets).
2. **Reverse Autograd Order:** During backpropagation, layers compute gradients from output layer $L$ to input layer 1. Buckets are ordered matching this reverse traversal.
3. **Autograd Post-Accumulate Hooks:** Each parameter registers an autograd hook. When $\nabla w_i$ is computed, it is immediately copied into its assigned contiguous bucket.
4. **Asynchronous Stream Dispatch:** Once all parameters assigned to a bucket have materialized their gradients, an asynchronous `all_reduce` collective is enqueued on a dedicated NCCL CUDA stream.

---

## 5. Gradient Bucketing Mechanics: Cap Trade-Offs

Why bucket parameters into chunks of 25MB (`bucket_cap_mb = 25`)?

```
Individual Collective per Tensor:
[ AllReduce w1 ] -> latency a -> [ AllReduce w2 ] -> latency a -> ... (High overhead!)

Bucketed Collective:
[ Packed Bucket 1 (25 MB) ─────────────────────────── AllReduce ] -> (Optimal bandwidth saturation!)
```

| Bucket Capacity | Latency Overhead | Network Bandwidth Saturation | Overlap Opportunity | Overall Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Too Small ($< 1\text{ MB}$)** | High ($\sum \alpha$ dominates) | Poor (packet headers dominate) | High | Poor |
| **Optimal ($25\text{ MB}$)** | Low ($\le 40$ collectives / 1B params) | **Near-Line-Rate ($>95\%$ RoCE/Infiniband)** | **Optimal** | **Maximum** |
| **Too Large ($> 500\text{ MB}$)** | Very Low | Near-Line-Rate | Zero (must wait for full backward) | Poor (sequential stall) |

---

## 6. Overlap Analysis & Amdahl Scaling Limits

Let $T_{\text{bwd}}$ be the total backward computation time and $T_{\text{comm}}$ be the total gradient communication time.

- **Sequential Execution:** $T_{\text{total}} = T_{\text{bwd}} + T_{\text{comm}}$
- **Bucketed Overlapped Execution:**
  $$T_{\text{total}} = T_{\text{bwd}} + \max\left(0, T_{\text{comm}} - T_{\text{overlap}}\right)$$
Whenever network bandwidth is sufficient ($T_{\text{comm}} \le T_{\text{bwd}}$) and buckets are properly sized, **gradient synchronization communication is completely hidden behind computation**, achieving near $100\%$ scaling efficiency.

---

## 7. Hardware Kernel Acceleration: Fused Bucket Accumulation

In multi-tenant or CPU-orchestrated distributed environments, bucket accumulation is accelerated via fused vector kernels:

```wgsl
// WGSL Fused DDP Bucket Accumulate & Average Kernel
struct Uniforms {
    world_size: u32,
    bucket_size: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> rank_grads: array<f32>; // [world_size, bucket_size]
@group(0) @binding(2) var<storage, read_write> averaged_grads: array<f32>; // [bucket_size]

@compute @workgroup_size(64, 1, 1)
fn ddp_bucket_accumulate_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.bucket_size) { return; }

    var acc: f32 = 0.0;
    for (var r = 0u; r < u.world_size; r++) {
        acc += rank_grads[r * u.bucket_size + idx];
    }
    averaged_grads[idx] = acc / f32(u.world_size);
}
```

---

## 8. Implementation Guidelines & Cross-Framework Parity

To maintain transitive numerical parity ($< 10^{-5}$) across PyTorch, JAX, and Candle:

1. **Divisor Precision:** Divide by $W$ as a floating point multiplication by $(1.0 / W)$ across all tracks to ensure identical floating-point rounding.
2. **Deterministic Bucketing Boundaries:** When splitting flattened parameter vectors into buckets of size $B_{\text{cap}}$, compute $N_{\text{buckets}} = \lceil N_{\text{params}} / B_{\text{cap}} \rceil$ and handle the final partial bucket identically.
3. **Cross-Track Parity:** Verify element-wise sum and bucket-averaged outputs against golden safetensors generated from seed 42.
