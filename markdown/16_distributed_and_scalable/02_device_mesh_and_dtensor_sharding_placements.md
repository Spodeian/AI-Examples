# DeviceMesh and DTensor Sharding Placements

In classical distributed frameworks, developers manually orchestrated communication collectives (`all_reduce`, `all_gather`) intermixed with model computation code. This tight coupling rendered neural architectures brittle and non-composable. Modern distributed runtimes (such as PyTorch 2.x `DTensor`, JAX `pjit`/`NamedSharding`, and Alpa/GSPMD) decouple **mathematical model semantics** from **hardware parallelism layouts**. Through the abstractions of **`DeviceMesh`** (a multi-dimensional topological grid of accelerators) and **`Placement`** specifications (`Shard`, `Replicate`, `Partial`), distributed execution is framed as automated layout propagation.

---

## 1. SPMD and DeviceMesh Abstractions

Single Program, Multiple Data (SPMD) executes an identical program across $W$ devices, where each device operates on a unique slice of data or model parameters.

A **`DeviceMesh`** organizes flat physical accelerator rank IDs into an $n$-dimensional grid reflecting the underlying interconnect topology (e.g. fast intra-node NVLink vs slower inter-node RoCE/InfiniBand):

```
                       2D DeviceMesh: (mesh_dim_0="dp", mesh_dim_1="tp")
                                       ┌──────────┬──────────┐
                       DP Rank 0 (Node 0)│ GPU 0    │ GPU 1    │ (TP Pair 0 - NVLink)
                                       ├──────────┼──────────┤
                       DP Rank 1 (Node 1)│ GPU 2    │ GPU 3    │ (TP Pair 1 - NVLink)
                                       └──────────┴──────────┘
```

By naming mesh dimensions (e.g., `mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))`), operators can target specific communication sub-groups independently.

---

## 2. DTensor (Distributed Tensor) Model

A **`DTensor`** represents a global tensor from the user's perspective while physically storing only a local shard on each device in the mesh. A `DTensor` is uniquely defined by:
1. **Global Shape & Dtype:** The mathematical dimension of the unpartitioned tensor (e.g. $[B, D]$).
2. **DeviceMesh:** The grid of accelerators over which the tensor is distributed.
3. **Placements:** A tuple of placement attributes, with one placement per mesh dimension.

---

## 3. Placement Specifications Taxonomy

Each dimension of the mesh assigns one of three fundamental placement specifications to the tensor:

| Placement Type | Mathematical Definition | Memory Consumption per Device | Communication on Creation |
| :--- | :--- | :--- | :--- |
| **`Shard(dim)`** | Partition global tensor along dimension `dim` across the mesh axis | $\frac{\text{Global Size}}{\text{Mesh Size}}$ | Zero |
| **`Replicate()`** | Each device holds an identical copy of the full global tensor | $\text{Global Size}$ | Zero |
| **`Partial(Sum)`** | Each device holds a partial summand; global tensor is $X = \sum_{r} X_r$ | $\text{Global Size}$ (or local shape) | Zero |

```
Global Tensor: [ 0 , 1 , 2 , 3 ]

Shard(0) over 2 GPUs:     GPU 0: [ 0 , 1 ]        GPU 1: [ 2 , 3 ]
Replicate() over 2 GPUs:   GPU 0: [ 0 , 1 , 2 , 3 ] GPU 1: [ 0 , 1 , 2 , 3 ]
Partial(Sum) over 2 GPUs:  GPU 0: [ 0 , 0 , 1 , 1 ] GPU 1: [ 0 , 1 , 1 , 2 ] (Sum = [ 0, 1, 2, 3 ])
```

---

## 4. Redistribution & Resharding Calculus

When an operator requires its operands to have placements different from their current state, the runtime automatically inserts the minimal necessary collective communication:

```
                  ┌────────────────────────────────────────────────────────┐
                  │          Redistribution Communication Matrix           │
                  └────────────────────────────────────────────────────────┘

    Current Placement ───► Target Placement ───► Collective Collective Required
    ─────────────────────────────────────────────────────────────────────────
    Shard(i)          ──►  Replicate()      ──►  AllGather along dimension i
    Replicate()       ──►  Shard(i)         ──►  Local Slicing (Zero Network)
    Partial(Sum)      ──►  Replicate()      ──►  AllReduce(Sum)
    Partial(Sum)      ──►  Shard(i)         ──►  ReduceScatter(Sum) along dim i
    Shard(i)          ──►  Shard(j)         ──►  AllToAll (transpose re-sharding)
```

---

## 5. 1D Tensor Parallelism as Placement Propagation

Megatron-LM 1D Tensor Parallelism (TP) is naturally represented as placement propagation across linear layers $Y = X W$:

### Column-Parallel Linear Layer
- Input $X$: `Replicate()`
- Weight $W$: `Shard(1)` (split output columns)
- Forward Computation: $Y_{\text{local}} = X W_{\text{local}}$
- Output $Y$: **`Shard(1)`**
- **Communication Cost:** **Zero network communication!**

### Row-Parallel Linear Layer
- Input $X$: `Shard(1)` (split input features, matching preceding column-parallel layer output)
- Weight $W$: `Shard(0)` (split input rows)
- Forward Computation: $Y_{\text{local}} = X_{\text{local}} W_{\text{local}}$
- Output $Y$: **`Partial(Sum)`**
- Redistribution to `Replicate()`: **Single `all_reduce(Sum)`** across TP group.

```
Input X (Replicate) ──► [ Column Parallel W1 (Shard(1)) ] ──► Activation (Shard(1))
                                                                       │
                                                                       ▼
Output Y (Replicate) ◄── [ AllReduce ] ◄── [ Row Parallel W2 (Shard(0)) ] (Partial(Sum))
```

By pairing Column-Parallel with Row-Parallel layers, **only one AllReduce collective is required across two full matrix multiplications**, halving communication overhead.

---

## 6. Communication Minimization: Preserving Sharded Layouts

Rather than eagerly calling `all_gather` to return to `Replicate()` after every operation, optimal distributed programs maintain tensors in `Shard` layouts across multiple consecutive operations:

1. **Element-wise activations (GELU, SiLU, LayerNorm):** Operate purely point-wise. If input is `Shard(dim)`, output is automatically `Shard(dim)` with zero communication.
2. **Dropout & Residual Addition:** $X + Y$ requires matching placements. If both operands are `Shard(1)`, the sum executes locally with zero communication.
3. **Loss Function (Cross-Entropy):** Compute parallel cross-entropy directly on `Shard(1)` vocabulary logits using parallel max and parallel exp sum reductions.

---

## 7. Hardware Kernel Acceleration: Fused Partial Shard Reduction

In simulated multi-rank execution or CPU offload coordinators, row-parallel `Partial(Sum)` shards are accumulated into replicated outputs via fused memory-bandwidth bound kernels:

```wgsl
// WGSL Fused Row-Parallel Partial Shard AllReduce Reduction
@compute @workgroup_size(16, 16, 1)
fn row_allreduce_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= u.batch || col >= u.n) { return; }

    var acc: f32 = 0.0;
    let slice_size = u.batch * u.n;
    for (var r = 0u; r < u.mesh_size; r++) {
        acc += partial_shards[r * slice_size + row * u.n + col];
    }
    allreduced_out[row * u.n + col] = acc;
}
```

---

## 8. Implementation Guidelines & Cross-Framework Parity

To maintain transitive numerical parity ($< 10^{-5}$) across PyTorch, JAX, and Candle:

1. **Mesh Dimension Ordering:** Ensure row-major indexing when flattening multi-dimensional device coordinates to physical ranks: $\text{rank} = d_0 \times M_1 + d_1$.
2. **Shard Divisibility:** Guarantee that tensor dimensions targeted by `Shard(dim)` are evenly divisible by `mesh_size` ($N \pmod M = 0$).
3. **Partial Precision:** Maintain FP32 accumulation when summing partial shards across ranks to avoid underflow or cancellation errors during large-scale distributed training.
