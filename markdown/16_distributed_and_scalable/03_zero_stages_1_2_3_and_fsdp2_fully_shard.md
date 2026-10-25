# ZeRO Stages 1, 2, 3 and FSDP2 Fully Shard

Standard Distributed Data Parallel (DDP) replicates model weights, gradients, and optimizer states redundantly across every accelerator in the cluster. For modern LLMs with tens or hundreds of billions of parameters, this redundancy causes severe out-of-memory (OOM) errors even on 80GB HBM GPUs. **ZeRO (Zero Redundancy Optimizer)** and **FSDP2 (Fully Sharded Data Parallel 2)** eliminate memory redundancies by sharding optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3/FSDP2) across data-parallel ranks, enabling trillion-parameter training with near-linear scaling efficiency.

---

## 1. Memory Bottlenecks in Foundation Model Training

In standard mixed-precision training (FP16 or BF16) using the Adam optimizer, memory consumption is dominated by static model states rather than activations:

For a model with $\Psi$ parameters:
1. **FP16 Parameters:** $2 \Psi$ bytes
2. **FP16 Gradients:** $2 \Psi$ bytes
3. **Adam Optimizer States:**
   - FP32 Master Parameters: $4 \Psi$ bytes (required for numerical stability under small gradient updates)
   - FP32 First Momentum ($m$): $4 \Psi$ bytes
   - FP32 Second Momentum ($v$): $4 \Psi$ bytes
   - Total Optimizer States: $12 \Psi$ bytes

$$\text{Total Static State Memory per GPU} = 2\Psi + 2\Psi + 12\Psi = 16\Psi \text{ bytes}$$

For a 70B parameter model, static state memory requires $16 \times 70 \times 10^9 \approx 1,120\text{ GB}$ of memory—demanding at least 14 $\times$ 80GB GPUs solely to store static parameters and optimizer states before allocating a single byte for activation memory or KV cache!

---

## 2. The ZeRO Paradigm: Progressive Memory De-Duplication

**ZeRO (Rajbhandari et al., SC 2020)** eliminates this redundancy without altering model architectures or incurring the high communication overhead of classical model parallelism:

```
                                  ZeRO Memory Hierarchy
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  ▼                                  ▼
     ZeRO-1 (P_os)                   ZeRO-2 (P_os+g)                   ZeRO-3 / FSDP2 (P_os+g+p)
Partition Optimizer States         Partition Optimizer States         Partition Optimizer States,
(4x Memory Reduction,              & Gradients (8x Reduction,         Gradients, & Model Parameters
 Identical 2S Communication)        Identical 2S Communication)       (16x Reduction, 3S Communication)
```

| Strategy | Parameters / Rank | Gradients / Rank | Optimizer States / Rank | Memory per Rank ($N_d$ ranks) | Communication Volume |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline DDP** | $2\Psi$ | $2\Psi$ | $12\Psi$ | $16\Psi$ | $2 S \beta$ (AllReduce) |
| **ZeRO-1 ($P_{os}$)** | $2\Psi$ | $2\Psi$ | $\frac{12\Psi}{N_d}$ | $4\Psi + \frac{12\Psi}{N_d}$ | $2 S \beta$ (AllReduce) |
| **ZeRO-2 ($P_{os+g}$)** | $2\Psi$ | $\frac{2\Psi}{N_d}$ | $\frac{12\Psi}{N_d}$ | $2\Psi + \frac{14\Psi}{N_d}$ | $2 S \beta$ (ReduceScatter) |
| **ZeRO-3 / FSDP2** | $\frac{2\Psi}{N_d}$ | $\frac{2\Psi}{N_d}$ | $\frac{12\Psi}{N_d}$ | **$\frac{16\Psi}{N_d}$** | $3 S \beta$ ($+50\%$ overhead) |

---

## 3. ZeRO Stage 1 (P_os): Optimizer State Partitioning

In ZeRO-1:
1. Model parameters and gradients remain replicated across all ranks ($2\Psi + 2\Psi$).
2. The $12\Psi$ bytes of FP32 optimizer states are partitioned equally across $N_d$ data-parallel ranks. Rank $r$ owns only the slice $\left[ \frac{r \Psi}{N_d}, \frac{(r+1)\Psi}{N_d} \right)$.
3. After backpropagation computes the full gradients, gradients are synchronized via standard AllReduce.
4. Each rank executes Adam only on its local parameter slice.
5. An `all_gather` collective distributes updated parameters to all ranks.

**Communication Impact:** ZeRO-1 has **zero communication overhead** compared to standard DDP ($2 S \beta$).

---

## 4. ZeRO Stage 2 (P_os+g): Gradient Partitioning

In ZeRO-2:
1. Gradients are no longer accumulated into full $2\Psi$ buffers on each rank.
2. During the backward pass, as gradients for each layer materialize, a **ReduceScatter** collective immediately reduces and shards the gradients across the data-parallel group.
3. Each rank retains only $\frac{2\Psi}{N_d}$ gradients corresponding to its assigned optimizer state partition.
4. Parameters remain full ($2\Psi$).

**Communication Impact:** ReduceScatter transfers $\frac{N_d-1}{N_d} S$ bytes, followed by AllGather of parameters ($\frac{N_d-1}{N_d} S$ bytes). Total volume is $2 \frac{N_d-1}{N_d} S \beta$, **exactly matching DDP** while cutting static memory by up to $8\times$.

---

## 5. ZeRO Stage 3 & FSDP: Parameter Partitioning & Dynamic Lifecycles

In ZeRO-3 and FSDP:
Parameters are partitioned across ranks at rest. A parameter only materializes into full unpartitioned memory immediately before it is used, and is deallocated immediately afterward:

```
Forward Pass Lifecycle:
[ Layer L Local Shard ] ──► AllGather ──► [ Full W_L ] ──► Compute x @ W_L ──► Free Full W_L
                                                                                   │
                                                                                   ▼
Backward Pass Lifecycle:                                               Proceed to Layer L+1
[ Layer L Local Shard ] ──► AllGather ──► [ Full W_L ] ──► Compute Grad ────► Free Full W_L
                                                                  │
                                                                  ▼
                                                      ReduceScatter Grad into Local Shard
```

### Communication Volume Derivation:
- Forward pass: AllGather parameters ($1.0 \times S$)
- Backward pass: AllGather parameters ($1.0 \times S$)
- Backward pass: ReduceScatter gradients ($1.0 \times S$)
- Total Communication: $3 S \beta$ ($1.5\times$ baseline DDP).

In exchange for a modest $50\%$ increase in communication volume, ZeRO-3/FSDP reduces memory consumption by up to **$16\times$**, allowing arbitrarily large models to fit across clusters.

---

## 6. PyTorch FSDP2 Architecture: Per-Parameter DTensor Sharding

Classical FSDP1 flattened entire submodules into monolithic 1D flat buffers. While memory efficient, flat buffers introduced rigid padding, complicated selective activation checkpointing, and clashed with `torch.compile`.

**FSDP2 (2024)** replaces flat buffers with native **DTensor per-parameter sharding**:
1. **Unflat Parameter Preservation:** Each weight tensor retains its native multidimensional shape ($[d_{\text{in}}, d_{\text{out}}]$) and is sharded along dimension 0 (`Shard(0)`).
2. **Explicit Resharding Hooks:** AllGather and ReduceScatter collectives are inserted cleanly as DTensor redistribution operations.
3. **Non-Blocking Copy Overlap:** Dynamic memory management overlaps the AllGather of layer $l+1$ with the matrix multiplication of layer $l$, eliminating communication stalls on fast NVLink interconnects.

---

## 7. Hardware Kernel Acceleration: Fused Sharded Adam Updates

When running ZeRO optimizer steps on local parameter partitions, memory bandwidth is saturated by reading and writing four separate FP32 tensors ($p, g, m, v$). Fused kernels combine these operations into a single streaming memory pass:

```wgsl
// WGSL Fused Sharded Adam Kernel
@compute @workgroup_size(64, 1, 1)
fn zero_adam_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.num_params) { return; }

    let grad = g[idx];
    let m_val = u.beta1 * m[idx] + (1.0 - u.beta1) * grad;
    let v_val = u.beta2 * v[idx] + (1.0 - u.beta2) * (grad * grad);

    m[idx] = m_val;
    v[idx] = v_val;

    let m_hat = m_val / u.bias_corr1;
    let v_hat = v_val / u.bias_corr2;

    p[idx] = p[idx] - u.lr * m_hat / (sqrt(v_hat) + u.eps);
}
```

---

## 8. Implementation Guidelines & Cross-Framework Parity

To maintain transitive numerical parity ($< 10^{-5}$) across PyTorch, JAX, and Candle:

1. **Bias Correction Power Operations:** Compute $\beta_1^t$ and $\beta_2^t$ using integer powers (`powi(step as i32)`) to avoid floating-point transcendental approximations.
2. **Consistent Partition Boundaries:** Slice parameter tensors along dimension 0 using exact contiguous chunking ($d_{\text{chunk}} = d_{\text{in}} / N_d$).
3. **Double-Precision Intermediate Epsilon:** Add $\epsilon = 10^{-8}$ inside the square root denominator to prevent division-by-zero during initial zero-gradient warmup steps.
