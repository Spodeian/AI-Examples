# Memory Optimization: Selective Checkpointing, Offloading, and Grad Hooks

## 1. Mathematical Foundations & Activation Memory Scaling

In deep neural network training, memory consumption is dominated not by static model parameters or optimizer states, but by intermediate **forward activations** stored in High-Bandwidth Memory (HBM) to compute backward gradients:
1. **The Quadratic Context Wall**: In standard Transformer self-attention (Vaswani et al. 2017), storing the $B \times H \times S \times S$ attention matrix scales quadratically with sequence length $S$, quickly exhausting GPU memory during long-context training.
2. **Rematerialization / Gradient Checkpointing (Chen et al. 2016)**: Discards forward activations during the forward pass and selectively recomputes them from saved checkpoints during the backward pass. This reduces activation memory from $\mathcal{O}(N)$ to $\mathcal{O}(\sqrt{N})$ at the cost of one additional forward pass (~33% compute overhead).
3. **Selective Activation Checkpointing (Korthikanti et al. 2023 - Megatron-LM)**: Recognizes that not all operations have equal compute-to-memory trade-offs. Matrix multiplications (GEMMs) account for over 90% of FLOPs but only a fraction of activation memory, whereas element-wise operations (Dropout, Softmax, GeLU, LayerNorm) consume substantial memory while requiring negligible compute. Selective checkpointing saves only GEMM outputs and recomputes the cheap point-wise activations.

---

## 2. Checkpointing Taxonomies: Full vs Selective Checkpointing

```
Standard Eager Mode:
  [Linear 1] ──► Save h1 (Memory: D_ffn)
  [SiLU Gate] ──► Save a1 (Memory: D_ffn)
  [Linear 2] ──► Save h2 (Memory: D_model)
  Total Memory: 2 * D_ffn + D_model

Selective Activation Checkpointing:
  [Linear 1] ──► Save h1 (High FLOP GEMM output)
  [SiLU Gate] ──► DISCARD a1 (Cheap O(N) elementwise op)
  [Linear 2] ──► Save h2 (High FLOP GEMM output)
  Total Memory: D_ffn + D_model  (Saves ~50% MLP activation memory with <2% FLOP overhead!)
```

1. **Full Checkpointing**: Saves only layer input $\mathbf{x}$; during backward, re-executes all projections, attention, and feed-forward networks within the layer.
2. **Selective Checkpointing**: Saves boundary projections $\mathbf{z}$, recomputing only activations $\sigma(\mathbf{z})$ and normalizations on-the-fly.

---

## 3. Selective Activation Recomputation Calculus: Gated SiLU Block

Consider a gated feed-forward block:

$$\mathbf{z} = \mathbf{x} \mathbf{W} + \mathbf{b}, \quad \mathbf{a} = \operatorname{SiLU}(\mathbf{z}) = \mathbf{z} \odot \sigma(\mathbf{z}), \quad \mathbf{y} = \mathbf{a} \odot \mathbf{g}$$

In selective checkpointing:
- **Forward Pass**: Save pre-activation $\mathbf{z} \in \mathbb{R}^{B \times D_{\text{out}}}$ and gate $\mathbf{g}$; discard activation $\mathbf{a}$.
- **Backward Pass**: Recompute $\mathbf{a} = \mathbf{z} \odot \sigma(\mathbf{z})$ and $\sigma(\mathbf{z})$ from saved $\mathbf{z}$.

Evaluating the derivative of $\operatorname{SiLU}(z) = z \cdot (1 + e^{-z})^{-1}$:

$$\frac{d \operatorname{SiLU}(z)}{dz} = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z)) = \sigma(z) \left(1 + z(1 - \sigma(z))\right)$$

Given incoming upstream adjoint $\mathbf{g}_y = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$:

$$\mathbf{g}_a = \mathbf{g}_y \odot \mathbf{g}$$

$$\mathbf{g}_z = \mathbf{g}_a \odot \left[ \sigma(\mathbf{z}) \odot \left(1 + \mathbf{z} \odot (1 - \sigma(\mathbf{z}))\right) \right]$$

$$\nabla_{\mathbf{W}} = \mathbf{x}^T \mathbf{g}_z, \quad \nabla_{\mathbf{b}} = \sum_{i=1}^B \mathbf{g}_{z, i}, \quad \nabla_{\mathbf{x}} = \mathbf{g}_z \mathbf{W}^T, \quad \nabla_{\mathbf{g}} = \mathbf{g}_y \odot \mathbf{a}$$

Because $\sigma(\mathbf{z})$ is computed in registers and immediately consumed by $\mathbf{g}_z$, memory round-trips to HBM are completely avoided.

---

## 4. CPU and NVMe Activation Offloading

When batch sizes exceed device memory even with selective checkpointing, **activation offloading** transfers non-active layer checkpoints to host system DRAM (ZeRO-Offload, Ren et al. 2021):
1. **Asynchronous Pinned Streams**: Host memory buffers are allocated using page-locked (pinned) memory (`torch.empty(..., pin_memory=True)`), enabling asynchronous Direct Memory Access (DMA) over PCIe without CPU intervention.
2. **Double-Buffering & Prefetching**: While layer $L_k$ executes backward on the GPU, layer $L_{k-1}$ is simultaneously prefetched from host DRAM over PCIe in a dedicated CUDA copy stream:
   $$t_{\text{step}} = \max(t_{\text{compute}}(L_k), t_{\text{PCIe\_transfer}}(L_{k-1}))$$
   When $t_{\text{compute}} > t_{\text{PCIe}}$, offloading introduces zero wall-clock overhead.

---

## 5. Post-Accumulate-Grad Hooks & Overlapping Communication

In distributed training (DDP / FSDP), synchronizing gradients via `AllReduce` or `ReduceScatter` is a major latency bottleneck.
1. **The Eager Gradient Accumulation Trap**: Waiting until the entire backward pass finishes before initiating all-reduce serializes computation and network communication.
2. **`register_post_accumulate_grad_hook`**: Attaches a callback directly to an individual parameter tensor. The moment autograd finishes computing $\nabla_{\mathbf{W}_k}$, the hook fires:
   - Queues asynchronous non-blocking collective communication (`dist.all_reduce`) on an independent NCCL communication stream.
   - Immediately executes local optimizer updates (e.g. AdamW momentum update $\mathbf{m}_{\text{new}} = \beta \mathbf{m} + (1-\beta)\nabla_{\mathbf{W}}$).
   - Frees gradient tensor memory ahead of time, dramatically reducing peak memory usage.

---

## 6. Numerical Invariants and Parity Contracts

- **Mathematical Invariance**: Recomputed activations must be numerically identical to retained activations down to IEEE-754 precision:
  $$\max |\mathbf{y}_{\text{recomputed}} - \mathbf{y}_{\text{standard}}| < 10^{-6}$$
- **VJP Gradient Exactness**: Adjoint gradients $\nabla_{\mathbf{W}}, \nabla_{\mathbf{b}}, \nabla_{\mathbf{x}}$ derived from recomputed activations must maintain transitive parity across PyTorch 2.x, JAX, and Candle ($< 10^{-5}$).

---

## 7. PyTorch 2.x, JAX, and Candle Architectural Parity

| Feature | PyTorch 2.x | JAX | Rust Candle |
| :--- | :--- | :--- | :--- |
| **Selective Rematerialization** | `torch.utils.checkpoint.checkpoint` with policy | `jax.checkpoint` / `jax.remat` with `policy` | Explicit recomputation structs in Rust |
| **Offload Primitive** | `torch.autograd.graph.save_on_cpu` | Host callback / `jax.experimental.maps` | Explicit CPU-device tensor copy |
| **Post-Grad Hooks** | `param.register_post_accumulate_grad_hook` | Custom autodiff `jax.custom_vjp` hook | Struct callback / Arc drop trigger |
| **Memory Profiling** | `torch.cuda.memory_summary()` | `jax.profiler.start_trace()` | Custom allocator / OS memory tracking |

---

## 8. Production Diagnostics, Memory Profiling, and Failure Modes

1. **Tracking Fragmented Memory**:
   ```python
   print(torch.cuda.memory_summary(abbreviated=True))
   ```
   Differentiates allocated memory (active tensors) from reserved memory (cached by the caching allocator).
2. **RNG State Desynchronization**: If checkpointed sections contain stochastic operators (Dropout), random number generator states must be preserved via `preserve_rng_state=True` to ensure backward recomputations use identical dropout masks.
3. **In-place Mutation Hazards**: Mutating a saved tensor between the forward and backward pass invalidates recomputation, triggering `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.
