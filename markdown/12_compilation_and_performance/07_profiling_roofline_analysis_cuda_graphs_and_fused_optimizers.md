# Profiling, Roofline Analysis, CUDA Graphs, and Fused Optimizers

## 1. Mathematical Foundations & Performance Modeling Lineage

Maximizing deep learning hardware efficiency requires quantitative diagnosis of system bottlenecks across memory, compute, and host-device communication:
1. **The Roofline Performance Model (Williams et al. 2009)**: Bridges the gap between algorithmic complexity and physical hardware capacity by mapping attainable performance as a function of **Arithmetic Intensity** ($I$).
2. **CPU Launch Bottleneck & Kernel Launch Latency**: Eager execution incurs 3–10 microseconds of CPU driver dispatch latency per kernel launch. When models contain hundreds of fine-grained layers (e.g. small matrix multiplications, LayerNorm, activations), the GPU spends significant time idle waiting for the CPU work queue.
3. **CUDA Graphs (NVIDIA 2019)**: Captures sequences of GPU kernel launches, memory copies, and synchronization events into a static task DAG. The entire graph is launched onto the GPU with a single driver invocation, slashing CPU launch overhead from hundreds of microseconds to negligible host time.
4. **Fused Optimizers (Loshchilov & Hutter 2019 / Megatron-LM)**: In standard training loops, the optimizer step (AdamW) updates parameters after gradients are computed. In eager execution, this launches 4+ separate kernels per parameter tensor, bottlenecking on DRAM bandwidth. Fused multi-tensor optimizers combine moment updates, bias correction, and weight decay into a single coalesced memory pass.

---

## 2. The Williams Roofline Model: Arithmetic Intensity and Hardware Bounds

```
 Attainable Performance (TFLOPs/s)
        ▲
 P_peak ┼───────────────────────╭───────────────────────── (Compute-Bound Ceiling)
        │                      ╱
        │                     ╱  Slope = Peak Bandwidth (BW_peak)
        │                    ╱
        │                   ╱
        │                  ╱   (Memory-Bound Regime)
        │                 ╱
        │                ╱
        └───────────────┴──────────────────────────────────► Arithmetic Intensity (FLOPs/Byte)
                        I* (Ridge Point)
```

1. **Arithmetic Intensity ($I$)**: The ratio of total floating-point operations performed to total bytes transferred over the memory bus:
   $$I = \frac{\text{FLOPs}}{\text{Bytes}_{\text{read}} + \text{Bytes}_{\text{written}}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]$$
2. **Hardware Ridge Point ($I^*$)**: The operational intensity where an algorithm transitions from memory-bandwidth-bound to compute-bound:
   $$I^* = \frac{P_{\text{peak}} \text{ (TFLOPs/s)}}{\text{BW}_{\text{peak}} \text{ (TB/s)}}$$
   *Example: An NVIDIA A100 SXM (312 TFLOPs TF32, 2.039 TB/s HBM2e) has $I^* = 312 / 2.039 \approx 153 \text{ FLOPs/Byte}$.*
3. **Attainable Performance Upper Bound**:
   $$P_{\text{attainable}} = \min\left(P_{\text{peak}}, \; I \times \text{BW}_{\text{peak}}\right)$$
   - If $I < I^*$: **Memory Bandwidth Bound**. Doubling compute capacity yields 0% speedup; performance improves only through kernel fusion, activation checkpointing, or memory access coalescing.
   - If $I \ge I^*$: **Compute Bound**. Fully utilizes tensor cores; performance improves via lower precision (FP16, BF16, FP8) or algorithmic reduction.

---

## 3. CUDA Graphs: Eliminating CPU Driver Launch Latency

When tensor dimensions and physical memory addresses are static across iterations:
1. **Stream Capture (`torch.cuda.graph`)**:
   ```python
   g = torch.cuda.CUDAGraph()
   # Warmup iteration
   static_out = model(static_in)
   # Capture graph
   with torch.cuda.graph(g):
       static_out = model(static_in)
   # Replay (zero CPU dispatch overhead)
   g.replay()
   ```
2. **Static Address Invariant**: CUDA Graphs store raw hardware memory pointers. In-place tensor overwrites are valid, but reallocating tensors or changing buffer addresses invalidates the graph.
3. **Host-Device Synchronization**: CUDA Graphs strictly prohibit CPU-GPU synchronization (e.g., `.item()`, `print()`, or dynamic control flow) during capture.

---

## 4. Fused Multi-Tensor Optimizers: Fused AdamW Formulation

AdamW (Loshchilov & Hutter 2019) updates weight $\mathbf{w}_t$ given gradient $\mathbf{g}_t$, first moment $\mathbf{m}_t$, and second moment $\mathbf{v}_t$:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\mathbf{g}_t \odot \mathbf{g}_t)$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

$$\mathbf{w}_t = \mathbf{w}_{t-1} \cdot (1 - \eta \lambda) - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

### Memory Traffic Analysis: Eager vs Fused AdamW
- **Unfused Eager Execution**:
  1. Read $\mathbf{m}, \mathbf{g}$; write updated $\mathbf{m}$ (3 DRAM transfers)
  2. Read $\mathbf{v}, \mathbf{g}$; write updated $\mathbf{v}$ (3 DRAM transfers)
  3. Read $\mathbf{w}, \mathbf{m}, \mathbf{v}$; write updated $\mathbf{w}$ (4 DRAM transfers)
  *Total: 10 DRAM transfers per parameter element.*
- **Fused AdamW**:
  Reads $\mathbf{w}, \mathbf{g}, \mathbf{m}, \mathbf{v}$ once into GPU registers, performs all arithmetic in registers, and writes $\mathbf{w}, \mathbf{m}, \mathbf{v}$ back to DRAM.
  *Total: 7 DRAM transfers per element (30% reduction in memory traffic, eliminating 3 kernel launches).*

---

## 5. PyTorch Profiler & Trace Diagnostics

Diagnosing performance bottlenecks is executed via `torch.profiler`:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profile"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(steps):
        train_step()
        prof.step()
```
- **GPU Idle Bubbles**: Visualized in Chrome Trace (`chrome://tracing` or Perfetto) as whitespace between kernels, indicating CPU dispatch bottlenecks solvable via CUDA Graphs or `torch.compile(mode="reduce-overhead")`.
- **SM Efficiency**: Measures Streaming Multiprocessor tensor core occupancy vs memory wait stalls.

---

## 6. Numerical Invariants and Parity Contracts

- **Transitive Parameter Parity**: Fused single-pass AdamW updates must match eager step updates to within machine precision:
  $$\max |\mathbf{w}_{\text{fused}} - \mathbf{w}_{\text{eager}}| < 10^{-6}$$
- **Momentum State Invariance**: Running first and second order moment tensors $\mathbf{m}_t, \mathbf{v}_t$ must adhere to $< 10^{-5}$ error across PyTorch 2.x, JAX, and Candle.

---

## 7. PyTorch 2.x, JAX, and Candle Architectural Parity

| Feature | PyTorch 2.x | JAX | Rust Candle |
| :--- | :--- | :--- | :--- |
| **Roofline Analysis** | PyTorch Profiler / Kineto | JAX Profiler / XPlane | HW counter analysis / Criterion |
| **Graph Capture** | `torch.cuda.CUDAGraph` | XLA HLO / ahead-of-time compilation | Pre-allocated static buffer graphs |
| **Fused Optimizers** | `torch.optim.AdamW(fused=True)` | `optax.adamw` (JIT-compiled) | Custom fused Rust loops / WGSL |
| **Static Memory Management** | Caching allocator memory pools | XLA memory buffers | Contiguous Tensor allocations |

---

## 8. Production Deployment, Memory Layout Guarantees, and Failure Modes

1. **Re-allocation Hazards in CUDA Graphs**: Creating new tensor allocations during graph replay corrupts the static graph. All input and output buffers must reside in permanent memory addresses allocated before capture.
2. **Optimizer Bias Correction Overflow**: At very large step counts ($t > 10^7$), $\beta_2^t \to 0$; numerical stability is preserved by verifying $1 - \beta^t > 0$.
3. **Weight Decay Ordering**: Decoupled weight decay (AdamW) scales the parameter prior to applying the gradient step, avoiding $L_2$ regularization decay on adaptive momentum gradients.
