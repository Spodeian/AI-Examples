# Pipeline Parallelism: GPipe, 1F1B, Zero-Bubble, and DualPipe

Pipeline Parallelism (PP) partitions a deep neural network sequentially across $P$ accelerators along the layer dimension. Unlike Tensor Parallelism, which decomposes intra-layer matrix multiplications and demands ultra-high-speed NVLink interconnects, Pipeline Parallelism transfers only boundary activations and gradients between adjacent stages. This low communication volume makes Pipeline Parallelism the premier choice for scaling across distributed multi-node clusters over standard InfiniBand or Ethernet fabrics.

---

## 1. Architectural Motivation & Inter-Layer Partitioning

When training hundreds of transformer layers (e.g. 60–128 layers in frontier LLMs), no single GPU or NVLink node can host all parameters and optimizer states. Pipeline Parallelism partitions the $L$ layers into $P$ contiguous stages:
- **Stage $0$:** Embedding layer + layers $0 \dots \frac{L}{P}-1$.
- **Stage $p$:** Layers $p \frac{L}{P} \dots (p+1) \frac{L}{P}-1$.
- **Stage $P-1$:** Final layers + Output/LM Head + Loss function.

Inter-stage communication is strictly point-to-point (P2P): stage $p$ sends forward activations $h_p$ to stage $p+1$, and receives backward gradients $\delta_{p+1}$ from stage $p+1$.

---

## 2. The Pipeline Bubble Formulation

If a batch of size $B$ were processed naively as a single monolithic forward and backward pass, only one stage could compute at any instant, leaving $P-1$ stages completely idle. The pipeline bubble fraction would be $\frac{P-1}{P}$ (e.g., $75\%$ idle time on 4 stages!).

To maximize hardware utilization, the batch is split into $M$ smaller **microbatches** ($M \gg P$). The theoretical idle bubble fraction across $M$ microbatches on $P$ stages is:
$$F_{\text{bubble}} = \frac{P - 1}{M + P - 1}$$

As $M \to \infty$, $F_{\text{bubble}} \to 0$. However, increasing $M$ increases the number of un-freed microbatch activations held in memory, creating a fundamental tension between **pipeline bubble efficiency** and **activation memory consumption**.

---

## 3. GPipe: Synchronous Microbatch Pipelining

Introduced by Huang et al. (GPipe, 2019), GPipe executes all $M$ forward microbatches across stages $0 \dots P-1$, followed by all $M$ backward microbatches:

```
                            GPipe Execution Schedule (P=4, M=4)
                            
Stage 3:              [F0][F1][F2][F3][B3][B2][B1][B0]
Stage 2:          [F0][F1][F2][F3]        [B3][B2][B1][B0]
Stage 1:      [F0][F1][F2][F3]                [B3][B2][B1][B0]
Stage 0:  [F0][F1][F2][F3]                        [B3][B2][B1][B0]
          ◄─────────────── Flush Bubble ───────────────►
```

### Limitations of GPipe:
1. **Activation Memory Explosion:** Stage 0 must retain activations for all $M$ microbatches in HBM until the backward pass of microbatch $0$ finally returns:
   $$\text{Peak Memory} = O(M \cdot \text{Activation Size})$$
2. **Synchronization Barrier:** Gradients are accumulated across all $M$ microbatches before an optimizer step is executed at the boundary ("pipeline flush").

---

## 4. 1F1B (One Forward, One Backward) Scheduling

Introduced in PipeDream (Narayanan et al., 2019) and Megatron-LM (Shoeybi et al., 2019), **1F1B** bounds peak activation memory by interleaving forward and backward microbatches:
1. **Warmup Phase:** Stage $p$ computes $P - p - 1$ forward microbatches.
2. **Steady-State Phase:** Each stage alternates strictly between computing one forward microbatch and one backward microbatch (`1F1B`).
3. **Cooldown Phase:** Complete remaining backward microbatches.

```
                            1F1B Steady-State Schedule
                            
Stage 3:              [F0][B0][F1][B1][F2][B2][F3][B3]
Stage 2:          [F0][F1][B0][F2][B1][F3][B2]    [B3]
Stage 1:      [F0][F1][F2][B0][F3][B1]    [B2]    [B3]
Stage 0:  [F0][F1][F2][F3][B0]    [B1]    [B2]    [B3]
```

### Memory Advantage:
As soon as stage $p$ finishes the backward pass of microbatch $m$, the activations for microbatch $m$ are immediately freed! Consequently, peak activation memory is strictly capped at $O(P)$ rather than $O(M)$:
$$\text{Peak Memory}_{\text{1F1B}} = O(P) \ll O(M)$$

---

## 5. Zero-Bubble Pipeline Parallelism: Decoupling B_A and B_W

Standard backward computation conflates two distinct mathematical steps:
1. **Activation Gradient ($B_A$):** $\nabla X_m = \delta_m W^\top$. This gradient is **latency-critical** because the upstream stage $p-1$ cannot begin its backward pass until it receives $\nabla X_m$.
2. **Weight Gradient ($B_W$):** $\nabla W = X_m^\top \delta_m$. This gradient is **latency-tolerant**; it is purely accumulated locally and is not needed by any other stage until the end-of-step optimizer update!

**Zero-Bubble Pipeline Parallelism (Qi et al., 2024)** exploits this insight:
- Immediately compute $B_A$ and transmit the activation gradient upstream to unlock stage $p-1$.
- Defer the computation of $B_W$, inserting $B_W$ tasks directly into the idle bubbles of the pipeline schedule.
- Because $B_A$ execution time is only $\approx \frac{1}{3} \times$ full backward time ($t_{B_A} \approx \frac{1}{3} t_B, t_{B_W} \approx \frac{2}{3} t_B$), scheduling $B_W$ into bubble slots shrinks the effective bubble fraction to near zero:
  $$F_{\text{Zero-Bubble}} \approx 0$$

---

## 6. DualPipe: Bidirectional Overlapped Pipeline (DeepSeek-V3)

**DualPipe** (DeepSeek-V3, 2024) advances pipeline design by scheduling two independent pipelines (or overlapping chunks) moving in opposite directions:
1. **Forward and Backward Overlap:** Forward microbatches of stream $A$ overlap with backward microbatches of stream $B$.
2. **Fine-Grained $B_A$ / $B_W$ Splitting:** Prioritizes $B_A$ propagation while interleaving $B_W$ during communication windows.
3. **P2P Communication Hiding:** Intra-node NVLink and inter-node InfiniBand P2P transfers are completely hidden behind local GEMM operations on concurrent CUDA streams.

---

## 7. Multi-Framework Parity & Verification

Cross-track consistency is validated across PyTorch 2.x, JAX, Candle, and hardware kernels:
- **PyTorch 2.x:** Simulates 2-stage pipeline forward execution across microbatches and executes the decoupled $B_A$ and $B_W$ backward decomposition.
- **JAX:** Functional implementation of microbatch processing and gradient accumulation using pure array operations.
- **Candle:** High-performance Rust implementation with batched linear passes and transposition.
- **WGSL Kernel:** Implements `FUSED_DUALPIPE_BW_ACCUMULATE_WGSL` for GPU-accelerated deferred weight gradient accumulation into persistent parameter buffers.

Numerical checks verify absolute difference $< 10^{-5}$ across all framework outputs against the seed-42 golden checkpoint.

---

## 8. Failure Modes, Pathology, and Debugging

| Failure Mode | Root Cause | Observable Symptom | Mitigation / Remedy |
| :--- | :--- | :--- | :--- |
| **Microbatch Starvation Bubble** | Too few microbatches ($M < P$) | Idle bubble exceeds compute time ($F_{\text{bubble}} > 50\%$) | Increase global batch size or reduce PP degree so $M \ge 4P$. |
| **Activation Memory OOM under GPipe** | Storing all $M$ activations without gradient checkpointing | Device crashes with Out-Of-Memory on stage 0 | Switch from GPipe to 1F1B or 1F1B with selective activation recomputation. |
| **Pipeline Stage Imbalance** | Unequal layer distribution (e.g., stage 0 contains embedding, stage $P-1$ contains LM head) | Other stages spend substantial time waiting on stage 0 or $P-1$ | Assign fewer transformer layers to stages 0 and $P-1$ to balance wall-clock latency. |
| **P2P Communication Hang** | Mismatched send/recv order between stage $p$ and $p+1$ | Pipeline execution deadlocks on step 1 | Pair non-blocking `isend` before `irecv` or use bidirectional communication primitives. |
