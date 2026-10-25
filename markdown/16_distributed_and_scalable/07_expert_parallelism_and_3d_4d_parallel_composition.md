# Expert Parallelism and 3D / 4D Parallel Composition

Frontier foundation models scale compute and parameter counts along two synergistic axes: **conditional routing (Mixture of Experts, MoE)** and **multi-dimensional parallel composition (3D/4D/5D parallelism)**. By placing distinct experts across different accelerators and hierarchically mapping Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP), Context Parallelism (CP), and Expert Parallelism (EP) onto physical cluster topologies, models exceeding hundreds of billions to trillions of parameters can be trained with near-linear scaling efficiency.

---

## 1. Mixture of Experts (MoE) at Scale

Dense transformer models activate all parameters for every processed token ($O(\Psi)$ FLOPs per token for a model of size $\Psi$). Mixture of Experts (MoE) decouples parameter count from per-token compute:
- A model contains $E$ feedforward experts (e.g. $E=8, 64, 256$).
- A parameterized router $W_{\text{gate}} \in \mathbb{R}^{D \times E}$ computes gating logits: $G(x) = x W_{\text{gate}}$.
- Top-$k$ gating selects only $k \ll E$ experts per token (e.g. $k=1, 2, 8$):
  $$y = \sum_{j \in \text{Top-}k} \text{Softmax}(G(x))_j \cdot \text{Expert}_j(x)$$
A 670B MoE model (like DeepSeek-V3 or Mixtral 8x22B) activates only $\approx 37\text{B}$ parameters per token, slashing inference and training FLOPs by up to $18\times$ compared to a dense counterpart of equal capacity.

---

## 2. Expert Parallelism (EP): Sharding Experts Across GPUs

In standard Tensor Parallelism, every expert is sharded internally across all GPUs, requiring two `AllReduce` operations per expert layer. In contrast, **Expert Parallelism (EP)** assigns entire experts to dedicated accelerators:
- If a cluster has $EP$ expert-parallel ranks and $E$ total experts, each accelerator hosts $\frac{E}{EP}$ complete experts:
  $$\text{Experts per Rank} = \frac{E}{EP}$$
- Parameters and optimizer states for an expert reside exclusively on its host GPU.

```
                           Expert Parallelism Lifecycle
                           
        [Token x_0 routes to E0 (GPU 0)]       [Token x_1 routes to E3 (GPU 1)]
                                       │
                                       ▼
                       All-to-All Dispatch Communication
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
     [GPU 0: Hosts E0, E1]                                         [GPU 1: Hosts E2, E3]
Compute: y_0 = E0(x_0)                                         Compute: y_1 = E3(x_1)
        └──────────────────────────────┬──────────────────────────────┘
                                       ▼
                        All-to-All Combine Communication
                                       │
        [Rank 0 receives y_0 * p_0]            [Rank 1 receives y_1 * p_1]
```

---

## 3. The All-to-All Dispatch and Combine Lifecycle

Expert Parallelism converts dense matrix multiplications into a two-phase communication pattern:
1. **Dispatch (Token Routing):**
   Tokens on rank $r$ are routed to target experts. An `All-to-All` collective transmits tokens from source ranks to the destination GPU hosting each chosen expert.
2. **Local Batched GEMM:**
   Each GPU groups received tokens by local expert index and computes feedforward projections using grouped GEMM (`cutlass` / `torch._grouped_gemm`).
3. **Combine (Output Gathering):**
   A second `All-to-All` collective transmits the expert outputs back to their original token owner GPUs, where outputs are weighted by the routing probabilities $p_j$.

---

## 4. Token Capacity & Load Balancing

Because token routing is data-dependent, naive gating can route the majority of tokens to a handful of "popular" experts, creating a crippling **expert imbalance** where overloaded GPUs bottleneck the entire cluster.
- **Expert Capacity Factor ($C$):** Sets a hard ceiling on the number of tokens an expert can process:
  $$\text{Capacity} = C \times \frac{T \cdot k}{E}$$
  Tokens exceeding capacity are dropped or routed through residual shortcut connections.
- **Auxiliary Balancing Loss:** Encourages uniform routing across all $E$ experts:
  $$\mathcal{L}_{\text{balance}} = \alpha \cdot E \sum_{e=1}^E f_e \cdot P_e$$
  where $f_e$ is the fraction of tokens routed to expert $e$, and $P_e$ is the average routing probability.
- **Auxiliary-Loss-Free Routing (DeepSeek-V3):** Introduces learnable expert bias terms $b_e$ updated dynamically to balance load without degrading main language modeling performance.

---

## 5. 3D Parallelism Composition (DP × PP × TP)

Frontier models compose three foundational parallelism paradigms:
$$\text{Total GPUs } N = DP \times PP \times TP$$

```
                         3D Parallelism Grid Hierarchy
                         
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           Data Parallel (DP)                           │
  │   Replicates pipeline instances across nodes / racks                   │
  │                                                                        │
  │   ┌────────────────────────────────────────────────────────────────┐   │
  │   │                    Pipeline Parallel (PP)                      │   │
  │   │   Slices layers across contiguous nodes (InfiniBand P2P)       │   │
  │   │                                                                │   │
  │   │   ┌────────────────────────────────────────────────────────┐   │   │
  │   │   │                 Tensor Parallel (TP)                   │   │   │
  │   │   │   Shards intra-layer GEMMs inside single node (NVLink) │   │   │
  │   │   └────────────────────────────────────────────────────────┘   │   │
  │   └────────────────────────────────────────────────────────────────┘   │
  └────────────────────────────────────────────────────────────────────────┘
```

### Cartesian Rank Mapping:
Every global rank $R \in [0, N-1]$ maps to a unique 3D coordinate $(d, p, t)$:
$$d = \left\lfloor \frac{R}{PP \cdot TP} \right\rfloor, \quad p = \left\lfloor \frac{R \pmod{PP \cdot TP}}{TP} \right\rfloor, \quad t = R \pmod{TP}$$

- **TP Group:** Accelerators sharing identical $d$ and $p$ ($\text{stride} = 1$).
- **PP Group:** Accelerators sharing identical $d$ and $t$ ($\text{stride} = TP$).
- **DP Group:** Accelerators sharing identical $p$ and $t$ ($\text{stride} = PP \cdot TP$).

---

## 6. 4D and 5D Parallelism: Integrating CP and EP

When training long-context MoE models, two additional dimensions are introduced:
$$\text{Total GPUs } N = \frac{DP}{\text{EP}} \times PP \times TP \times CP \times EP$$

### Hardware Topology Mapping Rules:
1. **Highest Bandwidth (Intra-Node NVLink $\ge 900\text{ GB/s}$):**
   - Place **Tensor Parallelism ($TP \le 8$)** and **Context Parallelism (Ulysses)** inside the 8-GPU NVLink domain.
2. **Intermediate Bandwidth (Optical / Inter-Node RoCE / InfiniBand):**
   - Place **Expert Parallelism ($EP$)** and **ZeRO-3 / FSDP2** across high-speed switch fabrics.
3. **Lowest Bandwidth (Cross-Spine / Inter-Rack):**
   - Place **Pipeline Parallelism ($PP$)** and **Data Parallelism ($DP$)**, which require only boundary activations or non-blocking gradient all-reduces.

---

## 7. Multi-Framework Parity & Verification

Mathematical consistency is validated across PyTorch 2.x, JAX, Candle, and hardware kernels:
- **PyTorch 2.x:** Implements Top-2 routing, dispatch masking, and 3D mesh coordinate mapping.
- **JAX:** Expresses expert dispatch and top-k selection via pure functional transformations (`jax.lax.top_k`).
- **Candle:** High-performance Rust implementation with batched expert linear passes and dispatch mask generation.
- **WGSL Kernel:** Implements `FUSED_EXPERT_COMBINE_WGSL` for hardware-accelerated weighted expert output gathering.

Parity tests guarantee absolute error $< 10^{-5}$ across all framework outputs against the golden checkpoint.

---

## 8. Failure Modes, Pathology, and Debugging

| Failure Mode | Root Cause | Observable Symptom | Mitigation / Remedy |
| :--- | :--- | :--- | :--- |
| **Expert Load Imbalance** | Routing collapse to 1–2 dominant experts | Severe GPU straggler delay; one GPU at 100% compute while others idle | Increase auxiliary balancing loss weight $\alpha$ or use bias-corrected routing. |
| **All-to-All Hotspotting** | Non-uniform token distribution causing bisection congestion | All-to-All collective latency spikes by $> 10\times$ | Enforce strict capacity limits or random token permutation prior to dispatch. |
| **Mesh Dimension Mismatch** | $DP \times PP \times TP \ne N$ | Process group initialization failure at cluster boot | Validate Cartesian product matches total GPU count strictly. |
| **EP / TP Collective Conflict** | Overlapping All-to-All and AllReduce on the same default CUDA stream | High serialization stalls and low hardware concurrency | Allocate separate CUDA streams for EP All-to-All and TP AllReduce. |
