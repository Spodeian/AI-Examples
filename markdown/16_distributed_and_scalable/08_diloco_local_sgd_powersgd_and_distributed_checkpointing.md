# DiLoCo, Local SGD, PowerSGD, and Distributed Checkpointing

Distributed model training is frequently constrained by network latency, limited bisection bandwidth, or massive checkpoint I/O overhead. When training clusters are geographically dispersed across continents, or connected over commodity public internet without InfiniBand, standard per-step synchronous AllReduce halts training progress. **DiLoCo (Distributed Low-Communication Training)**, **PowerSGD low-rank gradient compression**, and **Distributed Checkpointing (DCP)** provide the algorithmic and systems infrastructure enabling asynchronous, bandwidth-efficient, and fault-tolerant large-scale AI.

---

## 1. The Global Network Bottleneck in Distributed AI

Modern foundation models rely on dense data parallelism where every GPU synchronizes parameter gradients after every single batch step ($T_{\text{step}} = T_{\text{compute}} + T_{\text{AllReduce}}$).
- **Intra-Datacenter Interconnect:** InfiniBand / RoCE ($400\text{--}800\text{ Gbps}$, $< 2\ \mu\text{s}$ latency).
- **Cross-Datacenter / Internet:** Commodity WAN ($1\text{--}10\text{ Gbps}$, $30\text{--}100\text{ ms}$ latency).

Synchronous AllReduce over cross-datacenter WAN incurs a $> 10,000\times$ latency penalty, rendering standard DDP completely unviable. Training across heterogeneous, federated, or geographically separated clusters requires **low-frequency synchronization** and **lossy gradient compression**.

---

## 2. DiLoCo: Distributed Low-Communication Training

Developed by Douillard et al. (Google DeepMind, 2023), **DiLoCo** reformulates distributed training as a bilevel optimization problem composed of fast local inner steps and periodic global outer steps:

```
                               DiLoCo Optimization Architecture
                               
                                Global Checkpoint theta_0
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
     Worker Cluster 1 (e.g. EU)                                Worker Cluster 2 (e.g. US)
Runs H local steps (AdamW on local data):                Runs H local steps (AdamW on local data):
theta_{k+1} = theta_k - eta_in * Adam(grad)             theta_{k+1} = theta_k - eta_in * Adam(grad)
               │                                                         │
         Yields theta^(1)                                          Yields theta^(2)
               │                                                         │
   Delta^(1) = theta_0 - theta^(1)                           Delta^(2) = theta_0 - theta^(2)
               └────────────────────────────┬────────────────────────────┘
                                            ▼
                           Averaged Pseudo-Gradient Delta_bar
                                            │
                             Outer Optimizer (SGD + Momentum)
                               v = beta * v + Delta_bar
                              theta_new = theta_0 - eta_out * v
                                            │
                         Broadcast theta_new to all clusters
```

### The Bilevel Algorithm:
1. **Inner Loop (Local Computation):**
   Each worker cluster $w \in [1, W]$ performs $H$ independent training steps (e.g., $H=500$) using standard AdamW on its local dataset partition:
   $$\theta_{k+1}^{(w)} = \theta_k^{(w)} - \eta_{\text{inner}} \text{AdamW}(\nabla \mathcal{L}(\theta_k^{(w)}))$$
2. **Pseudo-Gradient Materialization:**
   After $H$ steps, each worker computes its parameter trajectory delta (the pseudo-gradient):
   $$\Delta^{(w)} = \theta_0 - \theta_H^{(w)}$$
3. **Outer Loop (Global Synchronization):**
   Workers communicate only once every $H$ steps via an outer AllReduce:
   $$\bar{\Delta} = \frac{1}{W} \sum_{w=1}^W \Delta^{(w)}$$
   The global parameters are updated using an outer optimizer (typically Nesterov SGD with momentum $\beta=0.9, \eta_{\text{outer}}=0.7$):
   $$v_{\text{outer}} \leftarrow \beta \cdot v_{\text{outer}} + \bar{\Delta}$$
   $$\theta_{\text{new}} \leftarrow \theta_0 - \eta_{\text{outer}} \cdot v_{\text{outer}}$$

---

## 3. Communication Scaling & Invariance

Because communication occurs only once every $H$ steps, DiLoCo achieves an exact $H\times$ reduction in communication bandwidth requirements:
$$\text{Communication Volume}_{\text{DiLoCo}} = \frac{1}{H} \times \text{Communication Volume}_{\text{DDP}}$$

For $H = 500$, cross-cluster bandwidth requirements drop by $99.8\%$, enabling multi-continental training with zero loss degradation compared to tightly-coupled DDP baselines.

---

## 4. PowerSGD: Low-Rank Gradient Compression

When even $H$-step updates or intermediate gradient syncs are constrained by bandwidth, **PowerSGD (Vogels et al., NeurIPS 2019)** compresses gradient matrices using alternating power iteration.

For a weight gradient $G \in \mathbb{R}^{M \times N}$, PowerSGD approximates $G$ via low-rank factors $P \in \mathbb{R}^{M \times r}$ and $Q \in \mathbb{R}^{N \times r}$ ($r \ll \min(M, N)$):
1. **Left Projection:** $P = G Q$ (where $Q$ is warm-started from the prior step).
2. **Orthogonalization:** $P_{\text{norm}} = \text{orth}(P)$ (via QR decomposition or column normalization).
3. **Right Projection:** $Q_{\text{new}} = G^\top P_{\text{norm}}$.
4. **AllReduce Factors:** Transmit only $P_{\text{norm}}$ ($M \times r$ floats) and $Q_{\text{new}}$ ($N \times r$ floats) instead of full $G$ ($M \times N$ floats).
5. **Reconstruction:** $\hat{G} = P_{\text{norm}} Q_{\text{new}}^\top$.

$$\text{Compression Factor} = \frac{M \cdot N}{(M + N) \cdot r}$$
For a $4096 \times 4096$ matrix and rank $r=4$, compression reaches $512\times$!

---

## 5. Error Feedback and Residual Accumulation

Because low-rank approximation drops high-frequency gradient information, naive compression causes divergence. PowerSGD maintains an **error feedback buffer** $E \in \mathbb{R}^{M \times N}$:
$$G_{\text{effective}} = G_t + E_t$$
$$\hat{G}_t = \text{Compress}(G_{\text{effective}})$$
$$E_{t+1} = G_{\text{effective}} - \hat{G}_t$$
Uncompressed gradient residuals are stored locally and added into the next step's gradient, guaranteeing asymptotic convergence identical to uncompressed SGD.

---

## 6. Distributed Checkpointing (DCP) & Resharding

In frontier models with billions of parameters sharded via FSDP2 or Megatron TP/PP/EP, checkpointing presents a severe bottleneck:
- **Naive Monolithic Checkpointing:** Gathering sharded states to rank 0 causes host CPU Out-Of-Memory and takes hours over network filesystems.
- **Distributed Checkpointing (DCP):** Every accelerator saves and loads its local parameter and optimizer shards concurrently in parallel without gathering.

### Resharding Invariance:
Modern DCP abstractions (e.g. `torch.distributed.checkpoint` and Sharded Safetensors) record global coordinate metadata for each tensor slice:
$$\text{Metadata: } \left[ \text{tensor\_name}, \text{global\_shape}, \text{local\_slice\_offset} \right]$$
This decouples checkpoint structure from GPU topology: a model trained on 1024 GPUs with $TP=8, DP=128$ can be restored directly onto 256 GPUs with $TP=4, DP=64$ with automatic zero-copy re-slicing.

---

## 7. Multi-Framework Parity & Verification

Mathematical precision is maintained across all four backends:
- **PyTorch 2.x:** Implements DiLoCo outer updates, PowerSGD power iteration, and DCP shard concatenation/re-slicing.
- **JAX:** Functional implementation of bilevel trajectory averaging and rank-1 gradient compression.
- **Candle:** High-performance Rust implementation using `Tensor::cat`, `Tensor::narrow`, and vector dot-products.
- **WGSL Kernel:** Implements `FUSED_DILOCO_OUTER_STEP_WGSL` for hardware-accelerated outer parameter updates.

Parity tests confirm absolute difference $< 10^{-5}$ across all framework outputs against the golden checkpoint.

---

## 8. Failure Modes, Pathology, and Debugging

| Failure Mode | Root Cause | Observable Symptom | Mitigation / Remedy |
| :--- | :--- | :--- | :--- |
| **Worker Trajectory Divergence** | Inner step count $H$ too large on non-IID data | Training loss plateaus or diverges after 3 outer steps | Reduce $H$ (e.g. from 1000 to 300) or increase outer momentum $\beta$. |
| **PowerSGD Vanishing Gradient** | Missing error feedback residual buffer | Loss stalls early; weights stop updating | Accumulate uncompressed gradient residuals $E_{t+1} = G - \hat{G}$. |
| **DCP Resharding Key Collision** | Sharded metadata names not fully qualified with layer/expert index | Corrupted weights loaded during cluster restart | Ensure every tensor slice key incorporates module hierarchy and slice offset. |
| **Outer Learning Rate Instability** | Setting outer learning rate $\eta_{\text{outer}} > 1.0$ | Wild gradient norm oscillations in outer loop | Keep $\eta_{\text{outer}} \in [0.5, 0.8]$ and apply gradient clipping to $\bar{\Delta}$. |
