# Production Serving Dynamic Batcher PD Disaggregation and Streaming

## 1. Architectural Role & Context

In enterprise LLM serving architectures, serving workloads exhibit dual bottlenecks with opposing hardware demands:
1. **Prefill Phase (Prompt Ingestion):** High-arithmetic-intensity ($> 50\text{ FLOPs/Byte}$), compute-bound operations that benefit from massive tensor-parallel GPU clusters to minimize **Time To First Token (TTFT)**.
2. **Decode Phase (Token-by-Token Generation):** Low-arithmetic-intensity ($\approx 1\text{ FLOP/Byte}$), memory-bandwidth-bound operations where decoding a single token requires streaming the entire key-value (KV) cache through GPU HBM to satisfy **Time Per Output Token (TPOT)** SLAs.

When prefill and decode phases share the same hardware instance, incoming long-context prompts cause **Head-of-Line (HoL) Blocking**, stalling ongoing decode streams and causing severe TPOT latency spikes ($> 20\times$).

**Prefill-Decode (PD) Disaggregation** (Splitwise, DistServe, Sarathi-Serve) decouples prefill workers from decode workers, transferring the compiled KV cache across high-bandwidth RDMA / PCIe interconnects while utilizing **Dynamic Micro-Batching** and **Chunked Prefill** to maximize hardware saturation.

---

## 2. Mathematical Formalism

### 2.1 Prefill Forward Step
Given prompt tokens $X_{\text{prefill}} \in \mathbb{R}^{B \times S \times D}$:
$$Q = X W_q, \quad K = X W_k, \quad V = X W_v$$
$$A = \text{softmax}\left( \frac{Q K^T}{\sqrt{D}} + M_{\text{causal}} \right), \quad Y_{\text{prefill}} = A V W_o + b_o$$
where $M_{\text{causal}}$ masks upper-triangular future tokens ($j > i \implies -\infty$).

### 2.2 PD-Disaggregation KV Transfer & Decode Step
The prefill worker transmits $K, V \in \mathbb{R}^{B \times S \times D}$ to the decode worker. For incoming token $x_{\text{new}} \in \mathbb{R}^{B \times 1 \times D}$:
$$q_{\text{new}} = x_{\text{new}} W_q, \quad k_{\text{new}} = x_{\text{new}} W_k, \quad v_{\text{new}} = x_{\text{new}} W_v$$
$$K_{\text{all}} = [K; k_{\text{new}}] \in \mathbb{R}^{B \times (S+1) \times D}, \quad V_{\text{all}} = [V; v_{\text{new}}] \in \mathbb{R}^{B \times (S+1) \times D}$$
$$a_{\text{decode}} = \text{softmax}\left( \frac{q_{\text{new}} K_{\text{all}}^T}{\sqrt{D}} \right) \in \mathbb{R}^{B \times 1 \times (S+1)}$$
$$y_{\text{decode}} = a_{\text{decode}} V_{\text{all}} W_o + b_o \in \mathbb{R}^{B \times 1 \times D}$$

### 2.3 Mathematical Invariance Guarantee
Autoregressive causal attention guarantees that the incremental decode token activation $y_{\text{decode}}$ is mathematically identical to the final slice of a full causal forward pass:
$$y_{\text{decode}} \equiv Y_{\text{full}}[:, -1:, :]$$
where $Y_{\text{full}} = \text{Prefill}([X_{\text{prefill}}; x_{\text{decode}}])$.

---

## 3. Algorithmic Implementation & Serving Pipeline

1. **Dynamic Micro-Batcher:** Groups incoming user requests dynamically according to `max_tokens_per_batch` and `max_wait_ms`.
2. **Chunked Prefill Scheduler:** Breaks prompt sequences exceeding $C_{\text{chunk}}$ tokens into sub-sequences to prevent prefill starvation.
3. **KV Cache Transfer Engine:** Asynchronously sends computed key/value activations from prefill nodes to decode nodes via non-blocking DMA.
4. **Decode Token Loop:** Iteratively processes single tokens with $O(1)$ query projection and $O(S)$ cache dot-product attention.
5. **Streaming Emitter:** Emits generated tokens as SSE (Server-Sent Events) chunks with latency timestamps.

---

## 4. Cross-Track Verification & Parity

The full serving pipeline is verified across PyTorch 2.x, JAX, Rust Candle, WGPU/CPU Kernels, and Jupyter Notebook:
- Seed-42 golden tensors stored in `weights/17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming.safetensors`.
- Parity assertion $\|y_{\text{decode}} - y_{\text{golden}}\|_\infty < 10^{-5}$.
- Exact mathematical match between KV-cached incremental decode step and full-sequence causal forward pass.

---

## 5. Failure Modes & Misuse Guardrails

| Misuse Mode | Mechanism | Operational Guardrail |
| :--- | :--- | :--- |
| **Colocated Unchunked Prefills** | Inserting a $32\text{k}$-token prefill into an active decode batch causes massive compute starvation. | Enforce PD-Disaggregation or Chunked Prefill ($C \le 512$). |
| **Unbounded Batch Sizing** | Over-allocating decode batch size exhausts GPU HBM due to KV-cache expansion. | Implement PagedAttention / vLLM block allocation with strict pre-emption policies. |
| **Synchronous KV Transfer** | Blocking the decode worker on raw socket TCP KV-cache transfer introduces jitter. | Use RDMA / PCIe P2P direct memory copies with double-buffered KV pipelines. |

---

## 6. Roofline Analysis & Hardware Performance

- **Prefill Arithmetic Intensity:** $\approx 50\text{--}150\text{ FLOPs/Byte}$, saturated on Tensor Cores.
- **Decode Arithmetic Intensity:** $\approx 1\text{--}2\text{ FLOPs/Byte}$, bound strictly by HBM memory bandwidth.
- Disaggregating the two workloads allows prefill clusters to run at high clock frequencies with high power draw while decode clusters prioritize memory channel concurrency.

---

## 7. Cross-Domain Transfer & Isomorphisms

| Domain | PD-Disaggregation Analogue | Architectural Objective |
| :--- | :--- | :--- |
| **Database Engines** | OLAP vs OLTP Storage Separation | Isolate analytical batch scanning from low-latency point queries |
| **Operating Systems** | High-Priority Interactive vs Batch CPU Scheduling | Prevent batch compilation jobs from degrading UI thread responsiveness |
| **Robotics Control** | Global Trajectory Planning vs High-Hz Motor Decoding | Isolate heavy 10Hz obstacle map planning from 1000Hz motor joint control |

---

## 8. References & Further Reading

1. Crankshaw, D., et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI.
2. Patel, P., et al. (2024). *Splitwise: Efficient Generative LLM Serving Using Phase Splitting*. ISCA.
3. Zhong, Y., et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving*. OSDI.
4. Agrawal, A., et al. (2024). *Sarathi-Serve: Efficient LLM Serving with Chunked-Prefills*. OSDI.
