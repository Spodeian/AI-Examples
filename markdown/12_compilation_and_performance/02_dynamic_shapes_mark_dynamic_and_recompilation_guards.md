# Dynamic Shapes, Mark Dynamic, and Recompilation Guards

## 1. Mathematical Foundations & Lineage

Early deep learning compilers operated under strict **static shape assumptions**: every dimension (batch size $B$, sequence length $L$, spatial resolution $H \times W$) had to be known ahead of time:
1. **XLA Static Compilation (Frostig et al. 2018)**: Specialized kernels for exact input shapes. Feeding a batch with length $L=33$ after $L=32$ triggered a full recompilation pause.
2. **Specialization vs Generalization Trade-Off**: Specializing to fixed integer dimensions allows compilers to unroll loops, allocate static register tiles, and pre-compute stride arithmetic. However, variable-length natural language, ragged point clouds, and multi-scale vision images render static specialization unusable without padding waste.
3. **Symbolic Integer Representation (`SymInt` — Ansel et al. 2024)**: PyTorch 2 represents dynamic dimensions as symbolic variables $s_0, s_1 \in \mathbb{Z}^+$. TorchInductor generates kernels parameterized by symbolic strides and runtime loops rather than compiling static scalar constants.
4. **Shape Guards & Guard Evaluation**: When an input tensor is fed to a compiled function, Dynamo evaluates guards on its shape. If the compiler assumed a dimension was static (e.g. $B=16$), a new batch with $B=32$ invalidates the guard and triggers recompilation.
5. **Power-of-2 Bucketing**: Grouping dynamic sequence lengths into discrete power-of-2 boundaries ($L \in \{32, 64, 128, 256\}$) bounds total compiled kernel count to $\mathcal{O}(\log L_{\max})$ while minimizing wasted padding compute.

---

## 2. Symbolic Integers (SymInt) and Shape Propagation

Under dynamic shape tracking:
1. Instead of a concrete shape $(16, 128, 768)$, the tensor has symbolic shape:
   $$\text{Shape}(X) = (s_0, s_1, 768)$$
   where $s_0$ (batch size) and $s_1$ (sequence length) are dynamic `SymInt` variables.
2. Operations propagate symbolic expressions:
   $$\text{Reshape}(X, (s_0 \cdot s_1, 768)) \implies \text{Dim}_0 = s_0 \cdot s_1$$
   $$\text{Linear}(X, W) \implies \text{Flops} = 2 \cdot s_0 \cdot s_1 \cdot 768 \cdot D_{\text{out}}$$
3. **Inductor Kernel Generation**: The generated Triton/C++ kernel accepts $s_0$ and $s_1$ as runtime uniform scalar parameters, reusing a single compiled binary across arbitrary sequence lengths.

---

## 3. Explicit Dynamic Specification: `mark_dynamic` and `dynamic=True`

PyTorch provides two primary mechanisms to instruct Dynamo that dimensions should remain symbolic:
1. **Ahead-of-Time Model Compilation**:
   ```python
   compiled_model = torch.compile(model, dynamic=True)
   ```
   Tells Dynamo to treat all batch and sequence dimensions as symbolic from the very first frame evaluation, avoiding the initial static specialization phase.
2. **Fine-Grained Dimension Tagging**:
   ```python
   # Mark dimension 1 (sequence length) as dynamic with optional bounds [min, max]
   torch._dynamo.mark_dynamic(sample_input, 1, min=1, max=4096)
   ```
   Enables fine-grained hybrid execution: batch size and sequence length remain dynamic, while hidden channel dimension $D=768$ remains statically specialized for peak SIMD vectorization.

---

## 4. Power-of-2 Shape Bucketing & Masked Operations

When variable-length inputs cannot be executed with fully dynamic kernels, or when hardware architectures (e.g. TPUs or static FPGA accelerators) mandate static dimensions:
1. **Bucket Allocation**: Define discrete boundaries $\mathcal{B} = \{b_1, \dots, b_K\}$. For input length $L$:
   $$b^* = \arg\min \{b \in \mathcal{B} : b \ge L\}$$
2. **Padding & Masking**: Pad input to $b^*$ and apply a binary boolean mask $M \in \{0, 1\}^{B \times b^*}$:
   $$\text{MaskedPool}(X, M) = \frac{\sum_{t=1}^{b^*} X_{b, t} \cdot M_{b, t}}{\max\left(1, \sum_{t=1}^{b^*} M_{b, t}\right)}$$
3. **Recompilation Bound**: Restricts the maximum number of compiled program artifacts to $|\mathcal{B}|$ rather than $L_{\max}$, eliminating recompilation thrashing.

---

## 5. Multi-Ecosystem Implementation Matrix

| Concept | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Symbolic Dim** | `SymInt` / `mark_dynamic` | `jax.export` symbolic | Rust runtime slice length | Variable batch loop |
| **Masked Pooling** | Masked sum & clamp div | `jnp.sum` with mask | Vectorized masked pool | Fused masked reducer |
| **Bucket Alloc** | Min enclosing distance | `jnp.argmin` difference | Vector loop boundary search | Binary search bucket |
| **Dynamic RMSNorm** | Dynamic reduction loop | Functional dynamic mean | Single-pass RMS loop | `DYNAMIC_RMS_NORM_WGSL` |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Recompilation Thrashing (Guard Churn)**:
   - *Symptom*: Model pauses for 2–5 seconds every few iterations during training, consuming dozens of gigabytes of RAM.
   - *Cause*: Dynamo specialized to exact sequence lengths ($L=17, 18, 19, \dots$). Each new length triggers an expensive Triton compilation pass.
   - *Fix*: Pass `dynamic=True` to `torch.compile` or call `torch._dynamo.mark_dynamic(x, 1)`.
2. **0/1 Specialization Pitfalls**:
   - *Symptom*: Dynamo compiles a specialized branch for batch size $B=1$ that fails when $B > 1$.
   - *Cause*: By default, Dynamo specializes on $0$ and $1$ to optimize out singleton dimensions and broadcasting.
   - *Fix*: Explicitly mark $B$ as dynamic before the first forward pass.
3. **Memory Waste from Naive Static Max-Padding**:
   - *Symptom*: 80% of GPU compute time spent multiplying zeros on short sequences padded to $L=4096$.
   - *Cause*: Monolithic static padding without dynamic bucketing.
   - *Fix*: Implement Power-of-2 Bucketing to cap padding overhead below $25\%$.

---

## 7. Real-World Engineering Applications

1. **Ragged Document Retrieval (ColBERT / Dense RAG)**: Processing variable-length passage queries without static maximum padding waste.
2. **Speech Recognition & Audio Streaming (Conformer)**: Executing inference across speech audio clips ranging from 0.5 seconds to 30 seconds.
3. **LiDAR Point Cloud Perception in Robotics**: Handling dynamic point clouds where point count fluctuates between 10,000 and 100,000 per frame.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.safetensors`:
1. `masked_pool_ref`: Dynamic masked mean pooling representation.
2. `allocated_buckets_ref`: Allocated bucket boundaries for variable sequence lengths.
3. `dynamic_rms_ref`: Root Mean Square Normalization outputs across dynamic inputs.

### Synchronized Ecosystem Artifacts
- **Notebook:** [02_dynamic_shapes_mark_dynamic_and_recompilation_guards.ipynb](../../notebooks/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.ipynb)
- **PyTorch 2 Module:** [02_dynamic_shapes_mark_dynamic_and_recompilation_guards.py](../../pytorch2/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.py)
- **JAX Module:** [02_dynamic_shapes_mark_dynamic_and_recompilation_guards.py](../../jax/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.py)
- **Candle Module:** [02_dynamic_shapes_mark_dynamic_and_recompilation_guards.rs](../../candle/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.rs)
- **Hardware Kernel:** [02_dynamic_shapes_mark_dynamic_and_recompilation_guards.rs](../../kernels/12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.rs)
