# PyTorch 2 Compilation Stack: Modes, Backends, and Graph Breaks

## 1. Mathematical Foundations & Compiler Lineage

Deep learning execution in Python traditionally suffers from high framework interpreter overhead, fine-grained GPU kernel launch latency, and bandwidth-bound memory round-trips to high-bandwidth memory (HBM):
1. **Ahead-of-Time Graph IR & XLA (Frostig et al. 2018; Leary & Wang 2017)**: Introduced compiler graph optimization via static Single-Assignment Intermediate Representation (IR), performing operator fusion, constant folding, and buffer allocation before execution.
2. **TorchScript & Tracing Limitations (PyTorch Team 2018–2020)**: Attempted whole-program capture via static AST parsing (`torch.jit.script`) or forward tracing (`torch.jit.trace`). However, TorchScript required non-idiomatic Python and failed abruptly on dynamic data structures, third-party libraries, and Python control flow.
3. **TorchDynamo & Bytecode Interception (Ansel et al. 2024 / PyTorch 2.0)**: Fundamentally shifted the compiler paradigm by intercepting Python frame evaluation at the CPython bytecode level (`PEP 523`), dynamically extracting computational subgraphs into FX graphs while safely falling back to the standard Python interpreter on unsupported bytecode instructions.
4. **AOTAutograd (Suhan et al. 2022)**: Traces backward autodiff graphs ahead-of-time, capturing joint forward-backward graphs before passing them to the code-generation backend.
5. **TorchInductor & Triton Code Generation**: High-performance compiler backend translating FX IR into optimized OpenAI Triton kernels for GPUs and OpenMP/C++ SIMD loops for CPUs, fusing point-wise, reduction, and normalization operations.

---

## 2. PyTorch 2 Compilation Architecture: Dynamo, AOTAutograd, and Inductor

```
  Python Program (Idiomatic PyTorch)
             │
             ▼
┌──────────────────────────────────────────┐
│ TorchDynamo (Frame Evaluation PEP 523)   │ ──► Graph Break on Dynamic Python Logic
└──────────────────────────────────────────┘
             │ Captured FX Graphs + Guards
             ▼
┌──────────────────────────────────────────┐
│ AOTAutograd (Ahead-Of-Time Reverse Mode) │
└──────────────────────────────────────────┘
             │ Joint Forward + Backward FX Graph
             ▼
┌──────────────────────────────────────────┐
│ TorchInductor (Backend Optimization Engine)│
└──────────────────────────────────────────┘
             │ Operator Fusion, Memory Scheduling
             ▼
   GPU: Fused Triton Kernels  |  CPU: Fused C++ OpenMP / Vectorized Loops
```

1. **Bytecode Analysis & Frame Guards**: Dynamo parses bytecode instructions and constructs **guards** (e.g. `tensor.dtype == torch.float32`, `tensor.ndim == 2`). If guards evaluate to true on subsequent invocations, the cached compiled graph is executed directly; if guards fail, Dynamo falls back to recompilation.
2. **Pointwise Operator Fusion**: Uncompiled eager mode executes $y = \text{SwiGLU}(x, g) = (x \odot \sigma(x)) \odot g$ through multiple independent kernel launches, each writing intermediate outputs back to DRAM. Fused compilation collapses this into a single thread-level register operation, achieving near-theoretical Roofline peak memory bandwidth.

---

## 3. Graph Breaks: Diagnosis, Causes, and Mitigation

A **graph break** occurs when TorchDynamo encounters Python bytecode that cannot be symbolically traced into an FX graph. Dynamo cuts the graph into two separate compiled subgraphs and bounces execution back to the CPython interpreter:
- **Major Causes of Graph Breaks**:
  1. Inspecting tensor values with `.item()`, `tensor.tolist()`, or Python `float(tensor)` inside model execution.
  2. Data-dependent control flow: `if tensor.sum() > 0: ...` evaluated directly in Python.
  3. Printing or logging tensor values: `print(x)`.
  4. Unsupported external libraries: passing tensors into non-traced NumPy, SciPy, or native C-extensions.
- **Diagnostic Tooling**:
  ```python
  explanation = torch._dynamo.explain(model)(*sample_inputs)
  print(f"Graph count: {explanation.graph_count}, Break reasons: {explanation.break_reasons}")
  ```
- **Mitigation with Branchless Select / `torch.cond`**:
  Replace data-dependent Python branches with `torch.where` or `torch.cond(pred, true_fn, false_fn, operands)`.

---

## 4. Compiler Modes & Backends

PyTorch 2 provides tailored compilation presets via `torch.compile(model, mode=..., backend=...)`:
1. **`mode="default"`**: Fuses point-wise and reduction kernels via TorchInductor. Fast compile times with solid 1.2–1.5x throughput speedups.
2. **`mode="reduce-overhead"`**: Leverages **CUDA Graphs** to eliminate CPython-GPU launch latency. Captures static memory pointers; requires fixed tensor addresses and non-dynamic shapes.
3. **`mode="max-autotune"`**: Enables exhaustive autotuning over Triton block sizes, tile configurations, and matrix multiplication backends (Triton vs CUTLASS). Takes longer to compile initially, but achieves maximal production serving throughput.

---

## 5. Multi-Ecosystem Implementation Matrix

| Compiler Concept | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Pointwise Fusion** | `torch.compile` Inductor | `jax.jit` XLA fusion | LLVM loop vectorization | `FUSED_SWIGLU_WGSL` |
| **Fused Add-Norm** | Fused C++/Triton kernel | XLA fused reduction | Vectorized single-pass loop | Fused Add+LayerNorm |
| **Conditional Branch** | `torch.cond` / `torch.where` | `jax.lax.cond` | Rust branchless `select` | Elementwise select |
| **Whole-Graph Capture** | `torch.compile(fullgraph=True)` | `@jax.jit` pure jaxpr | Ahead-of-Time rustc binary | Fused pipeline dispatch |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Catastrophic Overhead from Frequent Graph Breaks**:
   - *Symptom*: Compiled model runs significantly *slower* than eager PyTorch execution!
   - *Cause*: A loop containing `.item()` forces dozens of graph breaks per forward pass. Bouncing between Python and compiled runtime incurs massive trampoline dispatch overhead.
   - *Fix*: Eliminate all `.item()` and print calls; verify `explanation.graph_count == 1`.
2. **CUDA Graph Memory Corruption in `reduce-overhead`**:
   - *Symptom*: Inaccurate outputs or silent memory corruption during `mode="reduce-overhead"` execution.
   - *Cause*: Reallocating input tensor memory buffers rather than performing in-place `.copy_()` into persistent memory addresses.
   - *Fix*: Ensure static memory buffer reuse when replaying captured CUDA Graphs.
3. **Guard Churn and Recompilation Thrashing**:
   - *Symptom*: Continuous compilation stalls throughout training epochs.
   - *Cause*: Passing changing raw Python ints (e.g. `epoch`, `step`) as function arguments instead of 0D tensors or marking dynamic dimensions.
   - *Fix*: Use `torch._dynamo.mark_dynamic` or wrap scalar integers in tensors.

---

## 7. Real-World Engineering Applications

1. **Large Language Model Serving (vLLM, TensorRT-LLM)**: Fusing SwiGLU, RMSNorm, and attention projection epilogues into unified kernel invocations.
2. **Autonomous Driving Control Loops**: Achieving sub-millisecond 1kHz inference latencies on embedded edge hardware by eliminating Python interpreter jitter.
3. **Scientific PDE Solvers & High-Resolution DiT**: Fusing stencil calculations and spatial diffusion blocks to maximize GPU Tensor Core utilization.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.safetensors`:
1. `swiglu_ref`: Fused SwiGLU point-wise activation outputs.
2. `add_ln_ref`: Fused Add + Layer Normalization representation tensor.
3. `cond_ref`: Branchless conditional selection output.

### Synchronized Ecosystem Artifacts
- **Notebook:** [01_torch_compile_modes_backends_and_graph_breaks.ipynb](../../notebooks/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.ipynb)
- **PyTorch 2 Module:** [01_torch_compile_modes_backends_and_graph_breaks.py](../../pytorch2/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.py)
- **JAX Module:** [01_torch_compile_modes_backends_and_graph_breaks.py](../../jax/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.py)
- **Candle Module:** [01_torch_compile_modes_backends_and_graph_breaks.rs](../../candle/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.rs)
- **Hardware Kernel:** [01_torch_compile_modes_backends_and_graph_breaks.rs](../../kernels/12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.rs)
