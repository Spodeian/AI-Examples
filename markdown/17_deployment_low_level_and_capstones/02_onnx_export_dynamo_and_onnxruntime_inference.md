# ONNX Export: TorchDynamo and ONNX Runtime Inference

The **Open Neural Network Exchange (ONNX)** is the open standard format for machine learning interoperability. Modern deep learning workflows frequently train models in PyTorch or JAX on Linux GPU clusters, but deploy inference runtimes across edge devices, Windows PCs, mobile phones, or browser WebAssembly (WASM). **TorchDynamo-based ONNX export (`torch.onnx.export(..., dynamo=True)`)** replaces brittle legacy TorchScript tracing with robust FX graph capture, providing seamless dynamic shape tracking and high-performance execution via **ONNX Runtime (ORT)**.

---

## 1. The Interoperability Imperative

Deploying deep neural networks in production environments encounters severe friction when relying on the full Python PyTorch runtime:
- **Python Runtime Dependency:** Embedding CPython ($> 500\text{ MB}$ container footprint) inside latency-critical microservices introduces GIL contention and unpredictable garbage collection pauses.
- **Hardware Heterogeneity:** Edge accelerators (Intel OpenVINO, Qualcomm Snapdragon NPU, Apple CoreML, AMD ROCm) lack full PyTorch backend support.
- **Platform Portability:** C++, Rust, C#, Go, and JavaScript web applications require self-contained, statically linkable runtime engines.

ONNX decouples model architecture definitions from runtime execution engines via a standardized protobuf computation graph schema.

---

## 2. Legacy TorchScript Tracing vs. TorchDynamo Export

Historically, `torch.onnx.export()` relied on legacy TorchScript tracing (`torch.jit.trace`):
- **Data-Dependent Control Flow Silently Truncated:** `if x.sum() > 0:` was evaluated once during dummy input tracing and statically baked into the graph, discarding the alternate branch.
- **Static Dim Freezing:** Dynamic sequence lengths or batch sizes were frequently locked to the dummy input's static dimensions.

### The TorchDynamo Exporter (`dynamo=True`):
PyTorch 2.x re-architected ONNX export on top of **TorchDynamo FX Graph Capture**:
- Captures true computational semantics via Python bytecode analysis.
- Preserves full dynamic shape symbolism via `torch.export` symbolic shapes.
- Guarantees exact parity between PyTorch 2.x eager execution and the generated ONNX protobuf graph.

---

## 3. Dynamic Shapes & Symbolic Dimension Tracking

Real-world language and vision models process variable-length inputs (dynamic batch sizes $B$, variable sequence lengths $S$, and dynamic image resolutions $H \times W$).
In TorchDynamo ONNX export:
```python
dynamic_shapes = {
    "x": {0: torch.export.Dim("batch_size"), 1: torch.export.Dim("seq_len")}
}
torch.onnx.export(model, (dummy_input,), dynamic_shapes=dynamic_shapes, dynamo=True)
```
Symbolic dimensions prevent ONNX Runtime from throwing shape mismatch errors when inference request dimensions differ from training traces.

---

## 4. ONNX Runtime (ORT) & Execution Providers

**ONNX Runtime (ORT)** executes ONNX graphs across diverse silicon through a modular **Execution Provider (EP)** abstraction:
- **CPU Execution Provider:** Leverages AVX2, AVX-512, and ARM NEON SIMD intrinsics.
- **CUDA / TensorRT Execution Provider:** Maps ONNX subgraphs directly to NVIDIA Tensor Cores and CUDA streams.
- **DirectML Execution Provider:** Cross-vendor hardware-accelerated DirectX 12 execution on Windows (supporting AMD, Intel, and NVIDIA GPUs).
- **WebAssembly (WASM) / WebGPU:** Executes client-side neural networks directly inside web browsers.

---

## 5. Automated Graph Optimizations in ORT

Prior to execution, ONNX Runtime applies aggressive hardware-agnostic and provider-specific graph transformations:
1. **Constant Folding:** Statically computes expressions involving weights and constants at initialization time.
2. **LayerNorm / RMSNorm Fusion:** Combines reduce-mean, subtract, square, reduce-mean, add epsilon, rsqrt, and multiply into a single optimized kernel.
3. **GEMM + Bias + Activation Fusion:** Fuses linear transformations and activations (e.g. GELU, ReLU) into a single compute pass, eliminating intermediate global memory roundtrips.

---

## 6. Cold-Start & Throughput Comparison

| Runtime Engine | Python Dependency | Dynamic Batch/Seq | Cold Start Latency | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch Eager (Python)** | Required (CPython) | Native | $\sim 1,200\text{ ms}$ | High ($> 1\text{ GB}$) |
| **TorchScript JIT** | Optional (C++ LibTorch) | Brittle | $\sim 450\text{ ms}$ | Moderate |
| **ONNX Runtime (CPU EP)** | None (Pure C/C++/Rust) | Native | **$< 25\text{ ms}$** | **Minimal ($< 50\text{ MB}$)** |
| **ONNX Runtime (CUDA EP)** | None (Pure C/C++/Rust) | Native | **$< 50\text{ ms}$** | **Minimal** |

---

## 7. Multi-Framework Parity & Verification

Mathematical consistency is validated across PyTorch 2.x, JAX, Candle, and hardware kernels:
- **PyTorch 2.x:** Simulates dynamic shape execution on $B=1, S=4$ and $B=2, S=2$ configurations.
- **JAX:** Functional implementation of dynamic shape forward pass using `jax.nn.gelu`.
- **Candle:** High-performance Rust implementation using flattened 3D-by-2D matrix multiplication and `gelu_erf()`.
- **WGSL Kernel:** Implements `FUSED_DYNAMO_GELU_BIAS_WGSL` for GPU-accelerated fused bias addition and GELU activation.

Parity tests guarantee absolute error $< 10^{-5}$ across all framework outputs against the golden checkpoint.

---

## 8. Failure Modes, Pathology, and Debugging

| Failure Mode | Root Cause | Observable Symptom | Mitigation / Remedy |
| :--- | :--- | :--- | :--- |
| **Static Shape Freezing** | Exporting without specifying dynamic axes | `InvalidArgument: Got input shape [2, 8] but expected [1, 4]` | Specify `dynamic_shapes` explicitly during `torch.onnx.export`. |
| **Unsupported Python Control Flow** | Using un-exportable Python state mutations | Graph capture error: `TorchDynamo unable to trace` | Refactor code to use PyTorch tensor control flow (`torch.where`, `torch.cond`). |
| **Data Type Discrepancy** | Exporting FP32 model and querying with FP16 inputs in ORT | Runtime error: `Type mismatch in input tensor` | Enforce explicit type casting or export quantized INT8 / FP16 graphs. |
| **Silent Operator Fallback** | Execution provider lacks custom kernel for specific operator | Severe latency degradation as ORT falls back to CPU EP | Verify execution provider node placement with `ort.SessionOptions.log_severity_level = 0`. |
