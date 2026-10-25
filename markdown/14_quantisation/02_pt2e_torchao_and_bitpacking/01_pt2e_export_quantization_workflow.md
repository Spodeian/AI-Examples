# PyTorch 2 Export (PT2E) Quantization Workflow: Operator Fusion, Q/DQ Graph Lowering, and Hardware Backend Targeting

The PyTorch 2 Export Quantization (PT2E) workflow is the modern standard for deploying post-training quantized (PTQ) and quantization-aware trained (QAT) neural networks to edge hardware, mobile runtimes, and high-performance server accelerators. Unlike legacy PyTorch eager-mode quantization (which relied on fragile module swaps and Python hooks), PT2E operates on a clean, sound, functionalized `ExportedProgram` captured via `torch.export.export`. PT2E performs exact algebraic operator fusions (such as Conv-BatchNorm and Linear-BatchNorm folding), inserts simulated Quantize-Dequantize (`torch.ops.quantized_decomposed.quantize_per_tensor`) nodes during `prepare_pt2e`, records calibration statistics, and lowers the graph into hardware-aligned fused operator patterns during `convert_pt2e`.

---

## 1. PyTorch 2 Export Architecture & FX Graph Representation

PT2E separates quantization into three distinct, mathematically rigorous compiler phases:

1. **Graph Capture:** `torch.export.export(model, sample_inputs)` captures an `ExportedProgram` containing a strict ATen-level functional FX graph with explicit dynamic shape constraints.
2. **Annotation & Preparation (`prepare_pt2e`):** A backend-specific `Quantizer` (e.g., `XNNPACKQuantizer`) traverses the graph, annotating pairs of producer/consumer nodes with quantization specs (per-tensor or per-channel scales and zero-points), and inserting observation nodes.
3. **Graph Lowering (`convert_pt2e`):** Observers are finalized into constant scales, and Q/DQ pairs around linear operators are lowered into fused quantized operator kernels (e.g. `xnnpack::qlinear`, `onednn::qconv`).

---

## 2. Algebraic Conv-BatchNorm and Linear-BatchNorm Folding

Before quantizing weight tensors, all preceding or succeeding affine batch normalization layers must be folded into the weights. Consider a 2D convolution followed by 2D Batch Normalization:

$$
y_{\text{conv}} = x * W_{\text{conv}} + b_{\text{conv}}
$$
$$
y_{\text{bn}} = \frac{y_{\text{conv}} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

Substituting $y_{\text{conv}}$ into the BatchNorm equation:

$$
y_{\text{bn}} = \left( x * W_{\text{conv}} + b_{\text{conv}} - \mu \right) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
$$
y_{\text{bn}} = x * \underbrace{\left( W_{\text{conv}} \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \right)}_{W_{\text{folded}}} + \underbrace{\left( (b_{\text{conv}} - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta \right)}_{b_{\text{folded}}}
$$

For output channel $c$:
- **Folded Scale Factor:** $s_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}$
- **Folded Weights:** $W_{\text{folded}}[c, :, :, :] = W_{\text{conv}}[c, :, :, :] \cdot s_c$
- **Folded Bias:** $b_{\text{folded}}[c] = (b_{\text{conv}}[c] - \mu_c) \cdot s_c + \beta_c$

This eliminates the runtime BatchNorm operator entirely, reducing memory bandwidth by $50\%$ and removing an entire kernel dispatch.

---

## 3. The Unfused Quantization Pitfall: Noise Amplification

A critical mistake in naive quantization pipelines is quantizing convolution weights *before* folding BatchNorm:

$$
\hat{W}_{\text{unfused}} = Q(W_{\text{conv}}) = W_{\text{conv}} + \Delta W
$$

When this quantized weight passes through the subsequent BatchNorm operator, the quantization noise $\Delta W$ is multiplied by $s_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}$:

$$
\text{Error}_{\text{unfused}} = \Delta W \cdot s_c = \Delta W \cdot \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}
$$

In modern deep networks, running variance $\sigma_c^2$ can be as small as $0.01$, yielding $s_c \approx 10.0$. Thus, unfused quantization **amplifies weight quantization noise by $10\times$ (or $+20\text{ dB}$ noise floor!)**, leading to severe accuracy collapse.

When Conv-BN is folded *before* quantization:

$$
\hat{W}_{\text{folded}} = Q(W_{\text{folded}}) = W_{\text{folded}} + \Delta W'
$$

The quantization grid discretizes the true effective operating weights, and noise $\Delta W'$ remains strictly bounded by $\frac{\Delta}{2}$ with zero scale magnification.

---

## 4. Simulated Q/DQ Graph Lowering

During the `prepare_pt2e` and `convert_pt2e` pipeline, simulated quantization is modeled via symmetric INT8 Quantize-Dequantize operators:

$$
Q(x; s) = \text{clamp}\left( \left\lfloor \frac{x}{s} \right\rceil, -128, 127 \right) \cdot s
$$

The lowered graph executes:

$$
\hat{y} = Q(x; s_x) * Q(W_{\text{folded}}; s_w) + b_{\text{folded}}
$$

In integer hardware backends, this is computed as integer matrix multiplication with dyadic rescaling:

$$
q_y = \left( \sum q_x \cdot q_w \right) \cdot \frac{s_x \cdot s_w}{s_y} + q_b
$$

---

## 5. Backend Quantizer Specifications: XNNPACK vs. TensorRT

Different deployment targets impose specific structural constraints on quantization graphs:

| Backend | Activation Spec | Weight Spec | Bias Treatment | Operator Fusion |
| :--- | :--- | :--- | :--- | :--- |
| **XNNPACK (Mobile/CPU)** | Per-tensor symmetric INT8 | Per-channel symmetric INT8 | FP32 or INT32 unquantized | Fused Conv+ReLU / Linear+GELU |
| **TensorRT (NVIDIA GPU)** | Per-tensor symmetric INT8 | Per-channel symmetric INT8 | FP32 | Fused Conv+Bias+Activation |
| **Qualcomm QNN (NPU)** | Per-tensor asymmetric UINT8 | Per-channel asymmetric UINT8 | INT32 with zero-point offset | Fused NPU Tile Graph |

---

## 6. Graph Transformation Verification & Invariants

To verify compiler correctness across PT2E lowering passes, the following invariants are enforced:

1. **Mathematical Equivalence of Folding:**
   $$\| y_{\text{bn}}(y_{\text{conv}}(x)) - y_{\text{folded}}(x) \|_\infty < 10^{-5}$$
2. **Dynamic Range Preservation:**
   $$s_w = \frac{\max_c |W_{\text{folded}}[c]|}{127}$$
3. **Deterministic Seed Parity:**
   Outputs across PyTorch 2, JAX, and Candle must agree to within $10^{-5}$ max absolute error against `01_pt2e_export_quantization_workflow.safetensors`.

---

## 7. Cross-Framework Numerical Parity

| Framework | Implementation File | Precision Parity |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/01_...py` | Reference |
| **JAX** | `jax/14_quantisation/02_pt2e_torchao_and_bitpacking/01_...py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/02_pt2e_torchao_and_bitpacking/01_...rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/02_pt2e_torchao_and_bitpacking/01_...rs` | WGSL / CPU |

---

## 8. Production Deployment & PT2E Best Practices

1. **Always Export in `eval` Mode:** `model.eval()` freezes running mean $\mu$ and running variance $\sigma^2$ in BatchNorm layers; exporting in `train` mode captures dynamic statistical updates that cannot be folded statically.
2. **Avoid Manual Module Mutation:** Rely on `prepare_pt2e` and `convert_pt2e` rather than manual weight overwrites to ensure FX metadata and downstream Inductor patterns remain sound.
3. **Verify with Representative Calibration Tensors:** Pass realistic domain inputs through the prepared model so observer scales accurately reflect production distribution dynamic ranges.
