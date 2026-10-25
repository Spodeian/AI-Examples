# TorchAO Modern API and Affine Quantized Tensor (AQT)

`torchao` (PyTorch Architecture Optimization) is PyTorch's native library for low-precision data types, quantization, and sparsity. Rather than relying on legacy module wrapping or intrusive graph rewrites, `torchao` introduces composable **Tensor Subclasses** via Python's `__torch_dispatch__` mechanism. The centerpiece of this design is the `AffineQuantizedTensor` (AQT), which packages quantized integer payloads, scales, zero-points, and block grouping parameters into a unified tensor object that behaves seamlessly within standard PyTorch eager and compiled workflows (`torch.compile`).

---

## 1. The `torchao` Philosophy: Composable Tensor Subclasses

Traditional quantization frameworks mutate model definitions by replacing `nn.Linear` layers with custom quantized module wrappers (e.g. `QuantizedLinear`). This breaks:
- Graph capture (`torch.export.export`)
- Distributed tensor parallel sharding (`DTensor`, `FSDP2`)
- Custom autograd and optimizer pipelines

`torchao` treats quantization as a property of the **tensor representation**, not the layer module. A model's weights remain standard `nn.Parameter` objects, but their underlying storage is swapped with an `AffineQuantizedTensor`. During the forward pass, PyTorch's dispatcher intercepts operations (such as `torch.ops.aten.mm.default`) and executes fused quantized kernels without requiring module changes.

---

## 2. Mathematical Formulation of `AffineQuantizedTensor` (AQT)

An affine-quantized tensor maps a floating-point matrix $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ into discrete integer codes $q \in [q_{\min}, q_{\max}]$ across blocks of size $G$ (e.g., $G=16, 32, 64, 128$):

$$
s_g = \frac{\max(W_g) - \min(W_g)}{q_{\max} - q_{\min}}, \quad zp_g = \text{round}\left( -\frac{\min(W_g)}{s_g} \right) + q_{\min}
$$

For symmetric quantization centered at zero ($zp_g = 0$):

$$
s_g = \frac{\max(|W_g|)}{q_{\max}}, \quad q = \text{clamp}\left( \left\lfloor \frac{W_g}{s_g} \right\rceil, q_{\min}, q_{\max} \right)
$$

### Dequantization Invariant:

$$
\hat{W}_g = (q_g - zp_g) \cdot s_g
$$

---

## 3. Weight-Only Quantization (W4A16 & W8A16)

In memory-bandwidth-bound inference regimes (e.g., auto-regressive token generation at batch size $B=1$), reading model weights from High Bandwidth Memory (HBM) to on-chip SRAM accounts for $>90\%$ of total token latency. Weight-only quantization compresses weights to 4 or 8 bits while keeping activations in 16-bit floating point (BF16/FP16):

$$
Y = X \cdot \hat{W} + b = X \cdot \left( (Q_W - zp_W) \cdot s_W \right) + b
$$

- **`Int8WeightOnlyConfig`**: Compresses weights by $2\times$ (8 bits/param), maintaining near-lossless perplexity across all model sizes.
- **`Int4WeightOnlyConfig(group_size=64)`**: Compresses weights by $3.5\times$ (packed 4 bits + block scales), fitting 70B models onto a single 48 GB workstation GPU.

---

## 4. Dynamic Activation & Weight Quantization (W8A8)

In compute-bound regimes (e.g., prompt prefill, high-concurrency serving with $B \ge 64$), arithmetic throughput dominates latency. `Int8DynamicActivationInt8WeightConfig` quantizes both activations and weights to 8-bit integers, executing high-throughput integer TensorCore matrix multiplication:

1. **Dynamic Activation Quantization (Per-Token):**
   $$s_X[b] = \frac{\max_j |X[b, j]|}{127}, \quad X_q[b, :] = \text{clamp}\left( \left\lfloor \frac{X[b, :]}{s_X[b]} \right\rceil, -128, 127 \right)$$
2. **Static Weight Quantization (Per-Channel):**
   $$s_W[c] = \frac{\max_i |W[i, c]|}{127}, \quad W_q[:, c] = \text{clamp}\left( \left\lfloor \frac{W[:, c]}{s_W[c]} \right\rceil, -128, 127 \right)$$
3. **Integer Accumulation & Mixed Output Rescaling:**
   $$Y = (X_q \cdot W_q) \odot (s_X \cdot s_W) + b$$

---

## 5. Performance Regime Crossover & Misuse Guardrails

| Metric | Memory-Bound Decoding ($B \le 8$) | Compute-Bound Prefill ($B \ge 64$) |
| :--- | :--- | :--- |
| **Limiting Resource** | HBM Memory Bandwidth | Arithmetic TensorCore FLOPs |
| **Optimal Format** | **Weight-Only W4A16 / W8A16** | **Dynamic W8A8 / FP8** |
| **Why?** | Memory transfer scales down $2\text{–}4\times$; dequant FLOPs hidden behind memory stalls. | INT8/FP8 TensorCores double GEMM throughput; integer accumulation dominates. |

> [!WARNING]
> **Misuse Alert:** Applying Weight-Only W4A16 during large-batch prefill ($B=256$) makes execution **slower than standard BF16** because dequantization adds extra overhead to an already compute-bound matrix multiplication!

---

## 6. `__torch_dispatch__` Architecture

When an `AffineQuantizedTensor` is passed to an ATen operator:

```python
class AffineQuantizedTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.mm.default:
            x, w_aqt = args
            # Fused W4/W8 dequant + GEMM kernel dispatch
            return fused_dequant_mm(x, w_aqt.int_data, w_aqt.scale)
        return super().__torch_dispatch__(func, types, args, kwargs)
```

This guarantees zero Python dispatch overhead under `torch.compile`, lowering directly to fused Inductor/Triton kernels.

---

## 7. Cross-Framework Numerical Parity

| Framework | Implementation File | Precision Parity |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/02_...py` | Reference |
| **JAX** | `jax/14_quantisation/02_pt2e_torchao_and_bitpacking/02_...py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/02_pt2e_torchao_and_bitpacking/02_...rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/02_pt2e_torchao_and_bitpacking/02_...rs` | WGSL / CPU |

---

## 8. Production Deployment with `autoquant`

`torchao.autoquant` automates quantization format selection across individual model layers:
1. Profiles candidate linear layers across multiple batch sizes.
2. Benchmarks candidate configs (`Int8WeightOnly`, `Int4WeightOnly`, `Int8DynamicActivation`).
3. Selects the Pareto-optimal format maximizing tokens/second while meeting SQNR criteria.
