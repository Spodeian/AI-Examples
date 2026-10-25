# Custom Tensor Subclasses Quantized and Tracked Tensors

## 1. Architectural Role & Context

In PyTorch 2.x, the extension of tensor semantics has shifted away from monolithic C++ engine modifications toward compositional Python wrapper subclasses. Using `torch.Tensor._make_wrapper_subclass`, developers can construct first-class `torch.Tensor` objects that carry arbitrary custom metadata, alternate physical layouts, or multi-field numerical duals while behaving transparently inside existing PyTorch neural network modules.

This architectural pattern forms the foundation of modern PyTorch ecosystem primitives:
1. **`torch.distributed._tensor.DTensor`:** Wraps shards across multi-GPU process meshes.
2. **`torchao` (Architecture Optimization):** Implements `AffineQuantizedTensor` for transparent INT4/INT8/FP8 weights.
3. **`DualTensor`:** Implements forward-mode automatic differentiation ($x + \epsilon \dot{x}$) to compute Jacobian-Vector Products ($JVP$) in a single forward pass without unrolling backward computational graphs.
4. **`LazyDequantTensor`:** Stores weights in compact quantized representations and dequantizes on-the-fly during linear dispatch.

---

## 2. Mathematical Formalism

### 2.1 Dual Numbers & Forward-AD Algebra
Let $\mathbb{D} = \{x + \epsilon \dot{x} \mid x, \dot{x} \in \mathbb{R}, \epsilon^2 = 0\}$ represent the ring of dual numbers:
- **Matrix Multiplication:**
  $$(A + \epsilon \dot{A})(B + \epsilon \dot{B}) = AB + \epsilon(A\dot{B} + \dot{A}B)$$
- **Affine Addition:**
  $$(X + \epsilon \dot{X}) + (Y + \epsilon \dot{Y}) = (X + Y) + \epsilon(\dot{X} + \dot{Y})$$
- **Nonlinear Activation ($f(x) = \text{silu}(x)$):**
  $$f(x + \epsilon \dot{x}) = f(x) + \epsilon \cdot f'(x) \dot{x}$$

### 2.2 Transparent Dequantization Algebra
Given an INT8 tensor $W_{\text{int}} \in [-128, 127]^{K \times N}$, per-tensor scale $s \in \mathbb{R}_{>0}$, and zero-point $z \in \mathbb{R}$:
$$W_{\text{fp}} = (W_{\text{int}} - z) \cdot s$$
When a `DualTensor` interacts with a `LazyDequantTensor`:
$$Y_{\text{dual}} = X_{\text{dual}} W_{\text{fp}} + b = (X_p W_{\text{fp}} + b) + \epsilon (X_t W_{\text{fp}})$$

---

## 3. Algorithmic Implementation & Subclass Construction

```python
class DualTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, primal, tangent):
        return torch.Tensor._make_wrapper_subclass(
            cls, primal.shape, dtype=primal.dtype, device=primal.device
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Dispatches primal op and evaluates tangent derivative rules
        ...
```

---

## 4. Cross-Track Verification & Parity

All operations are cross-verified across PyTorch 2.x, JAX, Rust Candle, Hardware Kernels, and Jupyter Notebooks against seed-42 golden weights:
- Linear anchor $\|Y_{\text{track}} - Y_{\text{golden}}\|_\infty < 10^{-5}$.
- Dequantized weight tensor $\|W_{\text{dequant}} - W_{\text{golden}}\|_\infty < 10^{-5}$.
- Forward dual primal output and tangent $JVP$ perturbation $\|Y_{\text{primal}} - Y_{\text{golden}}\|_\infty < 10^{-5}$ and $\|Y_{\text{tangent}} - Y_{\text{golden}}\|_\infty < 10^{-5}$.

---

## 5. Failure Modes & Misuse Guardrails

| Misuse Mode | Mechanism | Operational Guardrail |
| :--- | :--- | :--- |
| **Breaking View Autograd Metadata** | Subclassing via standard Python `__init__` instead of `_make_wrapper_subclass` breaks view aliasing and gradient propagation. | Always use `torch.Tensor._make_wrapper_subclass` with explicit `strides` and `storage_offset`. |
| **Materializing Eager Dequantization** | Dequantizing the entire model weights upfront defeats the memory savings of INT8/INT4 storage. | Implement lazy operator-level dequantization inside `__torch_dispatch__`. |
| **Missing Tangent Rule for New Ops** | Calling an unsupported Aten op on a `DualTensor` silently zeroes the tangent field. | Raise an explicit `NotImplementedError` when an op has no registered forward derivative rule. |

---

## 6. Roofline Analysis & Hardware Performance

- **Memory Efficiency:** INT8 quantized weight subclasses cut weight memory traffic from 4 bytes/param to 1 byte/param, increasing effective decode memory bandwidth by $4\times$.
- **Forward-AD Efficiency:** Computing $JVP$ via `DualTensor` requires exactly $1\times$ forward pass memory, eliminating the need to cache intermediate activations for backpropagation.

---

## 7. Cross-Domain Transfer & Isomorphisms

| Domain | Wrapper Subclass Analogue | Operational Goal |
| :--- | :--- | :--- |
| **Scientific Computing** | Physical Unit-Checked Quantities (AstroPy / Pint) | Prevent dimensional analysis errors (e.g., adding meters to seconds) at compile time |
| **Graphics & Robotics** | Dual Quaternions ($\mathbb{H} + \epsilon \mathbb{H}$) | Seamless representation of rigid body rotation and translation without gimbal lock |
| **Cryptographic ML (ZKML)** | Modular Field Integer Tensor Subclasses | Execute neural operations over finite prime fields $\mathbb{F}_p$ for cryptographic proofs |

---

## 8. References & Further Reading

1. PyTorch Core Team. (2022). *Subclassing torch.Tensor with __torch_dispatch__*. PyTorch Developer Guides.
2. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
3. Ansel, J., et al. (2024). *PyTorch 2 Architecture and Subclass Dispatches*. ASPLOS.
