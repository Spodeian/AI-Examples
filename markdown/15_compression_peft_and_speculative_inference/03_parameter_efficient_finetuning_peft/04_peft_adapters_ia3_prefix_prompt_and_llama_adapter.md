# Parameter-Efficient Adapters: (IA)^3, Prefix-Tuning, Prompt-Tuning, and LLaMA-Adapter

While low-rank factorization methods (LoRA, AdaLoRA, DoRA) adapt models by inserting low-rank matrix pairs into linear projections, alternative PEFT paradigms operate on activations directly or prepend learnable continuous virtual tokens. This topic explores four foundational activation- and prompt-level adaptation architectures: **$(IA)^3$** (rescaling inner activations via learned vectors), **Prefix-Tuning** (prepending virtual keys and values), **Prompt-Tuning** (prepending virtual embeddings), and **LLaMA-Adapter** (zero-initialized gated prefix attention).

---

## 1. Taxonomy of Parameter-Efficient Adaptation: Beyond Matrix Factorization

Rather than reparameterizing dense weight matrices $\Delta W = B A$, activation and prefix adapters modulate intermediate transformer tensors:

| Adaptation Strategy | Target Injection Point | Trainable Parameters | Parameter Ratio vs Base | Key Citation |
| :--- | :--- | :--- | :--- | :--- |
| **$(IA)^3$** | Key, Value, & FFN activations | 3 vectors per layer ($l_k, l_v, l_{\text{ffn}}$) | **$< 0.01\%$** ($\sim 10\times$ less than LoRA) | Liu et al. (NeurIPS 2022) |
| **Prefix-Tuning** | Attention Key & Value sequences | $2 \times L_{\text{prefix}} \times d$ per layer | $\approx 0.1\% - 0.5\%$ | Li & Liang (ACL 2021) |
| **Prompt-Tuning** | Input embedding sequence | $L_{\text{prompt}} \times d$ (Layer 0 only!) | **$< 0.001\%$** (model-depth agnostic) | Lester et al. (EMNLP 2021) |
| **LLaMA-Adapter** | Attention with zero-gate $g$ | $2 \times L_{\text{prefix}} \times d + 1$ | $\approx 0.1\%$ | Zhang et al. (ICLR 2023) |

---

## 2. (IA)^3: Infused Adapter by Inhibiting and Amplifying Inner Activations

**$(IA)^3$ (Liu et al., NeurIPS 2022)** introduces element-wise multiplicative scaling vectors directly into the self-attention and feed-forward intermediate representations:

### Attention Modulation
$$K_{\text{adapted}} = K \odot l_k, \quad V_{\text{adapted}} = V \odot l_v$$
where $l_k \in \mathbb{R}^d$ and $l_v \in \mathbb{R}^d$ are learned vectors initialized to ones ($l_k = \mathbf{1}, l_v = \mathbf{1}$).

### FFN Modulation
For intermediate two-layer feed-forward networks:
$$\text{FFN}(x) = \left(\sigma(x W_1) \odot l_{\text{ffn}}\right) W_2$$
where $l_{\text{ffn}} \in \mathbb{R}^{d_{\text{ffn}}}$ modulates hidden neuron activations before down-projection.

### Zero-Latency Weight Merging
Because scaling is linear and associative, during inference the vectors can be absorbed into the base frozen weights:
$$W_k^{\text{merged}} = W_k \operatorname{diag}(l_k), \quad W_v^{\text{merged}} = W_v \operatorname{diag}(l_v), \quad W_1^{\text{merged}} = W_1 \operatorname{diag}(l_{\text{ffn}})$$
This provides **exact zero-latency inference** with no architectural modifications.

---

## 3. Prefix-Tuning: Continuous Virtual Context in Attention Space

**Prefix-Tuning (Li & Liang, 2021)** prepends $L_{\text{prefix}}$ learnable virtual continuous tokens to the Key and Value matrices across every transformer layer.

```
Query Q:      [   q_1   ,   q_2   ,  ...  ,   q_S   ]
Key K:        [  P_K,1  ,  P_K,2  |   k_1   ,  ...  ,   k_S   ]
Value V:      [  P_V,1  ,  P_V,2  |   v_1   ,  ...  ,   v_S   ]
               └──── Prefix ────┘ └──── Input Tokens ────┘
```

### Mathematical Formulation
$$K_{\text{cat}} = \left[ P_K \,;\, K \right] \in \mathbb{R}^{B \times (L_p + S) \times d}$$
$$V_{\text{cat}} = \left[ P_V \,;\, V \right] \in \mathbb{R}^{B \times (L_p + S) \times d}$$
$$\text{Attention}(Q, K_{\text{cat}}, V_{\text{cat}}) = \operatorname{softmax}\left(\frac{Q K_{\text{cat}}^\top}{\sqrt{d}}\right) V_{\text{cat}}$$

Unlike hard prompt text, virtual prefix embeddings $P_K, P_V$ are not restricted to real vocabulary tokens and occupy continuous semantic coordinates.

---

## 4. Prompt-Tuning & P-Tuning: Soft Input Prompting

**Prompt-Tuning (Lester et al., 2021)** restricts learnable parameters strictly to the input embedding layer:
$$X_{\text{prompt}} = \left[ P_1, P_2, \dots, P_{L_p} \,;\, E(t_1), E(t_2), \dots, E(t_S) \right]$$
where $P \in \mathbb{R}^{L_p \times d}$ is prepended to the token embeddings before Layer 1. All transformer blocks ($1 \dots L$) remain 100% frozen.

- **Scale Invariance:** As model size scales past $10\text{B}+$ parameters, prompt-tuning matches the performance of full model fine-tuning.
- **Serving Efficiency:** A single frozen foundation model can serve thousands of distinct tasks concurrently by simply concatenating task-specific prefix vectors to the user request.

---

## 5. LLaMA-Adapter: Zero-Initialized Gating and Multi-Modal Adaptation

Standard prefix-tuning randomly initializes $P_K, P_V$. In early training steps, random attention scores on the prefix tokens disrupt the pre-trained attention distributions, destabilizing early gradients.

**LLaMA-Adapter (Zhang et al., 2023)** resolves this via **Zero-Initialized Gating**:

```
Keys:   [ P_K ; K ]  ──►  Softmax Attention Scores
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       S_prefix [P_len]               S_seq [Seq_len]
               │                             │
        @ Prefix Values               @ Base Values
               │                             │
            * gate g                         │
               │                             │
               └──────────────► + ◄──────────┘
                                │
                         Adapted Output
```

### Governing Equations
1. Concatenate adapter prefix keys: $K_{\text{cat}} = [P_K \,;\, K]$.
2. Compute full softmax attention:
   $$A = \operatorname{softmax}\left(\frac{Q K_{\text{cat}}^\top}{\sqrt{d}}\right) = \left[ A_{\text{prefix}} \;\Big|\; A_{\text{seq}} \right]$$
3. Apply zero-gated accumulation:
   $$\text{Output} = A_{\text{seq}} V + g \cdot \left(A_{\text{prefix}} P_V\right)$$
where $g \in \mathbb{R}$ is a learnable scalar initialized to $0.0$. At step 0, $g = 0 \implies \text{Output} = A_{\text{seq}} V$, perfectly preserving the original pre-trained model outputs.

---

## 6. Mathematical Comparison: Parameter Footprint, FLOPs, and Serving

The following table summarizes resource consumption for a 7B parameter LLM ($L = 32, d = 4096, d_{\text{ffn}} = 11008$):

| Strategy | Trainable Parameters | Storage per Checkpoint | FLOPs Overhead during Training | KV Cache Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | $7.0 \times 10^9$ | $14.0\text{ GB}$ (FP16) | $+0\%$ | None |
| **LoRA ($r=8$)** | $4.2 \times 10^6$ | $8.4\text{ MB}$ | $+2\%$ | None |
| **$(IA)^3$** | **$0.6 \times 10^6$** | **$1.2\text{ MB}$** | **$< 0.1\%$** | None |
| **Prefix-Tuning ($L_p=10$)** | $2.6 \times 10^6$ | $5.2\text{ MB}$ | $+3\%$ | $+10$ tokens/layer |
| **LLaMA-Adapter ($L_p=10$)** | $2.6 \times 10^6$ | $5.2\text{ MB}$ | $+3\%$ | $+10$ tokens/layer |
| **Prompt-Tuning ($L_p=20$)** | **$8.2 \times 10^4$** | **$164\text{ KB}$** | **$0\%$** | $+20$ tokens (input) |

---

## 7. Hardware & Kernel Acceleration: Fused Zero-Gating & Vector Rescaling

In high-throughput multi-tenant serving, $(IA)^3$ and LLaMA-Adapter can be executed in fused hardware kernels:

```wgsl
// WGSL Fused LLaMA-Adapter Zero-Gated Recombination
struct Uniforms {
    total_elements: u32,
    gate: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> seq_out: array<f32>;
@group(0) @binding(2) var<storage, read> prefix_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn llama_adapter_gate_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.total_elements) { return; }
    y[idx] = seq_out[idx] + u.gate * prefix_out[idx];
}
```

---

## 8. Implementation Guidelines & Cross-Framework Parity

To maintain transitive numerical parity ($< 10^{-5}$) across PyTorch, JAX, and Candle:

1. **Exact Scaled Attention Softmax**: Calculate $1 / \sqrt{d}$ in double precision before downcasting to single precision to avoid slight exponent divergence in softmax.
2. **Dimension Partitioning**: In LLaMA-Adapter, ensure attention tensor slicing $\left[ A_{\text{prefix}} \mid A_{\text{seq}} \right]$ along dimension 2 uses contiguous strides matching between JAX (`[:, :, :p_len]`), PyTorch (`narrow(2, 0, p_len)`), and Candle (`narrow(2, 0, p_len)`).
3. **Seed-42 Golden Parity**: All parameters ($q, k, v, \text{prefix\_k}, \text{prefix\_v}, \text{ia3\_scale}, \text{gate}, \dots$) are strictly verified against golden safetensors.
