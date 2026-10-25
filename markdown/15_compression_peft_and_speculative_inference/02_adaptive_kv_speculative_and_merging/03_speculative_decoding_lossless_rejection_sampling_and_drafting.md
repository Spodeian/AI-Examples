# Speculative Decoding: Lossless Rejection Sampling and Drafting Paradigms

Autoregressive decoding generates text token-by-token: producing $N$ tokens requires $N$ sequential memory reads of the entire multi-billion-parameter model from GPU High Bandwidth Memory (HBM). Because matrix-vector multiplications during single-token decoding achieve only a fraction of peak arithmetic intensity, the GPU sits memory-bandwidth starved. **Speculative decoding (Leviathan et al., 2023; Chen et al., 2023)** resolves this bottleneck by decoupling token generation into two phases: a lightweight **drafting mechanism** generates $\gamma$ candidate tokens sequentially at high speed, followed by a single parallel forward pass of the large **target model** that simultaneously verifies all $\gamma$ tokens. Crucially, lossless rejection sampling guarantees that the output token distribution is mathematically identical to directly sampling from the target model.

---

## 1. Memory-Bound Generation & The Speculative Decoding Paradigm

The operational intensity of standard single-token decoding is $\approx 1 \text{ FLOP / byte}$, while modern hardware (e.g. NVIDIA H100) requires $>150 \text{ FLOPs / byte}$ to saturate Tensor Cores. Processing $\gamma > 1$ tokens in parallel transforms matrix-vector operations into matrix-matrix operations (GEMM), boosting arithmetic intensity by $\gamma\times$ at nearly identical latency.

| Decoding Paradigm | Sequential Target Passes per $\gamma$ Tokens | Verification Latency | Output Distribution | Hardware Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Autoregressive** | $\gamma$ | $\gamma \times T_{\text{target}}$ | Target exact | HBM Bandwidth Bound |
| **Speculative Decoding** | $1$ | $\gamma \times T_{\text{draft}} + T_{\text{target}}$ | Target exact (provably lossless) | Compute Bound ($\gamma \times$ higher intensity) |
| **Non-Autoregressive (NAR)** | $1$ | $T_{\text{NAR}}$ | Approximate / Degraded | Structural Multimodality Collapse |

---

## 2. Lossless Speculative Rejection Sampling: Mathematical Proof of Parity

Let $p(x) = P_{\text{target}}(x \mid x_{<i})$ and $q(x) = P_{\text{draft}}(x \mid x_{<i})$. For a draft token $\tilde{x} \sim q(x)$:

### 1. Acceptance Criterion
Draw uniform random variable $u \sim \mathcal{U}(0, 1)$:
$$\text{Accept } \tilde{x} \iff u \le \min\left(1, \frac{p(\tilde{x})}{q(\tilde{x})}\right)$$

### 2. Residual Distribution upon Rejection
If $\tilde{x}$ is rejected, discard all subsequent draft tokens $\tilde{x}_{>i}$ and sample replacement token $x$ from the normalized residual distribution:
$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y) - q(y))}$$

### 3. Proof of Exact Distributional Equivalence
The probability of emitting token $x$ is the sum of accepting it as a draft and sampling it after rejection:
$$\begin{aligned}
P(\text{emit } x) &= q(x) \min\left(1, \frac{p(x)}{q(x)}\right) + \left(1 - \sum_y q(y) \min\left(1, \frac{p(y)}{q(y)}\right)\right) p'(x) \\
&= \min(q(x), p(x)) + \left(1 - \sum_y \min(q(y), p(y))\right) \frac{\max(0, p(x) - q(x))}{\sum_y (p(y) - \min(q(y), p(y)))} \\
&= \min(q(x), p(x)) + \max(0, p(x) - q(x)) \\
&= p(x)
\end{aligned}$$
Thus, speculative rejection sampling is **strictly lossless**: no temperature distortion, no divergence, and no quality degradation.

---

## 3. Theoretical Speedup, Acceptance Rates, and Lookahead Horizon ($\gamma$)

The acceptance probability for a token is $\beta = 1 - D_{\text{TV}}(p, q)$, where $D_{\text{TV}}$ is the total variation distance.

### Expected Number of Accepted Tokens
For a speculative horizon of $\gamma$ tokens, the expected number of generated tokens per target pass (including the bonus/residual token) is:
$$\mathbb{E}[N] = \sum_{i=1}^\gamma \beta^i + 1 = \frac{1 - \beta^{\gamma+1}}{1 - \beta}$$

### Wall-Clock Speedup
Given cost ratio $c = \frac{\text{Latency}(M_{\text{draft}})}{\text{Latency}(M_{\text{target}})} \ll 1$:
$$\text{Speedup}(\gamma, \beta, c) = \frac{\mathbb{E}[N]}{\gamma \cdot c + 1} = \frac{1 - \beta^{\gamma+1}}{(1 - \beta)(\gamma c + 1)}$$

```
Speedup Factor
  3.0x ────                 ● Peak at γ=5 (Speedup: 2.85x, β=0.85)
  2.5x ───              ●
  2.0x ──       ●
  1.5x ──   ●
  1.0x ──●────────────────────────────────────── Baseline Autoregressive
       1    2    3    4    5    6    7    8   (Lookahead γ)
```

If $\gamma$ is too small, speculative opportunities are left on the table; if $\gamma$ is too large, the verification cost of unused draft steps diminishes returns.

---

## 4. Drafting Strategies: Small Draft Models, Quantized Ensembles, and n-Gram Lookups

1. **Independent Small Draft Models**:
   - Pair a 70B target model with a 7B or 1B draft model from the same tokenizer family (e.g. LLaMA-3-70B + LLaMA-3-8B).
   - Achieves $\beta \in [0.70, 0.85]$ across diverse conversational tasks.
2. **Prompt Lookup Decoding (Zero Parameters)**:
   - Identifies recurring n-grams in the input context (code repetition, document quoting, JSON structure) and copies following tokens directly.
   - Zero model weights required; achieves up to $3\times$ speedup on structured coding and document QA tasks.
3. **Quantized Self-Drafting**:
   - The target model itself running in 4-bit INT4 quantization acts as the draft model for its full-precision FP16 counterpart.

---

## 5. Parallel Target Verification & Attention Tree Masks

When verifying $\gamma$ draft tokens in the target model, naive linear drafting forces a single sequence path. If token 2 is rejected, tokens 3 and 4 are wasted.

```
Standard Linear Draft:  [ Prompt ] ──► T1 ──► T2 ──► T3 ──► T4

Tree Speculative Draft: [ Prompt ] ──► T1 ──┬──► T2a ──► T3a
                                            └──► T2b ──► T3b
```

Modern tree-based drafting constructs a branching candidate tree. By customizing the 2D causal attention mask:
$$M_{ij} = \begin{cases} 0 & \text{if token } j \text{ is an ancestor of token } i \\ -\infty & \text{otherwise} \end{cases}$$
the target model simultaneously verifies multiple candidate branches in a single parallel forward execution pass!

---

## 6. Multi-Framework Parity & Verification (PyTorch, JAX, Candle)

The lossless rejection sampling step is numerically cross-validated across PyTorch, JAX, and Candle:

```rust
// Candle Speculative Verification Step
let (acc_mask, n_acc, res_p) = topic::speculative_rejection_step(
    &w["target_probs"], &w["draft_probs"], &w["draft_tokens"], &w["uniform_rands"]
)?;
common::assert_max_diff(&acc_mask, &w["accepted_mask"], 1e-5)?;
common::assert_max_diff(&n_acc, &w["num_accepted"], 1e-5)?;
common::assert_max_diff(&res_p, &w["residual_probs"], 1e-5)?;
```

Deterministic uniform pseudo-random inputs confirm transitive equivalence down to $< 10^{-5}$ maximum difference.

---

## 7. GPU/WebGPU Kernel Verification & Memory Optimizations

The sequential rejection check can be written as an early-break GPU workgroup kernel in WGSL:

```wgsl
@compute @workgroup_size(64, 1, 1)
fn speculative_verify_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= u.batch) { return; }
    var all_prev_accepted = true;
    for (var i = 0u; i < u.gamma; i++) {
        let t_id = u32(draft_tokens[b * u.gamma + i]);
        let p_val = target_probs[(b * u.gamma + i) * u.vocab + t_id];
        let q_val = max(draft_probs[(b * u.gamma + i) * u.vocab + t_id], 1e-8);
        let ratio = min(1.0, p_val / q_val);
        if (all_prev_accepted && uniform_rands[b * u.gamma + i] <= ratio) {
            accepted_mask[b * u.gamma + i] = 1.0;
        } else {
            accepted_mask[b * u.gamma + i] = 0.0;
            all_prev_accepted = false;
        }
    }
}
```

---

## 8. Production Deployment: Batching Interference, Draft Quality Drift & Hardware Limits

1. **Batching Saturation Limit**: Speculative decoding delivers maximum speedup at low concurrency ($B \le 8$). At high batch sizes ($B \ge 64$), the GPU becomes compute-saturated; adding draft overhead actually *reduces* aggregate throughput.
2. **Tokenizer Alignment**: Target and draft models must share an identical token vocabulary and BPE merges; vocabulary mismatch requires expensive logit mapping.
3. **KV Cache Rolling Back**: When tokens are rejected, the target model's KV cache pointers must instantly rewind to the rejection point $i$ to avoid memory pollution.
4. **Dynamic Horizon ($\gamma$) Tuning**: Adjust $\gamma$ dynamically according to real-time acceptance rates: increase $\gamma$ when $\beta > 0.85$, decrease $\gamma$ when generating hard code or complex math where $\beta < 0.5$.
