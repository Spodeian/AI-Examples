# Supervised Fine-Tuning: Chat Masks, NEFTune, and Sequence Packing

## 1. Mathematical Foundations & Lineage

Supervised Fine-Tuning (SFT) is the foundational first stage of post-training, transforming pre-trained next-token predictors into coherent, instruction-following conversational agents and reasoning engines.

The modern SFT lineage comprises:
1. **InstructGPT (Ouyang et al. 2022)**: Formalized the three-stage post-training pipeline (SFT $\to$ Reward Modeling $\to$ PPO/RLHF), demonstrating that a 1.3B parameter SFT model outperforms a 175B base model on human preference evaluations.
2. **Standardized Chat Formats & ChatML (Meta Llama-2/3, OpenAI)**: Introduced structured special delimiter tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`), requiring exact token loss masking to prevent prompt memorization.
3. **NEFTune (Jain et al. 2023)**: Discovered that adding scaled uniform noise directly to input embeddings during instruction tuning dramatically reduces overfitting to surface phrasing and formatting tokens, boosting AlpacaEval scores by up to $3\text{--}5\%$.
4. **Sequence Packing (Multipack / FlashAttention VarLen)**: Replaced wasteful zero-padding with variable-length sequence concatenation, increasing GPU hardware compute efficiency by $2\times\text{--}4\times$ via block-diagonal causal attention masking.

---

## 2. Chat Template Loss Masking & NEFTune Noise Injection

### Completion-Only Loss Masking
In multi-turn conversational fine-tuning, training objectives must only penalize errors on assistant-generated tokens. Evaluating loss on user prompt tokens incentivizes the model to predict user prompts rather than user-conditioned completions.

Given conversational tokens $y_{1:L}$ and binary completion mask $m_{1:L} \in \{0, 1\}$:
$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{\sum_{t=1}^L m_t \ln P_\theta(y_t \mid y_{<t})}{\sum_{t=1}^L m_t + \epsilon}$$
where $m_t = 1$ if token $t$ belongs to an assistant turn, and $m_t = 0$ for system instructions, user inputs, and template headers.

### NEFTune Noise Injection
Standard SFT tends to overfit rapidly to repetitive chat template delimiters. NEFTune injects uniform noise into the embedding layer:
$$\mathbf{e}_{\text{noisy}} = \mathbf{e} + \frac{\alpha}{\sqrt{L \cdot D}} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \text{Uniform}(-1, 1)^{B \times L \times D}$$
where $L$ is sequence length, $D$ is embedding dimension, and $\alpha \in [5.0, 15.0]$ is the noise intensity hyperparameter. The $\frac{1}{\sqrt{L \cdot D}}$ scaling preserves the relative magnitude of the embedding norm across variable sequence lengths.

---

## 3. Sequence Packing, Block-Diagonal Masks & Position IDs

### Zero-Waste Sequence Packing
In batched LLM training, padding short sequences to the maximum context length $L_{\max}$ causes over $50\%$ of tensor cells to be wasted on `<pad>` tokens. Sequence packing concatenates multiple independent sequences into a single continuous buffer of size $L_{\max}$.

### Block-Diagonal Causal Attention Mask
To prevent tokens in sequence $D_{k+1}$ from attending to previous sequence $D_k$, the self-attention mechanism must enforce a 2D block-diagonal causal mask:
$$M_{i, j} = \begin{cases} 1.0 & \text{if } j \le i \text{ and } \text{seg}[i] = \text{seg}[j] \\ 0.0 & \text{otherwise} \end{cases}$$
where $\text{seg}[i]$ designates the document index of token $i$.

### Position ID Resetting
Rotary Position Embeddings (RoPE) require token positions to reflect intra-document distances rather than concatenated buffer positions. Position IDs must reset to 0 at the start of each packed sub-sequence:
$$\text{pos}[i] = i - \text{start}(\text{seg}[i])$$

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`neftune_add_noise`** | `embeddings + (alpha / sqrt(L*D)) * noise` | `embeddings + scale * noise` | Tensor scaling and elementwise addition | Parallel float noise addition |
| **`masked_completion_cross_entropy`** | `gather` + masked NLL reduction | `log_softmax` indexing + masked sum | Stable log-sum-exp + masked reduction | Fused cross-entropy kernel |
| **`packed_block_diagonal_causal_mask`** | `(causal & same_seg).float()` | `(causal & same_seg).astype(float32)` | 2D lower-triangular segment match | `PACKED_BLOCK_DIAGONAL_CAUSAL_MASK_WGSL` |
| **`packed_position_ids`** | Loop with segment transition reset | Loop with segment transition reset | Integer vector boundary reset | Parallel prefix scan |

---

## 5. Pathological Failure Modes & Diagnostic Postmortems

1. **Prompt Memorization & Autoregressive Babbling**:
   - *Symptom*: Fine-tuned model hallucinates user prompts and talks to itself in multi-turn dialogues instead of waiting for user input.
   - *Cause*: Unmasked training (loss evaluated across all tokens), which forces the model to learn conditional distributions over user prompts.
   - *Fix*: Apply strict **completion-only loss masking** ($m_t = 0$ on all user/system spans).
2. **Cross-Document Context Contamination in Sequence Packing**:
   - *Symptom*: Model leaks entities and facts from unrelated previous documents in the packed batch into downstream responses.
   - *Cause*: Standard lower-triangular causal mask used on packed sequence buffers without segment boundaries.
   - *Fix*: Enforce **2D block-diagonal causal masking** or use FlashAttention VarLen with `cu_seqlens`.
3. **NEFTune Destabilization**:
   - *Symptom*: Severe training loss explosion and degradation of grammatical coherence.
   - *Cause*: Over-scaled $\alpha > 25$ or missing $\frac{1}{\sqrt{L \cdot D}}$ normalization factor.
   - *Fix*: Constrain $\alpha \in [5.0, 10.0]$ and verify dimension scaling.

---

## 6. Cross-Domain Generalisation Matrix

1. **Foundation Model Chat Instruction Tuning (Llama 3 / Mistral)**: Completion masking and NEFTune prevent template collapse, boosting instruction adherence across human evaluators.
2. **Code Generation & Test Synthesis (DeepSeek-Coder)**: Sequence packing accelerates code training throughput by $3.4\times$ by eliminating padding across thousands of varying-length function definitions.
3. **Medical & Legal Document Reasoning**: Position ID resets prevent long-context contamination between unrelated patient records or legal briefs packed into 32k context windows.

---

## 7. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `neftune_noisy_ref`: Scaled uniform noise injection into token representations.
3. `masked_loss_ref`: Masked cross-entropy loss evaluated exclusively on completion tokens.
4. `packed_mask_ref`: 2D block-diagonal causal attention mask.
5. `packed_positions_ref`: Intra-sequence position IDs resetting at document boundaries.

Maximum absolute error across all tracks satisfies $\| \mathbf{y}_{\text{ref}} - \mathbf{y}_{\text{track}} \|_\infty < 10^{-5}$.

---

## 8. Synchronized Navigation Links

- **PyTorch Reference**: [`pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.py`](../../../pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.py)
- **JAX Reference**: [`jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.py`](../../../jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.py)
- **Candle Reference**: [`candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.rs`](../../../candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.rs)
- **Kernels**: [`kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.rs`](../../../kernels/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.rs)
- **Interactive Notebook**: [`notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.ipynb`](../../../notebooks/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.ipynb)
- **Topic Bridges**: [`INDEX_BY_GOAL.md`](../../../INDEX_BY_GOAL.md)
