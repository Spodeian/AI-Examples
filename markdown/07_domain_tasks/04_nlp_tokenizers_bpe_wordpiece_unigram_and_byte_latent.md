# NLP Tokenizers: BPE, WordPiece, Unigram & Byte-Latent Architectures

## 1. The Subword Paradigm: Solving the Out-of-Vocabulary Dilemma

Language models operate over discrete symbolic sequences. Historically, NLP relied on whitespace-delimited word vocabularies:
- **Out-of-Vocabulary (OOV) Catastrophe**: Any rare word, typo, or morphological variation (e.g., `unforgettably`) was mapped to an uninformative `[UNK]` token.
- **Unbounded Vocabulary Explosion**: In morphologically rich languages (German compound words, agglutinative Turkish), vocabulary size scaled exponentially.
- **Character-Level Inefficiency**: Pure character sequences increased sequence lengths $4\times\text{--}8\times$, incurring quadratic self-attention costs $\mathcal{O}(T^2)$.

**Subword tokenization** resolved this trade-off by dynamically decomposing rare words into shared frequent morphemes and character n-grams while preserving common words as atomic tokens.

---

## 2. Byte-Pair Encoding (BPE)

Originally an iterative data compression algorithm (Gage, 1994), Sennrich et al. (2016) adapted **BPE** for neural machine translation.

### 2.1 Training Procedure
1. Initialize vocabulary $V_0$ with all unique atomic characters (or all $256$ raw UTF-8 byte values for Byte-Level BPE).
2. Segment every training word into sequences of single symbols plus an end-of-word delimiter (e.g., `['l', 'o', 'w', '</w>']`).
3. Count frequencies of all adjacent symbol pairs $(u, v)$ across the corpus.
4. Identify the most frequent pair $(u^\star, v^\star) = \arg\max_{(u, v)} \text{freq}(u, v)$.
5. Merge all occurrences of $(u^\star, v^\star)$ into a new atomic token $uv$, adding it to $V$.
6. Record the tuple $(u^\star, v^\star)$ and its merge priority rank.
7. Repeat steps 3–6 until the target vocabulary size $|V|$ is achieved.

### 2.2 Inference / Tokenization
At inference time, greedy BPE scans the word and iteratively merges pairs in the exact chronological order of their training merge ranks until no further pairs exist in the merge table.

---

## 3. WordPiece: Likelihood-Maximizing Bigram Selection

Schuster and Nakajima (2012) and Wu et al. (2016) developed **WordPiece** for Google's Neural Machine Translation and BERT.

Unlike BPE, which greedily merges the highest-frequency pair, WordPiece merges the symbol pair that **maximizes the likelihood of the training corpus under a unigram language model**:
$$\text{Score}(u, v) = \frac{\text{count}(uv)}{\text{count}(u) \times \text{count}(v)}.$$

### 3.1 Pointwise Mutual Information (PMI)
Taking the logarithm reveals that WordPiece optimizes the Pointwise Mutual Information between adjacent units:
$$\log \text{Score}(u, v) = \log \frac{P(uv)}{P(u) P(v)} = \operatorname{PMI}(u, v).$$
This formulation prevents high-frequency individual characters (like `'e'` or `'s'`) from dominating merge decisions unless they appear together with unusually high mutual association.

---

## 4. Unigram Language Model & SentencePiece

Kudo (2018) introduced the **Unigram Language Model** tokenization algorithm, implemented in **SentencePiece**.

While BPE and WordPiece are bottom-up (starting with single characters and merging), Unigram is **top-down**:
1. Initialize a massive vocabulary $V$ with millions of candidate substrings and words.
2. Estimate the unigram probability distribution $P(x)$ for all $x \in V$ using the Expectation-Maximization (EM) algorithm.
3. Compute the marginal loss (relative entropy) caused by removing each token from $V$.
4. Prune the bottom $20\%$ of tokens with the smallest loss increase.
5. Repeat steps 2–4 until the vocabulary shrinks to the target size.

### 4.1 Multiple Segmentations & Subword Regularization
Given a word $W$, there may exist many valid segmentations $\mathbf{x} = (x_1, \dots, x_m)$. The optimal segmentation is found in linear time via the **Viterbi algorithm**:
$$\mathbf{x}^\star = \arg\max_{\mathbf{x}} \sum_{i=1}^m \log P(x_i).$$
Crucially, Unigram allows sampling non-optimal segmentations proportionally to $P(\mathbf{x}) = \prod P(x_i)$ (**Subword Regularization**), boosting model robustness against spelling errors and morphological variance.

---

## 5. Byte-Level & Byte-Latent Models: The Tokenizer-Free Frontier

Despite their success, subword tokenizers introduce fundamental pathologies:
- **Language Bias & Fairness**: Tokenizers trained on English require up to $5\times$ more tokens per sentence in low-resource scripts (Devanagari, Cyrillic, Arabic), multiplying compute and inference cost.
- **Spelling & Character Brittleness**: Subword models fail at basic character tasks (e.g., counting the letter `'r'` in `"strawberry"`).
- **Vulnerability to Adversarial Attacks**: Minor perturbations or unseen emojis degrade token sequences unpredictably.

### 5.1 Byte Latent Transformers (BLT) & MegaByte
Modern architectures process raw UTF-8 bytes directly ($V = 256$):
1. **Local Byte Patching**: Bytes are grouped into local patches (statically as in MegaByte, or dynamically based on entropy as in BLT).
2. **Heavyweight Global Transformer**: Processes compact latent patch representations, maintaining the FLOP efficiency of subword models.
3. **Lightweight Local Cross-Attention**: Decodes patch representations back into raw bytes.

---

## 6. Dynamic Patch Boundary Detection via Local Byte Entropy

In **Byte Latent Transformers (BLT)** (Patel et al., 2024), fixed patch sizes are replaced with **dynamic byte boundaries** guided by predictive entropy.

Let $P(b_t \mid b_{<t})$ be the probability of the next byte predicted by a tiny 1-layer n-gram or byte-level language model. The Shannon entropy at position $t$ is:
$$\mathcal{H}(t) = -\sum_{v=0}^{255} P(v \mid b_{<t}) \log_2 P(v \mid b_{<t}).$$

### 6.1 Entropy Boundary Dynamics
- **Low Entropy (Predictable Continuations)**: Inside a word stem (e.g., after `"catastroph"`), the next byte `'i'` has high probability $\implies \mathcal{H}(t)$ is low.
- **High Entropy (Information Spikes)**: At word boundaries, whitespace, or punctuation (e.g., after `"the "`), the identity of the next word's initial character is highly uncertain $\implies \mathcal{H}(t)$ spikes.
- **Patch Demarcation**: Placing patch boundaries whenever $\mathcal{H}(t)$ crosses a threshold or exceeds a local running average naturally aligns latent patches with morphemic and semantic boundaries.

---

## 7. Comparative Taxonomy & Operational Invariants

| Attribute | BPE | WordPiece | Unigram (SentencePiece) | Byte-Latent (BLT / MegaByte) |
| :--- | :--- | :--- | :--- | :--- |
| **Direction** | Bottom-up (Merge) | Bottom-up (Merge) | Top-down (Prune) | **Tokenizer-Free** |
| **Merge Criterion** | Pair frequency | $\frac{\text{count}(uv)}{\text{count}(u)\text{count}(v)}$ | Minimizes corpus loss | Entropy spikes $\mathcal{H}(t)$ |
| **Vocab Size** | $32\text{k}\text{--}128\text{k}$ | $30\text{k}$ | $16\text{k}\text{--}64\text{k}$ | **$256$ (Raw Bytes)** |
| **Lossy Pre-tok** | Often yes (regex) | Yes (BERT pre-tok) | Zero (Lossless) | **Zero (Raw UTF-8)** |
| **Multilingual Parity**| High English skew | Moderate skew | Moderate skew | **Universal Script Equity** |

---

## 8. Cross-Track Verification & Systems-Level Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, all tokenization and byte-level operations obey the following contracts:

| Operator | Verification Metric | Tolerance | Numerical Invariant |
| :--- | :--- | :--- | :--- |
| **WordPiece Score** | $C_{ij} / (C_i \cdot C_j + \epsilon)$ matrix calculation | $< 10^{-5}$ | Non-negative: $S_{ij} \ge 0$ |
| **Byte Patch Pooling** | Exact average pooling across patch dimension | $< 10^{-5}$ | Bounded by input norm: $\|P\|_\infty \le \|B\|_\infty$ |
| **Entropy Boundary Detector** | $-\sum p \log_2(p + \epsilon)$ Shannon entropy | $< 10^{-5}$ | Strictly non-negative: $\mathcal{H} \ge 0$ |
| **Linear Anchor** | Shared cross-track regression anchor | $< 10^{-5}$ | Deterministic match against golden tensor |
| **Entropy Bound** | Uniform distribution over $V$ tokens | Exact | $\mathcal{H}_{\max} = \log_2(V)$ |


