# Contrastive SSL: SimCLR, MoCo, CLIP, and SigLIP

## 1. The Self-Supervised Paradigm: Representation Learning Without Labels

Supervised deep learning is strictly bottlenecked by the availability of human annotations. Self-Supervised Learning (SSL) circumvents human labeling by defining pretext proxy tasks derived directly from raw, uncurated data. The contrastive learning paradigm operationalizes the principle of mutual information maximization: representations of different augmented views of the same instance (positive pairs) must be pulled together on the unit hypersphere, while representations of distinct instances (negative pairs) must be repelled.

## 2. InfoNCE and SimCLR: Normalized Temperature-Scaled Cross Entropy

SimCLR (Chen et al.) established that simple contrastive architectures can match or exceed supervised pre-training when paired with strong data augmentations (random cropping, color jitter, Gaussian blur) and a non-linear projection head $g(h) = W_2 \sigma(W_1 h)$:

- **Representation Normalization:** Latent projections are mapped to the unit hypersphere: $\hat{z}_i = z_i / \|z_i\|_2$.
- **NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss:** For positive pair $(i, j)$ in a batch of $2N$ augmented views:
  $$\mathcal{L}_{\text{NT-Xent}}(i, j) = -\log \frac{\exp(\langle \hat{z}_i, \hat{z}_j \rangle / \tau)}{\sum_{k=1}^{2N} \mathbb{I}(k \ne i) \exp(\langle \hat{z}_i, \hat{z}_k \rangle / \tau)}$$
- Temperature hyperparameter $\tau$ controls the penalty hardness on difficult negative samples: smaller $\tau$ focuses gradients disproportionately on the nearest interfering negative distractors.

## 3. Momentum Contrast (MoCo): Decoupling Queue Size from Batch Size

SimCLR requires gigantic batch sizes ($B = 4096$) across hundreds of GPUs so that each query encounters sufficient negative samples. MoCo (He et al.) decouples the negative dictionary size from GPU mini-batch capacity by framing contrastive learning as dynamic dictionary lookup:

- **Negative Memory Queue:** Maintains a FIFO queue of $M$ negative key embeddings ($M \approx 65,536$).
- **Momentum Key Encoder:** Rather than updating the key encoder with SGD gradients (which rapidly invalidates older queue entries, causing catastrophic representation drift), key encoder weights $\theta_k$ evolve slowly via Exponential Moving Average (EMA):
  $$\theta_k \leftarrow m \theta_k + (1 - m) \theta_q, \quad m = 0.999$$
- Smooth momentum updates ensure that representations in the queue remain mutually comparable and consistent.

## 4. Contrastive Language-Image Pre-Training (CLIP)

CLIP (Radford et al.) scales contrastive learning to multimodal image-text web pairs ($400\text{M}$ pairs):
- Two independent encoders map images $I \in \mathbb{R}^{B \times D}$ and text captions $T \in \mathbb{R}^{B \times D}$ into a shared semantic latent space.
- Dual-encoder cosine similarity matrix with learned temperature $\tau$ and bias:
  $$L_{ij} = \left( \frac{I_i}{\|I_i\|} \cdot \frac{T_j}{\|T_j\|} \right) \cdot e^{\tau} + b$$
- **Symmetric Cross-Entropy Objective:**
  $$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} \left[ \text{CrossEntropy}(L, \mathbf{y}) + \text{CrossEntropy}(L^T, \mathbf{y}) \right]$$
  where $\mathbf{y} = (0, 1, \dots, B-1)$ assigns diagonal entries as matching pairs.

## 5. SigLIP: Replacing All-Gather Softmax with Pairwise Sigmoid Losses

Standard CLIP computes softmax normalizations across the entire batch dimension. In multi-node distributed training across thousands of GPUs, evaluating the softmax denominator requires an expensive `all_gather` collective communication step across all ranks, stalling accelerator pipelines.

SigLIP (Zhai et al.) observes that softmax normalization is mathematically unnecessary for contrastive learning. Instead, it formulates contrastive alignment as independent pairwise binary classifications:

- Define binary ground truth $Y_{ij} = +1$ if $i = j$ (positive pair) and $-1$ if $i \ne j$ (negative pair).
- Pairwise Sigmoid Loss:
  $$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{B} \sum_{i=1}^B \sum_{j=1}^B \log \sigma\left( Y_{ij} L_{ij} \right) = \frac{1}{B} \sum_{i, j} \log\left( 1 + \exp(-Y_{ij} L_{ij}) \right)$$
- **Benefits:** Because every pair $(i, j)$ computes its loss independently without a shared global denominator, distributed ranks can evaluate arbitrary chunked matrix tiles without communicating across nodes, enabling unprecedented batch sizes ($B > 1\text{M}$).

## 6. Negative Sampling Dynamics & The Hard Negative Dilemma

- **False Negatives:** When batch sizes scale to millions, semantically identical or near-duplicate instances (e.g., two distinct images of golden retrievers) are treated as negative pairs, penalizing the model for learning semantic invariance.
- **Hard Negative Mining:** Gradients scale inversely with cosine distance $\exp(\langle q, k^- \rangle / \tau)$. While hard negatives drive rapid feature separation, excessively aggressive negative mining causes topological distortion and cluster fragmentation.

## 7. Hyperspherical Geometry: Alignment and Uniformity

Wang and Isola showed that contrastive representation quality decomposes into two fundamental geometric properties on the unit hypersphere $\mathcal{S}^{D-1}$:
1. **Alignment:** Positive pairs map to nearby points:
   $$\mathcal{L}_{\text{align}} = \mathbb{E}_{(x, x^+) \sim p_{\text{pos}}} \left[ \| f(x) - f(x^+) \|_2^\alpha \right]$$
2. **Uniformity:** The induced feature distribution should preserve maximal information, distributing uniformly over the sphere to maximize entropy:
   $$\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{x, y \sim p_{\text{data}}} \left[ \exp\left( -t \| f(x) - f(y) \|_2^2 \right) \right]$$
InfoNCE asymptotically balances alignment (numerator) and uniformity (denominator).

## 8. Cross-Track Verification and Parity

All contrastive algorithms are verified across the 7-track curriculum against seed-42 golden safetensors:
- **PyTorch 2.x:** Reference implementations of SimCLR NT-Xent loss, MoCo memory queue contrastive cross-entropy, symmetric dual-encoder CLIP loss, and SigLIP pairwise sigmoid loss.
- **JAX:** Functional implementation utilizing `jax.nn.log_softmax`, `jnp.diag`, and `jax.nn.softplus`, verified to $< 10^{-5}$ error.
- **Candle:** Native Rust implementation with batched matrix multiplications, diagonal log-probability reductions, and pairwise binary softplus evaluation, passing `cargo test -p ai-candle --test 08_learning_paradigms_rl_and_alignment test_01`.
- **WGSL Kernel:** Fused SigLIP pairwise compute shader evaluating cosine similarities, affine scalings, and softplus loss in a single GPU pass.
