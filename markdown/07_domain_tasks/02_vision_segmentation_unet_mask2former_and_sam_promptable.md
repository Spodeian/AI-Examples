# Vision Segmentation: UNet, Mask2Former & Promptable SAM

## 1. Evolution of Segmentation: From Dense Pixels to Mask Queries to Foundation Models

Image segmentation partitions an image into semantically coherent pixel groups. Over the past decade, the field has progressed through four major paradigm shifts:

1. **Per-Pixel Dense Classification (FCN, UNet, DeepLab)**:
   - Early deep architectures framed semantic segmentation as dense pixel classification.
   - Fully convolutional networks (FCN) and encoder-decoder topologies with skip connections (**UNet**) preserved high-frequency spatial details while aggregating multi-scale receptive fields (dilated/atrous convolutions, ASPP).
   - Limitation: Bound to semantic segmentation only; unable to distinguish between individual instances of the same class.

2. **Top-Down Detect-Then-Segment (Mask R-CNN)**:
   - Extended two-stage detectors with a binary mask branch inside each Region of Interest (RoIAlign).
   - Excellent for instance segmentation, but fundamentally disjoint from semantic segmentation of stuff classes (sky, road, grass).

3. **Universal Query-Based Segmentation (Mask2Former)**:
   - Unifies semantic, instance, and panoptic segmentation under a single meta-architecture: **predicting a set of binary masks paired with class predictions**.
   - Employs **masked cross-attention**, restricting query attention to the foreground regions of the previous mask prediction.

4. **Promptable Foundation Segmentation (SAM — Segment Anything Model)**:
   - Eliminates fixed category vocabularies in favor of **prompt-driven zero-shot segmentation** guided by points, bounding boxes, or rough text prompts.
   - Encodes sparse geometric prompts into continuous token embeddings using **Random Gaussian Fourier features**.

---

## 2. UNet: Symmetric Encoder-Decoder with Skip Connections

Ronneberger et al. (2015) introduced UNet for biomedical image segmentation, which became the cornerstone of dense visual prediction:

```
Contracting Path (Encoder)               Expanding Path (Decoder)
   Input [B, C, H, W]  ────────────────────────> Concat [B, 2C, H, W] ──> Output [B, K, H, W]
         │                                            ▲
      Downsample                                   Upsample
         ▼                                            │
   Feature [B, 2C, H/2, W/2] ───────────────> Concat [B, 4C, H/2, W/2]
         │                                            ▲
      Downsample                                   Upsample
         ▼                                            │
      Bottleneck Latent Representation [B, 4C, H/4, W/4]
```

### 2.1 The Role of Skip Connections
In a deep convolutional contracting path, pooling and strided convolutions rapidly downsample feature maps, discarding high-frequency localization cues in exchange for large receptive fields and semantic abstraction.

UNet addresses this spatial information loss by introducing **long skip connections** between corresponding encoder and decoder levels:
$$x_{\text{dec}}^{(l)} = \operatorname{Conv}\left( \left[ \operatorname{Up}\left(x_{\text{dec}}^{(l+1)}\right), \ x_{\text{enc}}^{(l)} \right] \right),$$
where $[\cdot, \cdot]$ denotes channel concatenation. These skip connections allow high-resolution edge and boundary gradients to backpropagate directly without vanishing through deep bottleneck layers.

---

## 3. Soft Dice Loss & Compound Segmentation Objectives

In medical and fine-grained segmentation, the target region of interest often occupies a minute fraction ($< 1\%$) of the overall image pixels. Standard Cross-Entropy (CE) loss treats every pixel equally:
$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \left( y_i \log p_i + (1 - y_i) \log(1 - p_i) \right).$$
When the background dominates, minimizing $\mathcal{L}_{\text{CE}}$ causes the network to collapse to trivial all-zero background predictions.

### 3.1 Soft Dice Loss Formulation
The Sørensen–Dice coefficient measures the geometric overlap between prediction $P$ and ground truth $T$:
$$\text{Dice}(P, T) = \frac{2 |P \cap T|}{|P| + |T|} = \frac{2 \sum_i p_i t_i}{\sum_i p_i^2 + \sum_i t_i^2}.$$

The differentiable **Soft Dice Loss** with Laplace smoothing $\epsilon > 0$ is defined as:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i t_i + \epsilon}{\sum_i p_i^2 + \sum_i t_i^2 + \epsilon},$$
where $p_i \in [0, 1]$ are sigmoid probabilities and $t_i \in \{0, 1\}$ are binary ground-truth labels.

### 3.2 Compound Loss
In modern segmentation pipelines, models are trained on a linear combination of Focal/Cross-Entropy loss and Soft Dice loss:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{ce}} \mathcal{L}_{\text{CE}} + \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}}.$$
While Cross-Entropy provides well-conditioned, isotropic gradient surfaces across all pixels, the Dice loss directly maximizes spatial intersection-over-union, immune to extreme foreground-background class imbalance.

---

## 4. Mask2Former: Masked-Attention Mask Transformer

Cheng et al. (2022) developed **Mask2Former**, demonstrating that a single architecture can solve semantic, instance, and panoptic segmentation without modification.

Rather than predicting dense pixel grids, Mask2Former represents an image as $N$ query tokens $Q \in \mathbb{R}^{N \times D}$, each predicting:
1. A category distribution $p_i \in \Delta^K$.
2. A continuous embedding vector that takes an inner product with a high-resolution pixel decoder feature map to yield a binary mask $\hat{M}_i \in [0, 1]^{H \times W}$.

### 4.1 Masked Cross-Attention
Standard Transformer cross-attention allows query tokens to attend globally over all $S = H \times W$ spatial locations:
$$\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{D}} \right) V.$$
In early training, global cross-attention converges slowly because queries attend uniformly across the entire image.

Mask2Former introduces **Masked Cross-Attention**, restricting attention to the foreground region predicted by the query in the preceding decoder layer:
$$\mathcal{M}_{(l-1)}(x, y) = \begin{cases} 0 & \text{if } \sigma(\hat{M}_{(l-1)}(x, y)) \ge 0.5, \\ -\infty & \text{otherwise}. \end{cases}$$
The attention logits are modulated directly:
$$\operatorname{MaskedAttn}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{D}} + \mathcal{M}_{(l-1)} \right) V.$$
This localized attention focuses query updates exclusively on the object's foreground boundaries, accelerating convergence by over $3\times$ while reducing memory consumption.

---

## 5. Segment Anything Model (SAM): Foundation Promptable Segmentation

Kirillov et al. (2023) introduced **SAM**, training on over 1.1 billion masks (SA-1B dataset) to produce an unprecedented zero-shot promptable visual foundation model.

SAM decomposes segmentation into three decoupled components:
1. **Heavyweight Image Encoder**: A Vision Transformer (ViT-H/L/B) run once per image to compute image feature embeddings $F \in \mathbb{R}^{h \times w \times C}$.
2. **Flexible Prompt Encoder**: Projects sparse prompts (points, bounding boxes, text) and dense prompts (rough masks) into $256$-dimensional geometric query embeddings.
3. **Lightweight Two-Way Mask Decoder**: Interleaves query-to-image and image-to-query attention to decode multi-mask candidates in $< 50$ ms on consumer hardware.

---

## 6. Random Gaussian Fourier Features for Spatial Prompts

To condition neural networks on continuous 2D coordinates (such as user click points $p = (x, y) \in [0, 1]^2$), feeding raw $(x, y)$ coordinates fails due to the spectral bias of MLPs toward low-frequency functions.

SAM projects coordinate prompts using **Random Gaussian Fourier Features** (Tancik et al., 2020):
$$\gamma(p) = \left[ \sin(2\pi \cdot B p), \ \cos(2\pi \cdot B p) \right]^T \in \mathbb{R}^{2 D_{\text{feat}}},$$
where $B \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_{D_{\text{feat}} \times 2})$ is a fixed, non-trainable random Gaussian projection matrix.

### 6.1 Mathematical Properties
1. **Stationary Kernel Equivalence**: The inner product between two Fourier embeddings approximates a stationary Gaussian RBF kernel:
   $$\mathbb{E}_B \left[ \gamma(p_1)^T \gamma(p_2) \right] \approx \exp\left( -2\pi^2 \sigma^2 \|p_1 - p_2\|_2^2 \right).$$
2. **Sub-Pixel Spatial Sensitivity**: Enables the downstream Transformer attention heads to differentiate between clicks separated by fractions of a pixel.
3. **Continuous Interpolation**: Guarantees smooth representation of interactive brush strokes and point trajectory drag prompts.

---

## 7. Architectural Comparison & Mathematical Invariants

| Attribute | UNet | Mask R-CNN | Mask2Former | SAM |
| :--- | :--- | :--- | :--- | :--- |
| **Output Formulation** | Per-pixel logits | RoI bounding box + mask | Query set of masks | Prompt-conditioned mask set |
| **Task Scope** | Semantic only | Instance only | Semantic, Instance, Panoptic | Zero-shot promptable |
| **Attention Mechanism** | Convolutions | Spatial pooling (RoIAlign) | Masked cross-attention | Bi-directional self + cross |
| **Prompt Support** | None | Bounding box proposal | Class/mask query initialization | Points, boxes, rough masks |
| **Loss Strategy** | CE + Soft Dice | Smooth-$L_1$ + BCE mask | Hungarian Bipartite Match | Focal Loss + Dice Loss + IoU |

---

## 8. Cross-Track Verification & Systems-Level Contracts

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every segmentation operator conforms to the following mathematical specifications:

| Operator | Verification Metric | Tolerance | Numerical Invariant |
| :--- | :--- | :--- | :--- |
| **Soft Dice Loss** | Analytical $\frac{2 \sum p t + \epsilon}{\sum p^2 + \sum t^2 + \epsilon}$ match | $< 10^{-5}$ | Strictly $\in [0, 1]$ for $p, t \in [0, 1]$ |
| **Masked Cross-Attention** | Logits masked with background $-\infty / -10^4$ | $< 10^{-5}$ | Softmax weights sum to $1.0$ across unmasked positions |
| **SAM Fourier Prompt Embed** | Random Gaussian projection $[\sin(2\pi B p), \cos(2\pi B p)]$ | $< 10^{-5}$ | Embedding norm bounded: $\|\gamma(p)\|_\infty \le 1$ |
| **Linear Anchor** | Shared cross-track regression anchor | $< 10^{-5}$ | Deterministic linear projection against golden tensor |
| **Coordinate Bound** | Normalized prompt points $(x, y)$ | Exact | Clamped to unit interval $[0.0, 1.0]$ |


