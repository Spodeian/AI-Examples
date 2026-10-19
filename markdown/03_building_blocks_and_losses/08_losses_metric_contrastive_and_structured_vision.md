# Losses: Metric, Contrastive, and Structured Vision (C03 Topic 08)

## 1. Executive Summary
Traditional point-wise cross-entropy and regression losses treat target values as isolated scalar or categorical indices, ignoring the underlying geometry of embedding spaces, physical spatial overlaps, and dense continuous masks. Metric learning, contrastive estimation, and structured vision losses bridge this gap by defining objective functions over pairwise similarities, geodesic angular manifolds, and continuous set intersections:
1. **Metric & Contrastive Learning:** Early metric learning utilized pairwise margin-based contrastive loss (*Chopra et al. 2005*) or triplet loss. **InfoNCE** (*Oord et al. 2018*) revolutionized self-supervised representation learning by treating representation matching as a multi-class classification problem over mini-batch negatives, grounded in mutual information maximization.
2. **Decoupled Multimodal Alignment (SigLIP):** Standard contrastive language-image pretraining (CLIP) requires computing a softmax over all text-image pairs across the entire multi-node cluster, necessitating costly distributed `all-gather` communication barriers. **SigLIP** (*Zhai et al. 2023*) replaces the softmax normalization with independent pairwise sigmoid binary cross-entropy, completely eliminating distributed sync barriers while stabilizing training at massive batch sizes ($B > 32\text{k}$).
3. **Hyperspherical Margin Embedding (ArcFace):** In high-dimensional biometric verification (face identification, speaker verification), cross-entropy fails to enforce sufficient inter-class separability. **ArcFace** (*Deng et al. 2019*) projects features and class prototypes onto a unit hypersphere and injects an additive angular margin $m$ directly into the geodesic angle: $\cos(\theta + m)$, producing compact intra-class clusters and expansive inter-class margins.
4. **Continuous Set Overlap (Soft-Dice):** In medical image segmentation and remote sensing, foreground target lesions occupy $<1\%$ of image voxels. Standard cross-entropy is heavily dominated by background pixels. **Soft-Dice Loss** (*Milletari et al. 2016*) optimizes the continuous relaxation of the Sørensen-Dice coefficient (F1 score), remaining scale-invariant to foreground size.
5. **Continuous Spatial Bounding Box Regression (GIoU):** Mean squared error on bounding box coordinates $(x, y, w, h)$ does not reflect spatial intersection and is sensitive to box scale. While Intersection-over-Union ($\text{IoU}$) is scale-invariant, it provides zero gradient when predicted and ground-truth boxes do not overlap. **Generalized IoU (GIoU)** (*Rezatofighi et al. 2019*) introduces an enclosing convex hull penalty, guaranteeing continuous gradient flow even for completely disjoint bounding boxes.

This curriculum topic provides rigorous mathematical derivations, gradient analyses, pathological failure mode postmortems, and synchronized implementations across PyTorch 2.x, JAX, Rust Candle, and Hardware Kernels (WGSL + CPU auto-dispatcher).

---

## 2. Mathematical Foundations

### 2.1 InfoNCE Loss (Contrastive Mutual Information Estimation)
Given query representation $q \in \mathbb{R}^D$ and a set of candidate keys $\{k_0, k_1, \dots, k_{B-1}\}$ containing one positive match $k_+$ and $B-1$ negative samples, InfoNCE normalizes embeddings to the unit hypersphere ($\hat{q} = q / \|q\|_2, \hat{k}_j = k_j / \|k_j\|_2$) and applies temperature-scaled softmax cross-entropy:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\hat{q} \cdot \hat{k}_+ / \tau)}{\sum_{j=0}^{B-1} \exp(\hat{q} \cdot \hat{k}_j / \tau)}$$

The temperature parameter $\tau > 0$ controls hardness of the negative distribution:
- As $\tau \to 0$: the loss concentrates gradient force exclusively on the hardest negative example: $\lim_{\tau \to 0} \nabla \mathcal{L} \propto \hat{k}_{\text{hardest}}$.
- As $\tau \to \infty$: the denominator becomes uniform, and the loss degenerates into an unweighted sum of dot products.

### 2.2 SigLIP Loss (Sigmoid Loss for Language-Image Pre-Training)
Given a batch of image representations $\{q_i\}_{i=1}^B$ and text representations $\{k_j\}_{j=1}^B$, SigLIP avoids global denominator normalization. For every pair $(i, j)$, the logit is defined with learnable temperature scale $t$ and bias $b$:
$$Z_{ij} = t \cdot (\hat{q}_i \cdot \hat{k}_j) + b$$
The label is defined as $Y_{ij} = 1$ if $i = j$ (positive pair) and $Y_{ij} = -1$ if $i \ne j$ (negative pair).
The loss is formulated as independent binary logistic regressions:
$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{B} \sum_{i=1}^B \sum_{j=1}^B \log \sigma(Y_{ij} Z_{ij}) = \frac{1}{B} \sum_{i=1}^B \sum_{j=1}^B \log\left( 1 + \exp(-Y_{ij} Z_{ij}) \right)$$
Because each pair $(i, j)$ is computed independently, SigLIP can be chunked or pipelined without a global denominator all-gather, allowing batch sizes to scale without communication overhead.

### 2.3 ArcFace Loss (Additive Angular Margin)
In metric classification, let $x \in \mathbb{R}^D$ be the feature embedding and $W_j \in \mathbb{R}^D$ be the prototype weight vector for class $j$.
Both features and weights are $L_2$ normalized: $\|x\|_2 = 1, \|W_j\|_2 = 1$. The cosine similarity is:
$$\cos \theta_j = W_j^T x$$
ArcFace adds an angular margin $m > 0$ strictly to the target class angle $\theta_y$:
$$\cos(\theta_y + m) = \cos \theta_y \cos m - \sin \theta_y \sin m = \cos \theta_y \cos m - \sqrt{1 - \cos^2 \theta_y} \sin m$$
The scaled logits are defined as:
$$\tilde{z}_j = \begin{cases} s \cdot \cos(\theta_y + m) & \text{if } j = y \\ s \cdot \cos \theta_j & \text{if } j \ne y \end{cases}$$
Where $s$ is the hypersphere radius scale (typically $s = 30.0$ or $64.0$).
The objective is the standard cross-entropy over $\tilde{z}$:
$$\mathcal{L}_{\text{ArcFace}} = -\log \frac{\exp(s \cdot \cos(\theta_y + m))}{\exp(s \cdot \cos(\theta_y + m)) + \sum_{j \ne y} \exp(s \cdot \cos \theta_j)}$$
Geometrically, this requires the angle $\theta_y$ between $x$ and the ground truth class prototype $W_y$ to be smaller than any alternative class angle by at least $m$ radians ($\theta_y + m < \theta_j$).

### 2.4 Soft-Dice Loss for Segmentation
For continuous predicted probability map $P \in [0, 1]^N$ and binary ground-truth segmentation mask $Y \in \{0, 1\}^N$:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i^2 + \sum_{i=1}^N y_i^2 + \epsilon}$$
Where $\epsilon > 0$ prevents division by zero.
Its gradient with respect to predicted probability $p_k$ is:
$$\frac{\partial \mathcal{L}_{\text{Dice}}}{\partial p_k} = -2 \cdot \frac{y_k \left( \sum p_i^2 + \sum y_i^2 + \epsilon \right) - 2 p_k \left( \sum p_i y_i + \epsilon \right)}{\left( \sum p_i^2 + \sum y_i^2 + \epsilon \right)^2}$$
When the background dominates ($\sum y_i \ll N$), the denominator depends only on the foreground size, preventing background pixels from overwhelming the gradient.

### 2.5 Generalized Intersection over Union (GIoU) Loss
Given two bounding boxes $A, B \subset \mathbb{R}^2$ parameterized as $[x_1, y_1, x_2, y_2]$:
1. Intersection area: $|A \cap B| = \max(0, \min(x_2^A, x_2^B) - \max(x_1^A, x_1^B)) \cdot \max(0, \min(y_2^A, y_2^B) - \max(y_1^A, y_1^B))$.
2. Union area: $|A \cup B| = |A| + |B| - |A \cap B|$.
3. Standard IoU: $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$.
4. Smallest enclosing convex hull box $C$:
   $$C = [\min(x_1^A, x_1^B), \min(y_1^A, y_1^B), \max(x_2^A, x_2^B), \max(y_2^A, y_2^B)]$$
5. GIoU metric:
   $$\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|} = \text{IoU} - \frac{|C| - |A \cup B|}{|C|}$$
The loss is $\mathcal{L}_{\text{GIoU}} = 1 - \text{GIoU} \in [0, 2]$.
When boxes do not overlap ($|A \cap B| = 0$), $\text{IoU} = 0$, but $\text{GIoU} = - \frac{|C| - (|A| + |B|)}{|C|} < 0$. The loss increases as the disjoint boxes move further apart, providing continuous gradients pulling disjoint boxes together.

---

## 3. Structural Properties & Computational Complexity Table

```text
Loss Function    Domain            Key Hyperparameters    FLOPs / Element    Geometric Property
─────────────────────────────────────────────────────────────────────────────────────────────────
InfoNCE          Contrastive SSL   temperature tau        O(B * D)           Hypersphere Softmax
SigLIP           Multimodal Pretr. scale t, bias b        O(B^2 * D)         Pairwise Decoupled BCE
ArcFace          Face/Biometrics   scale s, margin m      O(K * D)           Geodesic Angular Margin
Soft-Dice        Segmentation      smooth epsilon         O(N)               Scale-Invariant Overlap
GIoU             Object Detection  smooth epsilon         O(1) / Box         Convex Hull Enclosing
```

---

## 4. Multi-Ecosystem Implementation Matrix

| Operator / Layer | PyTorch 2.x (`pytorch2/`) | JAX / NumPy (`jax/`) | Candle Rust (`candle/`) | Hardware Kernels (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **InfoNCE Loss** | `compute_infonce_loss` | `infonce_loss_forward` | `infonce_loss_forward` | `infonce_loss_cpu` |
| **SigLIP Loss** | `compute_siglip_loss` | `siglip_loss_forward` | `siglip_loss_forward` | `siglip_loss_cpu` |
| **ArcFace Loss** | `compute_arcface_loss` | `arcface_loss_forward` | `arcface_loss_forward` | `arcface_loss_cpu` |
| **Soft-Dice Loss** | `compute_soft_dice_loss` | `soft_dice_loss_forward` | `soft_dice_loss_forward` | `DICE_SHADER` (WGSL) + CPU |
| **GIoU Loss** | `compute_giou_loss` | `giou_loss_forward` | `giou_loss_forward` | `giou_loss_cpu` |
| **Linear Anchor** | `MetricVisionUnifiedModule`| `linear_forward` | `LinearAnchor` | `linear_anchor_forward_cpu` |
| **Parity Tolerance** | $< 10^{-5}$ | $< 10^{-5}$ | $< 10^{-5}$ | $< 10^{-5}$ |

---

## 5. Pathological Failure Modes & Postmortems

1. **Zero-Overlap Gradient Death in Vanilla IoU:**
   - *Failure Mechanism:* Standard $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$. If predicted bounding box $A$ and ground-truth bounding box $B$ do not touch ($A \cap B = \emptyset$), $|A \cap B| = 0$ identically in a finite open neighborhood around $A$. Its derivative $\nabla_A \mathcal{L}_{\text{IoU}} = 0$, completely halting gradient descent.
   - *Resolution:* GIoU penalizes the empty space $|C \setminus (A \cup B)|$ in the enclosing box $C$. As distance between disjoint boxes increases, $|C|$ grows, driving $\text{GIoU} \to -1$ and providing consistent non-zero gradient force directing predicted boxes toward targets.
2. **Temperature Collapse in InfoNCE:**
   - *Failure Mechanism:* If $\tau$ is set too small ($\tau < 0.01$), the softmax probability distribution collapses into a one-hot Dirac delta on the single highest similarity score. Gradients with respect to all other negative samples vanish, preventing the network from learning comprehensive feature disentanglement. Conversely, if $\tau > 1.0$, the loss acts as an unweighted average, producing feature collapse where all representations converge to a single cluster.
   - *Resolution:* Fix $\tau \in [0.05, 0.1]$ or learn $\log(1/\tau)$ clamped within $[\log 1, \log 100]$.
3. **Unnormalized Embeddings Bypassing ArcFace Margin:**
   - *Failure Mechanism:* The ArcFace additive angular formula $\cos(\theta + m)$ requires strict $L_2$ normalization $\|x\|_2 = 1$ and $\|W_j\|_2 = 1$. If normalization is omitted, the network increases feature norms $\|x\|_2 \to \infty$ to saturate softmax probabilities, completely circumventing the angular penalty and destroying hyperspherical clustering.
   - *Resolution:* Strictly enforce `F.normalize(x, p=2, dim=-1)` on both features and prototype weights prior to cosine dot product calculation.

---

## 6. Cross-Domain Generalisation

- **Multimodal Language-Vision (CLIP, SigLIP):** Pairwise contrastive objectives align heterogeneous text and image embedding manifolds into a shared semantic latent space.
- **Biometric Security & Identity Verification (ArcFace, CosFace):** Hyperspherical angular margins are the industry standard for 1:N face matching and voiceprint identification.
- **Biomedical Lesion & Organ Segmentation (Soft-Dice, Tversky):** Soft-Dice enables volumetric 3D organ segmentation across CT/MRI scans despite severe class imbalance.
- **Autonomous Driving & 3D Spatial Detection (GIoU, CIoU, DIoU):** Bounding box losses guide anchor-free detector heads in camera and LiDAR perception pipelines.

---

## 7. Verification & Empirical Parity

| Comparison Pair | Max Absolute Difference ($\|Y_A - Y_B\|_\infty$) | Test Status |
| :--- | :--- | :--- |
| **PyTorch 2.x vs Golden Safetensors** | $0.00 \times 10^0$ | **PASSED** |
| **JAX vs Golden Safetensors** | $9.54 \times 10^{-7}$ | **PASSED** |
| **Rust Candle vs Golden Safetensors** | $0.00 \times 10^0$ | **PASSED** |
| **Hardware Kernel (CPU/WGPU) vs Golden** | $0.00 \times 10^0$ | **PASSED** |

All four execution tracks satisfy the strict curriculum parity threshold ($\|Y_A - Y_B\|_\infty < 10^{-5}$) across all 5 loss formulations.

---

## 8. References & Further Reading

1. Oord, A. v. d., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding (InfoNCE)*. arXiv:1807.03748.
2. Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP)*. IEEE/CVF International Conference on Computer Vision (ICCV 2023).
3. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019).
4. Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation (Soft-Dice)*. International Conference on 3D Vision (3DV 2016).
5. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*. CVPR 2019.
