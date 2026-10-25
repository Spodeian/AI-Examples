# Vision Detection: Anchor-Free FCOS & Set-Based DETR

## 1. Evolution of Object Detection: Anchor-Based to Anchor-Free to Direct Set Prediction

Traditional two-stage (Faster R-CNN) and single-stage (SSD, RetinaNet, YOLOv3) object detectors relied fundamentally on hand-crafted **anchor boxes**:
1. Thousands of predefined bounding boxes with varying aspect ratios and scales were tiled across spatial feature pyramids.
2. Positive and negative training labels were assigned via intersection-over-union (IoU) overlap thresholds against ground truth.
3. Neural networks predicted offsets $\Delta = (\Delta x, \Delta y, \Delta w, \Delta h)$ relative to each anchor.

While successful, anchor-based paradigms suffer from severe engineering and mathematical limitations:
- **Heuristic Sensitivity**: Detector accuracy is exquisitely sensitive to hyperparameter tuning of anchor scales, aspect ratios, and IoU thresholds.
- **Extreme Class Imbalance**: The vast majority of anchors belong to background, forcing complex sampling tricks (OHEM) or loss modifications (Focal Loss).
- **Post-Processing Bottleneck**: Greedy **Non-Maximum Suppression (NMS)** is mandatory to eliminate duplicate overlapping bounding boxes, introducing non-differentiable heuristics that break end-to-end backpropagation and struggle in crowded scenes.

Modern object detection eliminates anchors through two successive paradigm shifts:
1. **Anchor-Free Per-Pixel Detection (`FCOS`)**: Directly regresses bounding boxes per feature map pixel as distances to four boundaries $[l, t, r, b]$ coupled with a **centerness** objective to suppress low-quality edge detections.
2. **Direct Set Prediction Transformers (`DETR`)**: Treats object detection as a direct set-prediction problem, replacing spatial heuristics and NMS with global **bipartite matching** via the **Hungarian algorithm** and Transformer cross-attention.

---

## 2. Fully Convolutional One-Stage Object Detection (FCOS)

Let $F_i \in \mathbb{R}^{H \times W \times C}$ be the feature map at level $i$ with stride $s$. For any location $(x, y)$ on $F_i$, the corresponding coordinate on the input image is:
$$\left( \lfloor \frac{s}{2} \rfloor + x s, \ \lfloor \frac{s}{2} \rfloor + y s \right).$$

If a location $(x, y)$ falls inside a ground-truth bounding box $B_i = (x_0, y_0, x_1, y_1)$ with class $c^\star$, the regression targets are the four 4D distances from the location to the box boundaries:
$$l^\star = x - x_0, \quad t^\star = y - y_0, \quad r^\star = x_1 - x, \quad b^\star = y_1 - y.$$

### 2.1 Centerness Branch
A major failure mode of anchor-free regression is the emergence of numerous low-quality false-positive bounding boxes predicted by locations far from the center of an object. FCOS solves this by introducing a parallel branch predicting a scalar **centerness** metric $\in [0, 1]$:
$$\text{centerness}^\star = \sqrt{\frac{\min(l^\star, r^\star)}{\max(l^\star, r^\star)} \times \frac{\min(t^\star, b^\star)}{\max(t^\star, b^\star)}}.$$

During inference, the final detection score is the product of classification confidence and predicted centerness:
$$\text{Score} = \sqrt{P(\text{class}) \times \text{centerness}}.$$
Locations near object edges produce centerness values approaching $0$, naturally suppressing low-quality bounding boxes without requiring tight IoU thresholds.

---

## 3. Generalized Intersection over Union (GIoU) Loss

Standard $\ell_1$ or smooth-$\ell_1$ bounding box losses optimize coordinate offsets independently, ignoring scale differences and spatial correlations between coordinates. Furthermore, standard IoU loss:
$$\mathcal{L}_{\text{IoU}} = 1 - \frac{|A \cap B|}{|A \cup B|},$$
has zero gradient whenever two boxes do not overlap ($|A \cap B| = 0 \implies \nabla \mathcal{L} = 0$).

**Generalized IoU (GIoU)** resolves this by penalizing the non-overlapping empty space inside the smallest convex hull / enclosing box $C$ containing both boxes $A$ and $B$:
$$\text{GIoU} = \frac{|A \cap B|}{|A \cup B|} - \frac{|C \setminus (A \cup B)|}{|C|} = \text{IoU} - \frac{|C| - |A \cup B|}{|C|},$$
$$\mathcal{L}_{\text{GIoU}} = 1 - \text{GIoU} \in [0, 2].$$

Properties of GIoU:
1. When $A$ and $B$ perfectly coincide, $\text{GIoU} = 1 \implies \mathcal{L}_{\text{GIoU}} = 0$.
2. When $A$ and $B$ are completely disjoint and move infinitely far apart, $\frac{|A \cup B|}{|C|} \to 0 \implies \text{GIoU} \to -1 \implies \mathcal{L}_{\text{GIoU}} \to 2$.
3. GIoU provides smooth, non-vanishing gradients everywhere in coordinate space, driving disjoint boxes toward intersection.

---

## 4. DETR: Detection Transformers & Global Set Matching

Carion et al. (2020) introduced **DETR (DEtection TRansformer)**, completely eliminating anchors, feature pyramid spatial grids, and NMS post-processing.

DETR outputs a fixed-size set of $N$ predictions $\hat{y} = \{\hat{y}_i\}_{i=1}^N$ (where $N$ is typically $100$, significantly larger than the typical number of objects in an image). If an image contains $M < N$ objects, the ground truth is padded with $\varnothing$ (no object / background):
$$y = \{y_i\}_{i=1}^N, \quad y_i = (c_i, b_i), \quad b_i \in [0, 1]^4.$$

### 4.1 Bipartite Matching via the Hungarian Algorithm
To supervise set predictions without assuming arbitrary spatial orderings, DETR finds a permutation of $N$ elements $\sigma \in \mathfrak{S}_N$ with the lowest matching cost:
$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}).$$

The pairwise matching cost between ground-truth $y_i = (c_i, b_i)$ and prediction $\hat{y}_j = (\hat{p}_j, \hat{b}_j)$ balances class probability and box geometry:
$$\mathcal{L}_{\text{match}}(y_i, \hat{y}_j) = -\mathbf{1}_{\{c_i \ne \varnothing\}} \hat{p}_j(c_i) + \mathbf{1}_{\{c_i \ne \varnothing\}} \left( \lambda_{\text{bbox}} \|b_i - \hat{b}_j\|_1 + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}(b_i, \hat{b}_j) \right).$$

The optimal assignment $\hat{\sigma}$ is computed efficiently in polynomial time $\mathcal{O}(N^3)$ via the classical **Hungarian algorithm**.

### 4.2 Hungarian Loss
Once the optimal 1-to-1 matching $\hat{\sigma}$ is established, the model parameters are updated via standard backpropagation on the Hungarian loss:
$$\mathcal{L}_{\text{DETR}}(y, \hat{y}) = \sum_{i=1}^N \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbf{1}_{\{c_i \ne \varnothing\}} \left( \lambda_{\text{bbox}} \|b_i - \hat{b}_{\hat{\sigma}(i)}\|_1 + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right) \right].$$

Because every ground-truth object is matched to exactly *one* query prediction, DETR inherently penalizes duplicate detections, completely eliminating the need for Non-Maximum Suppression (NMS).

---

## 5. Deformable DETR & Multi-Scale Convergence Acceleration

Standard DETR suffered from two major operational limitations:
1. **Slow Training Convergence**: Requiring $500$ epochs to achieve competitive accuracy because global dense cross-attention requires hundreds of epochs to learn to look at localized object boundaries.
2. **Computational Bottleneck on High-Resolution Feature Maps**: Quadratic self-attention $\mathcal{O}((H W)^2)$ precluded feature maps with small strides, reducing small-object detection accuracy.

**Deformable DETR** (Zhu et al., 2020) resolved these bottlenecks via **Deformable Attention**:
Instead of attending across all $H \times W$ keys, each query $q$ samples only a small fixed number of $K \ll HW$ 2D offset points per attention head:
$$\operatorname{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^M W_m \sum_{k=1}^K A_{m, q, k} \cdot W_m' x(p_q + \Delta p_{m, q, k}),$$
where:
- $p_q$ is the 2D reference point of query $q$.
- $\Delta p_{m, q, k}$ are continuous 2D sampling offsets predicted by a linear projection of query feature $z_q$.
- $A_{m, q, k}$ are attention weights normalized over the $K$ points via softmax.
- Values at fractional coordinates $p_q + \Delta p$ are sampled via bilinear interpolation.

Deformable attention reduces computational complexity to $\mathcal{O}(N_{\text{query}} \cdot C)$ linear in resolution, accelerating convergence by $10\times$ while restoring full multi-scale feature pyramids.

---

## 6. End-to-End Pipeline & Modern Anchor-Free Paradigm

The complete detection paradigm is compared across classical and modern regimes:

| Architecture | Primitives | Spatial Assignment | Post-Processing | Loss Formulations |
| :--- | :--- | :--- | :--- | :--- |
| **Faster R-CNN** | Anchors ($k=9$) | Max IoU ($>0.7$) | Mandatory NMS | Smooth-$L_1$ + Cross-Entropy |
| **RetinaNet** | Dense Anchors | IoU ($>0.5$) | Mandatory NMS | Focal Loss + Smooth-$L_1$ |
| **FCOS** | Points $(x, y)$ | Ground-truth interior | Minimal NMS | Focal + GIoU + BCE Centerness |
| **DETR** | Object Queries | Hungarian Bipartite Match | **Zero (Direct Set)** | Cross-Entropy + $L_1$ + GIoU |
| **DINO / RT-DETR** | Multi-Scale Queries | Contrastive Denoising | **Zero (Real-Time)** | Focal + $L_1$ + GIoU + CDN |

---

## 7. Numerical Verification & Geometric Invariants

To guarantee exact numerical stability across computational backends:
1. **Centerness Boundedness**: Centerness targets must be strictly bounded $\in [0, 1]$. Adding $\epsilon \ge 10^{-6}$ to denominator terms prevents division by zero when an evaluated point lies exactly on an object bounding box edge ($l = 0$ or $t = 0$).
2. **GIoU Continuity**: In the case of completely disjoint boxes with infinite separation, $\text{GIoU} \to -1$. Clamping the enclosing box area $A_C \ge 10^{-6}$ prevents gradient explosion during early training when bounding boxes are randomly initialized outside the unit frame.
3. **Logit Probability Clamping in Matching**: In the Hungarian matching cost matrix, querying $-\hat{p}(c)$ directly instead of log-probabilities $-\log \hat{p}(c)$ ensures commensurable scaling with coordinate $L_1$ and GIoU terms, preventing classification cost from dominating box regression.

---

## 8. Cross-Track Verification & Architectural Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every detection operator obeys the following strict contracts:

| Operator | Verification Specification | Tolerance |
| :--- | :--- | :--- |
| **Centerness Parity** | FCOS centerness metric matches golden reference across batches | $< 10^{-5}$ |
| **GIoU Metric Invariance** | GIoU value matches analytical calculation across overlapping & disjoint boxes | $< 10^{-5}$ |
| **Bipartite Cost Matrix** | Combined class, $L_1$, and GIoU Hungarian cost matrix matches golden | $< 10^{-5}$ |
| **Coordinate Range Contract** | Predicted box coordinates $[cx, cy, w, h]$ remain valid within $[0, 1]$ | Strictly valid |
| **Scale Invariance** | GIoU invariant to uniform scaling of coordinates $(x, y) \to (\alpha x, \alpha y)$ | Exact |
