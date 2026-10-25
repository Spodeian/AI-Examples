# Non-Contrastive and Masked SSL: BYOL, DINO, VICReg, MAE, and JEPA

## 1. Mathematical Foundations & Lineage

Self-supervised visual representation learning historically relied on negative pairs (contrastive learning) to prevent representation collapse where all inputs map to a trivial constant vector $z = c$. Beginning with Grill et al. (2020 BYOL) and Caron et al. (2021 DINO), non-contrastive methods proved that negative pairs are unnecessary when asymmetry or information-theoretic constraints are introduced.

The mathematical evolution spans four primary pillars:
1. **Asymmetric Architectures (BYOL, SimSiam)**: Use an online branch with a predictor $q_\theta$ and a target branch parameterized by an Exponential Moving Average (EMA) $\xi \leftarrow \tau \xi + (1-\tau)\theta$ or explicit stop-gradient $\text{sg}[\cdot]$. The predictor prevents trivial identity mappings while the stop-gradient breaks symmetric equilibrium collapse.
2. **Information-Theoretic De-correlation (Barlow Twins, VICReg)**: Zbontar et al. (2021) and Bardes et al. (2022) impose statistical constraints directly onto the batch representation covariance matrix, penalizing cross-feature correlations and maintaining dimension variance above a threshold $\gamma$.
3. **Self-Distillation with Centering (DINO, DINOv2)**: Caron et al. (2021) interpret SSL as knowledge distillation without labels, where a teacher network provides soft pseudo-labels sharpened by low temperature $\tau_t$ and centered by a moving bias vector $c$. Oquab et al. (2023) introduce the Kozachenko-Leonenko (`KoLeo`) differential entropy estimator to enforce maximal spread over the unit hypersphere.
4. **Masked Modeling (MAE, I-JEPA)**: He et al. (2021) reconstruct raw pixel patches from sparse unmasked tokens (25% visible), whereas Assran et al. (2023) predict abstract semantic representations directly in latent feature space, eliminating pixel-level high-frequency noise reconstruction.

---

## 2. BYOL & SimSiam Non-Contrastive Mechanics

Given two augmented views $v, v'$ of image $x$, the online network outputs representation $y_\theta = f_\theta(v)$, projection $z_\theta = g_\theta(y_\theta)$, and prediction $p_\theta = q_\theta(z_\theta)$. The target network outputs $z'_\xi = g_\xi(f_\xi(v'))$.

The normalized mean squared error loss is defined on $\ell_2$-normalized vectors:
$$\bar{p}_\theta = \frac{p_\theta}{\|p_\theta\|_2}, \quad \bar{z}'_\xi = \frac{z'_\xi}{\|z'_\xi\|_2}$$
$$\mathcal{L}_{\text{MSE}}(\bar{p}_\theta, \bar{z}'_\xi) = \|\bar{p}_\theta - \bar{z}'_\xi\|_2^2 = 2 - 2 \langle \bar{p}_\theta, \bar{z}'_\xi \rangle$$

The total symmetric BYOL objective is:
$$\mathcal{L}_{\text{BYOL}} = \mathcal{L}_{\text{MSE}}(p_\theta(v), \text{sg}[z_\xi(v')]) + \mathcal{L}_{\text{MSE}}(p_\theta(v'), \text{sg}[z_\xi(v)])$$

SimSiam (Chen & He, 2021) demonstrates that the EMA target encoder is optional: a Siamese network with an asymmetric MLP predictor $q_\theta$ and `stop_gradient` on the alternate branch prevents collapse by behaving as an expectation-maximization-like alternating optimization over cluster representations.

---

## 3. Barlow Twins & VICReg Variance-Invariance-Covariance Regularization

Rather than contrasting instances, VICReg (Bardes et al., 2022) stabilizes representations by enforcing three explicit criteria on batches of latent projections $Z, Z' \in \mathbb{R}^{B \times D}$:

### Invariance Term $s(Z, Z')$
Penalizes discrepancy between positive pairs:
$$s(Z, Z') = \frac{1}{B} \sum_{i=1}^B \|z_i - z'_i\|_2^2$$

### Variance Term $v(Z)$
Forces the standard deviation of each embedding dimension across the batch to exceed a margin $\gamma$ (typically $\gamma = 1.0$), preventing dimensional collapse:
$$S(z^j, \epsilon) = \sqrt{\frac{1}{B} \sum_{i=1}^B (z_{i,j} - \bar{z}_j)^2 + \epsilon}$$
$$v(Z) = \frac{1}{D} \sum_{j=1}^D \max(0, \gamma - S(z^j, \epsilon))$$

### Covariance Term $c(Z)$
De-correlates different embedding dimensions, driving off-diagonal covariance entries to zero to prevent information redundancy:
$$C(Z) = \frac{1}{B - 1} \sum_{i=1}^B (z_i - \bar{z})(z_i - \bar{z})^T \in \mathbb{R}^{D \times D}$$
$$c(Z) = \frac{1}{D} \sum_{i \neq j} [C(Z)]_{i,j}^2$$

The composite loss function combines the three terms with balance coefficients:
$$\mathcal{L}_{\text{VICReg}} = \lambda s(Z, Z') + \mu [v(Z) + v(Z')] + \nu [c(Z) + c(Z')]$$

---

## 4. DINO Self-Distillation & KoLeo Differential Entropy

DINO (Caron et al., 2021) casts SSL as learning without labels via student-teacher distillation. Given student parameters $\theta$ and teacher parameters $\theta_t$ updated via EMA:
$$P_s(x)^{(k)} = \frac{\exp(g_\theta(x)^{(k)} / \tau_s)}{\sum_{j=1}^K \exp(g_\theta(x)^{(j)} / \tau_s)}$$
$$P_t(x)^{(k)} = \frac{\exp((g_{\theta_t}(x)^{(k)} - c^{(k)}) / \tau_t)}{\sum_{j=1}^K \exp((g_{\theta_t}(x)^{(j)} - c^{(j)}) / \tau_t)}$$

where the centering vector $c$ is updated as $c \leftarrow m c + (1-m) \frac{1}{B} \sum_{i=1}^B g_{\theta_t}(x_i)$. Centering prevents one dimension from dominating (mode collapse), while sharpening ($\tau_t < \tau_s$) prevents uniform collapse.

### Kozachenko-Leonenko (`KoLeo`) Regularizer
In DINOv2 (Oquab et al., 2023), student features $z \in \mathbb{R}^{B \times D}$ normalized to the unit sphere $\mathbb{S}^{D-1}$ are regularized by the Kozachenko-Leonenko nearest-neighbor differential entropy estimator:
$$\mathcal{L}_{\text{KoLeo}} = - \frac{1}{B} \sum_{i=1}^B \log \left( \min_{j \neq i} \|z_i - z_j\|_2 + \epsilon \right)$$
Minimizing $\mathcal{L}_{\text{KoLeo}}$ pushes each sample away from its nearest neighbor, maximizing the coverage and uniformity of the latent manifold.

---

## 5. Masked Autoencoding (MAE) & Latent Prediction (I-JEPA)

Masked image modeling trains vision encoders to predict missing information from partial observations:
- **MAE (He et al., 2021)**: Images are partitioned into regular non-overlapping patches (e.g., $16 \times 16$). A high masking ratio (75%) removes three-quarters of the patches. Crucially, the heavy ViT encoder processes **only the 25% visible patches**. Mask tokens are introduced only at the shallow, lightweight decoder to reconstruct pixel values under MSE loss.
- **I-JEPA (Assran et al., 2023)**: Pixel reconstruction forces models to allocate capacity to imperceptible high-frequency textures. Image-based Joint-Embedding Predictive Architecture (I-JEPA) predicts target representations directly in latent space from context patches:
$$\mathcal{L}_{\text{I-JEPA}} = \frac{1}{M \cdot D} \sum_{m=1}^M \|\hat{s}_y^{(m)} - s_y^{(m)}\|_2^2$$
Target representations $s_y$ are produced by an EMA target encoder without decoding back to raw pixels.

---

## 6. Cross-Framework Implementations & Numerical Parity

Every algorithm in this topic is verified against golden tensors stored in `02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.safetensors` across PyTorch 2.x, JAX, and Candle.

Key verified operations:
1. `linear_anchor`: Multi-framework linear mapping validation.
2. `byol_prediction_loss`: Normalized cosine similarity loss $2 - 2 \langle \bar{p}, \bar{z} \rangle$.
3. `vicreg_loss`: Invariance MSE + standard deviation ReLU hinge + off-diagonal covariance penalty.
4. `dino_loss`: Centered and sharpened cross-entropy distillation.
5. `koleo_loss`: Kozachenko-Leonenko nearest-neighbor log-distance entropy.
6. `mae_patch_gather`: Efficient gathering of unmasked patches.
7. `jepa_feature_loss`: Latent space MSE predictor loss.

All implementations match across frameworks with maximum absolute difference $< 10^{-5}$.

---

## 7. Hardware Acceleration & WGSL Compute Shaders

Pairwise distances and normalized cosine projections frequently bottleneck SSL training loops. In `kernels/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.rs`, we provide `BYOL_COSINE_SIM_WGSL`.

The compute shader operates on row vectors $p, z \in \mathbb{R}^{B \times D}$:
```wgsl
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= cfg.b) {
        return;
    }
    let offset = i * cfg.d;
    var dot: f32 = 0.0;
    var p_sq: f32 = 0.0;
    var z_sq: f32 = 0.0;
    for (var k: u32 = 0u; k < cfg.d; k = k + 1u) {
        let pv = p[offset + k];
        let zv = z[offset + k];
        dot = dot + pv * zv;
        p_sq = p_sq + pv * pv;
        z_sq = z_sq + zv * zv;
    }
    let p_norm = sqrt(p_sq + 1e-12);
    let z_norm = sqrt(z_sq + 1e-12);
    loss_vec[i] = 2.0 - 2.0 * (dot / (p_norm * z_norm));
}
```
This fused kernel computes $\ell_2$ normalization, dot product, and scalar loss in a single memory pass, eliminating three intermediate GPU buffer allocations.

---

## 8. Performance Profiling, Anti-Patterns & Misuse Guardrails

### Performance Guardrails
1. **MAE Visible-Only Encoding**: Never pass `[MASK]` tokens into the heavy ViT encoder. Processing only visible tokens yields a $4\times$ reduction in encoder sequence length and an approximate $16\times$ reduction in self-attention FLOPs ($O(L^2)$).
2. **VICReg Covariance Memory Efficiency**: Computing the full $D \times D$ covariance matrix scales quadratically with feature dimension. For large $D$ (e.g., $D > 8192$), sample or project dimensions prior to computing covariance.

### Misuse Guardrails & Anti-Patterns
1. **SimSiam Collapse via Missing Stop-Gradient**: Removing `stop_gradient` from target projections in SimSiam immediately triggers dimensional collapse to a trivial constant within 10 epochs.
2. **DINO Centering / Sharpening Imbalance**: Setting $\tau_t \ge \tau_s$ causes entropy collapse to a uniform distribution. Conversely, omitting the centering buffer $c$ causes representation collapse to a single coordinate dimension.
3. **KoLeo Self-Distance Zero Penalty**: Forgetting to mask the diagonal $i = j$ in the pairwise distance matrix causes $\min_{j} D_{ij} = 0$, leading to $\log(0) \to -\infty$ and numerical explosion.
