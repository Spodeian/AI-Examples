# Calibration Observers: MinMax, Moving Average, Percentile, MSE, KL Divergence, and ACIQ

Post-Training Quantization (PTQ) converts floating-point weights and activation tensors to low-precision integer representations without full retraining. While weights are fixed and can be inspected statically, activations depend on runtime input data and vary widely in dynamic range. Calibration observers pass representative calibration datasets through the model to monitor activation statistics and select optimal clipping boundaries $[-\alpha, \alpha]$ (or $[x_{\min}, x_{\max}]$). The fundamental objective of calibration is to find the boundary $\alpha$ that optimally balances clipping distortion against rounding (granularity) distortion.

---

## 1. The Clipping vs. Rounding Noise Tradeoff

For symmetric $b$-bit quantization with quantization step size $\Delta = \frac{\alpha}{q_{\max}}$ where $q_{\max} = 2^{b-1} - 1$, the total expected distortion $E = \mathbb{E}[(X - \hat{X})^2]$ decomposes into two opposing noise components:

$$
E(\alpha) = \underbrace{2 \int_{\alpha}^{\infty} (x - \alpha)^2 p_X(x) \, dx}_{E_{\text{clip}}(\alpha) \text{ (Clipping / Saturation Noise)}} + \underbrace{\int_{-\alpha}^{\alpha} (x - Q(x))^2 p_X(x) \, dx}_{E_{\text{round}}(\alpha) \text{ (Rounding / Granularity Noise)}}
$$

- **High $\alpha$ (Under-clipping)**: Prevents clipping noise ($E_{\text{clip}} \to 0$), but inflates step size $\Delta$, increasing rounding variance $\approx \frac{\Delta^2}{12}$.
- **Low $\alpha$ (Over-clipping)**: Reduces rounding noise ($\Delta \to 0$), but abruptly saturates tail activations, causing catastrophic clipping distortion.

---

## 2. MinMax and Moving Average Observers

The simplest observer tracks absolute extreme values:

$$
\alpha_{\text{MinMax}} = \max_{i} |x_i|
$$

While zero clipping distortion occurs on calibration samples, single outlier activations inflate $\alpha$, severely coarsening the quantization grid across the remaining $99.9\%$ of values.

For mini-batch calibration over $T$ steps, the **MovingAverageMinMax** observer dampens batch-to-batch variance using momentum $c \in (0, 1]$:

$$
\alpha_t = (1 - c) \cdot \alpha_{t-1} + c \cdot \max_{i} |x_{t, i}|
$$

This smooths transient spikes during calibration on small batch sizes.

---

## 3. Percentile and Histogram Filtering

The **Percentile Observer** discards extreme tail activations by selecting $\alpha$ as the $p$-th percentile of absolute values $|X|$:

$$
\alpha_p = \inf \{ \alpha \in \mathbb{R} : F_{|X|}(\alpha) \ge p \}
$$

Standard choices:
- $p = 99.9\%$ or $99.99\%$ for robust activation quantization in convolutional networks.
- $p = 95.0\%$ when aggressive clipping of heavy-tailed attention projection layers is required.

---

## 4. Mean Squared Error (MSE) Grid Search

The **MSE Observer** explicitly minimizes empirical $L_2$ reconstruction error over a candidate grid of thresholds:

$$
\alpha^* = \arg\min_{\alpha \in \mathcal{A}} \frac{1}{N} \sum_{i=1}^N \left( x_i - \text{clamp}\left( \left\lfloor \frac{x_i \cdot q_{\max}}{\alpha} \right\rceil, -q_{\max}, q_{\max} \right) \cdot \frac{\alpha}{q_{\max}} \right)^2
$$

Evaluating candidates $\alpha_k = \alpha_{\max} \cdot (0.55 + 0.05 \cdot k)$ across $k \in \{0, \dots, 9\}$ reliably discovers the global minimum of the convex $E(\alpha)$ curve in $\mathcal{O}(K \cdot N)$ operations.

---

## 5. TensorRT Entropy Calibration (KL Divergence)

NVIDIA TensorRT uses **Entropy Calibration**, which treats the unquantized and quantized representations as discrete probability distributions $P$ and $Q$, choosing the threshold that minimizes the Kullback-Leibler (KL) divergence (relative entropy):

$$
D_{\text{KL}}(P \parallel Q) = \sum_{i=0}^{T-1} P(i) \log \left( \frac{P(i)}{Q(i)} \right)
$$

### Algorithm Steps:
1. Construct high-resolution histogram of $|X|$ over $L = 64$ (or 2048) bins.
2. For each candidate threshold $T \in [K, L]$ (where $K = 2^{b-1}$ is the target quantized bins):
   - Truncate histogram to $T$ bins; accumulate all tail bins $i \ge T$ into bin $T-1$.
   - Quantize $P$ into $K$ uniform bins, then expand back to length $T$ by uniformly redistributing mass over positive bins to construct reference $Q$.
   - Smooth $P$ and $Q$ with $\epsilon = 10^{-7}$ and normalize to sum to 1.
   - Compute $D_{\text{KL}}(P \parallel Q)$.
3. Select $T^*$ yielding minimum divergence; threshold is $\alpha^* = \frac{T^*}{L} \cdot \alpha_{\max}$.

---

## 6. Analytical Clipping for Integer Quantization (ACIQ)

Banner et al. (NeurIPS 2019) derived closed-form expressions for the optimal clipping threshold $\alpha^* = \beta \cdot \sigma$ by solving $\frac{\partial E(\alpha)}{\partial \alpha} = 0$ analytically.

### Optimal Clipping Factors $\beta$:

| Bit-width ($b$) | Quantization Levels ($M$) | Gaussian $\beta_{\text{Gauss}}$ | Laplace $\beta_{\text{Laplace}}$ |
| :--- | :--- | :--- | :--- |
| **2-bit** | 4 | 1.7061 | 1.83 |
| **4-bit** | 16 | 2.8279 | 3.89 |
| **8-bit** | 256 | 3.9224 | 7.04 |

For zero-mean distributions:
- **Gaussian**: $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}$, yielding $\alpha^* = \beta_{\text{Gauss}} \cdot \sigma$.
- **Laplace**: $b = \frac{1}{N} \sum_{i=1}^N |x_i|$, yielding $\alpha^* = \beta_{\text{Laplace}} \cdot b$.

ACIQ requires zero histogram binning, zero grid search iterations, and executes in single-pass $\mathcal{O}(N)$ time.

---

## 7. Cross-Framework Numerical Parity

All 6 calibration observer strategies are verified against a deterministic seed-42 golden reference:

| Framework | Implementation File | Precision Parity |
| :--- | :--- | :--- |
| **PyTorch 2.x** | `pytorch2/14_quantisation/01_foundations_formats_and_observers/04_...py` | Reference |
| **JAX** | `jax/14_quantisation/01_foundations_formats_and_observers/04_...py` | $< 10^{-5}$ |
| **Candle** | `candle/14_quantisation/01_foundations_formats_and_observers/04_...rs` | $< 10^{-5}$ |
| **Kernels** | `kernels/14_quantisation/01_foundations_formats_and_observers/04_...rs` | WGSL / CPU |

---

## 8. Production Deployment & PTQ Pipelines

In modern deployment pipelines (ONNX Runtime, TensorRT, TorchAO):
1. **Calibration Sample Selection**: 128 to 512 representative domain inputs suffice to stabilize observer statistics without overfitting.
2. **Layer-Wise Observer Assignment**:
   - Weights: Static MinMax or MSE observer.
   - Dense MLP Activations: ACIQ Gaussian or MSE observer.
   - Attention Projections & Softmax: Percentile (99.9%) or KL entropy observer to suppress outlier activation heads.
