# Time Series: PatchTST, N-BEATS, Chronos, and Probabilistic Forecasting

## 1. Time-Series Representation and Distribution Drift

Time-series forecasting predicts future horizons $Y \in \mathbb{R}^{M \times H}$ conditioned on historical lookback observations $X \in \mathbb{R}^{M \times L}$, where $M$ is the number of channels (variates), $L$ is the context length, and $H$ is the prediction horizon. Real-world temporal data exhibits severe non-stationarity: mean and variance fluctuate dynamically across time due to seasonality, macroeconomic regimes, and localized anomalies. Direct application of standard neural networks causes covariate shift between training sequences and out-of-sample deployment.

## 2. Reversible Instance Normalization (RevIN)

To mitigate non-stationary distribution shift without destroying underlying seasonal and trend patterns, Reversible Instance Normalization (RevIN) introduces symmetric normalization and denormalization around model execution:

1. **Instance Normalization (Pre-processing):** Computes mean and standard deviation along the lookback temporal dimension for each instance $(b, m)$:
   $$\mu_{b, m} = \frac{1}{L} \sum_{t=1}^L X_{b, m, t}, \quad \sigma_{b, m}^2 = \frac{1}{L} \sum_{t=1}^L (X_{b, m, t} - \mu_{b, m})^2$$
   $$\hat{X}_{b, m, t} = \frac{X_{b, m, t} - \mu_{b, m}}{\sqrt{\sigma_{b, m}^2 + \epsilon}} \cdot \gamma_m + \beta_m$$

2. **Core Forecasting Model:** Processes normalized inputs $\hat{X}$ to output normalized predictions $\hat{Y} \in \mathbb{R}^{B \times M \times H}$.

3. **Reversible Denormalization (Post-processing):** Re-injects the original distribution statistics:
   $$Y_{b, m, h} = \left(\frac{\hat{Y}_{b, m, h} - \beta_m}{\gamma_m}\right) \sqrt{\sigma_{b, m}^2 + \epsilon} + \mu_{b, m}$$

## 3. Patching in Time Series: PatchTST and Channel Independence

Traditional temporal Transformers treat each individual time step as a separate token. For long lookback windows ($L \ge 512$), point-wise tokenization suffers from two critical flaws: excessive quadratic attention complexity $\mathcal{O}(L^2)$ and poor signal-to-noise ratio at single-step granularity.

PatchTST resolves both issues by aggregating consecutive time steps into subseries patches:
- Given patch length $P$ and stride $S$, the lookback sequence is unfolded into $N = \lfloor \frac{L - P}{S} \rfloor + 1$ overlapping tokens.
- **Channel Independence (CI):** Multivariate series $X \in \mathbb{R}^{B \times M \times L}$ are reshaped to $(B \cdot M) \times N \times P$. Each variate shares the same backbone network independently, eliminating spurious cross-variate correlations and parameter bloat.
- **Linear Patch Projection:** Each patch of length $P$ is projected into embedding dimension $D$: $E = X_{\text{patch}} W_{\text{proj}} + b_{\text{proj}}$.

## 4. N-BEATS Architecture: Deep Doubly Residual Stacking

N-BEATS (Neural Basis Expansion Analysis for Time Series) designs a purely MLP-based architecture that achieves exceptional forecasting accuracy through doubly residual stacking:
- Each block $\ell$ takes residual lookback $x_{\ell-1}$ and produces both a **backcast** $\hat{x}_\ell$ and a **forecast** $\hat{y}_\ell$.
- **Backcast Residual:** The input to the next block removes the portion of the signal explained by block $\ell$:
  $$x_\ell = x_{\ell-1} - \hat{x}_\ell$$
- **Forecast Accumulation:** The final prediction aggregates forecasts across all blocks:
  $$\hat{Y} = \sum_{\ell=1}^K \hat{y}_\ell$$
- Blocks can utilize generic linear expansion or specialized basis matrices for interpretable decomposition:
  - **Trend:** Polynomial basis $g_t^b(\theta) = \sum_{p=0}^P \theta_p t^p$.
  - **Seasonality:** Periodic Fourier basis $g_s^f(\theta) = \sum_{k=0}^{K-1} \left( \theta_{2k} \cos(2\pi k t / P) + \theta_{2k+1} \sin(2\pi k t / P) \right)$.

## 5. Chronos and Pretrained Foundation Models for Time Series

Chronos formulates time-series forecasting as language modeling by tokenizing continuous numerical values. Real-valued series are normalized via mean-scaling and quantized into a discrete vocabulary of $V$ bins (e.g., 4096 bins) using uniform or quantile binning:
$$q_t = \operatorname{Quantize}\left(\frac{x_t}{\bar{x}}\right) \in \{0, \dots, V-1\}$$
Standard autoregressive Transformer architectures (e.g., T5 or GPT style) are then trained via cross-entropy loss over historical token sequences to predict the distribution of future tokens.

## 6. Probabilistic Forecasting: Quantile Regression and Pinball Loss

Point forecasts fail to convey epistemic and aleatoric uncertainty, which is critical for supply chain optimization, risk hedging, and power grid scheduling. Quantile regression predicts specific percentiles $q \in (0, 1)$ (e.g., $q \in \{0.1, 0.5, 0.9\}$) of the predictive distribution without assuming Gaussianity.

The pinball (quantile) loss penalizes positive and negative prediction errors asymmetrically:

$$\mathcal{L}_q(y, \hat{y}_q) = \max\left(q(y - \hat{y}_q), (1 - q)(\hat{y}_q - y)\right)$$

When $q = 0.5$, pinball loss reduces to Mean Absolute Error (MAE), predicting the median; when $q = 0.9$, under-predicting incurs a $9\times$ higher penalty than over-predicting.

## 7. Multi-Horizon Forecasting and Continuous Ranked Probability Score (CRPS)

Evaluating probabilistic forecasts across entire horizons requires scoring rules that reward both calibration (probabilistic correctness) and sharpness (narrow confidence intervals). The Continuous Ranked Probability Score (CRPS) generalizes absolute error to arbitrary cumulative distribution functions $F$:

$$\text{CRPS}(F, y) = \int_{-\infty}^\infty \left( F(z) - \mathbb{I}(z \ge y) \right)^2 dz$$

For discrete quantile predictions across quantiles $\{q_1, \dots, q_K\}$, CRPS is approximated via the average quantile loss:

$$\text{CRPS} \approx \frac{2}{K} \sum_{k=1}^K \mathcal{L}_{q_k}(y, \hat{y}_{q_k})$$

## 8. Cross-Track Verification and Parity

All models and loss functions are verified across the 7-track curriculum against seed-42 golden safetensors:
- **PyTorch 2.x:** Reference implementation of RevIN, PatchTST unfolding, N-BEATS dual residual block, and pinball quantile loss.
- **JAX:** Functional implementation using `jnp.stack`, array slicing, and vectorised pinball loss, verified to error $< 10^{-5}$.
- **Candle:** Native Rust implementation with batched operations, dynamic patch unfolding, and dual-residual tensor addition, passing `cargo test -p ai-candle --test 07_domain_tasks test_09`.
- **WGSL Kernel:** Fused RevIN normalization compute shader computing per-instance mean/variance reduction in a single workgroup pass.
