# Audio DSP, Spectrograms, Conformer, CTC, and Neural Vocoders

## 1. Audio DSP Fundamentals and Short-Time Fourier Transform (STFT)

Speech and raw audio waveforms are continuous 1D physical oscillations sampled at discrete intervals (e.g., 16 kHz or 24 kHz). Direct time-domain modeling poses significant challenges due to high temporal sampling density and non-stationary spectral dynamics. The foundational representation in speech processing is the Short-Time Fourier Transform (STFT), which slices continuous waveforms into overlapping frames via windowing functions (e.g., Hann or Hamming):

$$x_m[n] = x[m H + n] w[n], \quad 0 \le n < N$$

where $H$ is the hop size (frame shift), $N$ is the window length, and $w[n]$ is the Hann analysis window. Applying the Discrete Fourier Transform (DFT) to each framed slice yields the complex spectrogram:

$$X[m, k] = \sum_{n=0}^{N-1} x_m[n] \exp\left(-j \frac{2\pi k n}{N}\right)$$

The magnitude spectrum $|X[m, k]|$ captures instantaneous frequency power across time frames $m$ and frequency bins $k \in [0, N/2]$.

## 2. Mel Scale Filterbanks and Log-Mel Spectrogram Extraction

Human auditory perception is non-linear across frequencies: pitch discrimination is fine-grained at low frequencies (< 1 kHz) and logarithmic at higher frequencies. The Mel scale models this psychoacoustic response through triangular overlapping bandpass filterbanks:

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

Given a precomputed Mel filterbank matrix $W_{\text{mel}} \in \mathbb{R}^{M \times F}$ (where $M$ is the number of mel bins, e.g., 80, and $F = N/2 + 1$ is the number of linear frequency bins) and magnitude spectrogram $S \in \mathbb{R}^{B \times F \times T}$:

$$\text{MelSpec} = W_{\text{mel}} S$$

Dynamic range compression via logarithmic transformation yields the log-mel spectrogram:

$$\text{LogMel}[b, m, t] = \log(\text{MelSpec}[b, m, t] + \epsilon)$$

Log-mel spectrograms serve as universal acoustic features for automatic speech recognition (ASR), speaker verification, and text-to-speech (TTS).

## 3. The Conformer Architecture: Marrying Attention with Convolutions

Conformer (Convolution-augmented Transformer) merges the global context modeling of Multi-Head Self-Attention (MHSA) with the local receptive field and translation equivariance of depthwise separable convolutions. Standard Transformers capture long-range dependencies across utterance frames but struggle with fine-grained phonetic transitions; standard CNNs excel at local spectral feature extraction but scale poorly across entire conversations. Conformer achieves state-of-the-art acoustic modeling by organizing operations in a Macaron-style dual feed-forward network sandwich:

$$\tilde{x}_i = x_i + \frac{1}{2} \text{FFN}(x_i)$$
$$x'_i = \tilde{x}_i + \text{MHSA}(\tilde{x}_i)$$
$$x''_i = x'_i + \text{Conv}(x'_i)$$
$$y_i = \text{LayerNorm}\left(x''_i + \frac{1}{2} \text{FFN}(x''_i)\right)$$

## 4. Conformer Convolution Module and Depthwise Convolutions

The Conformer Convolution module processes intermediate representations $X \in \mathbb{R}^{B \times T \times D}$ through a specialized sequence designed for maximum gradient flow and expressivity:

1. **Pre-LayerNorm:** Normalizes hidden activations across the channel dimension $D$.
2. **Pointwise Conv & Gated Linear Unit (GLU):** Projects $D \to 2D$, followed by gating $A \odot \sigma(B)$ to modulate feature transmission.
3. **1D Depthwise Convolution:** Groups=$D$ depthwise convolution with kernel size $K=31$ (or $K=3$ in compact setups) and symmetric padding. Each channel is filtered independently without cross-channel mixing:
   $$y[b, t, c] = \sum_{k=0}^{K-1} x_{\text{padded}}[b, t+k, c] w_{\text{dw}}[c, 0, k] + b_{\text{dw}}[c]$$
4. **Batch/Layer Normalization & SiLU Activation:** Smooth, non-monotonic activation $z \cdot \sigma(z)$.
5. **Pointwise Conv 2 & Residual:** Projects back $D \to D$ and sums with the input skip connection.

## 5. Connectionist Temporal Classification (CTC) Loss and Trellis

In speech recognition, frame-level alignments between audio frames $T$ and transcript characters $U$ are unknown, and speech rate varies dramatically across speakers. Connectionist Temporal Classification (CTC) introduces a blank token $\epsilon$ and maps an alignment path $\pi = (\pi_1, \dots, \pi_T)$ to target transcription $Y$ via collapse operator $\mathcal{B}$, which merges consecutive duplicate symbols and deletes blanks.

The total probability of target sequence $Y$ is the sum over all valid paths:

$$P(Y | X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi | X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^T y_{\pi_t}^t$$

To compute this efficiently in $\mathcal{O}(T \cdot U)$, CTC constructs a forward lattice $\alpha_t(s)$ over the blank-interleaved target sequence $\tau$ of length $S = 2U + 1$:

$$\alpha_t(s) = \left[\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \mathbb{I}(\tau_s \ne \epsilon \land \tau_s \ne \tau_{s-2}) \alpha_{t-1}(s-2)\right] \cdot P(X_t = \tau_s)$$

The CTC loss is the negative log-likelihood of the full lattice: $\mathcal{L}_{\text{CTC}} = -\log P(Y | X)$.

## 6. Neural Vocoders and Speech Synthesis

A neural vocoder synthesizes raw high-fidelity time-domain audio $y \in \mathbb{R}^{T_{\text{audio}}}$ conditioned on acoustic representations such as log-mel spectrograms. Modern architectures like HiFi-GAN and BigVGAN employ transposed 1D convolutions with Multi-Receptive Field Fusion (MRF) modules, operating in tandem with adversarial discriminators:
- **Multi-Period Discriminator (MPD):** Evaluates 1D audio reshaped into 2D matrices across prime periods ($p \in \{2, 3, 5, 7, 11\}$) to capture pitch periodicities.
- **Multi-Scale Discriminator (MSD):** Evaluates audio across multiple temporal resolutions via average pooling to enforce global structural coherence.

## 7. Multi-Resolution STFT (MR-STFT) Loss and Spectral Convergence

Training vocoders solely with adversarial objectives yields instability and artifacts. Multi-Resolution STFT (MR-STFT) loss stabilizes training by penalizing differences across multiple STFT analysis configurations (varying window lengths and hop sizes). Each STFT resolution combines two complementary terms:

1. **Spectral Convergence Loss:**
   $$\mathcal{L}_{\text{sc}}(Y, \hat{Y}) = \frac{\| |Y| - |\hat{Y}| \|_F}{\| |Y| \|_F + \epsilon} = \frac{\sqrt{\sum (|Y| - |\hat{Y}|)^2}}{\sqrt{\sum |Y|^2} + \epsilon}$$
2. **Log STFT Magnitude Loss:**
   $$\mathcal{L}_{\text{mag}}(Y, \hat{Y}) = \frac{1}{N} \sum_{m, k} \left| \log(|Y[m, k]| + \epsilon) - \log(|\hat{Y}[m, k]| + \epsilon) \right|$$

$$\mathcal{L}_{\text{MR-STFT}} = \sum_{s=1}^S \left( \mathcal{L}_{\text{sc}}^{(s)} + \mathcal{L}_{\text{mag}}^{(s)} \right)$$

This combined loss enforces both sharp spectral energy distribution and accurate low-energy harmonic preservation.

## 8. Cross-Track Verification and Parity

All mathematical components across the 7-track curriculum are verified against seed-42 golden safetensors:
- **PyTorch 2.x:** Reference implementation of Mel filterbank projection, Conformer convolution module, CTC dynamic programming forward lattice, and MR-STFT loss. Verified against PyTorch native `F.ctc_loss` with absolute error $< 10^{-6}$.
- **JAX:** Functional implementation utilizing `jnp.einsum`, vectorized depthwise convolution taps, and DP trellis logsumexp, achieving $< 10^{-5}$ numerical parity.
- **Candle:** Native Rust implementation with batched matrix multiplications, LayerNorm normalization, and dynamic programming CTC lattice, verified via `cargo test -p ai-candle --test 07_domain_tasks test_08`.
- **WGSL Kernel:** Fused compute shader executing parallel Log-Mel filterbank extraction with shared memory and uniform configuration buffers.
